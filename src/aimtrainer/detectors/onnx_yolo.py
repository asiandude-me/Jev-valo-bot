"""YOLO-family detector running locally through ONNX Runtime.

Bring your own weights: export any Ultralytics YOLO (v5/v8/v11) to ONNX with
``tools/export_yolo_onnx.py`` and point ``detector.model_path`` at it. Nothing here
is network-bound -- ONNX Runtime picks the best execution provider your machine
actually has (TensorRT > CUDA > DirectML > CPU) and everything stays on-device.

Output layouts differ between exports, so both of the common ones are handled:
``(1, 4+nc, N)`` (v8/v11) and ``(1, N, 5+nc)`` (v5, which carries an extra objectness
column ahead of the class scores).

Which axis is which is decided from the *values* -- see :func:`_layout_score`. Whether
there is an objectness column is genuinely ambiguous from one frame's numbers, since
``nc`` is not known ahead of time, so ``detector.layout`` can state it outright and
only falls back to a heuristic when left on ``auto``.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from ..config import DetectorConfig
from .base import Detection


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio and pad to a square.

    Returns the padded image plus the scale and offsets needed to map boxes back to
    the original frame. Stretching instead would be cheaper but distorts the aspect
    ratio the model was trained on, and costs more accuracy than the pad costs time.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=image.dtype)
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas, scale, dx, dy


def _layout_score(arr: np.ndarray) -> float:
    """How plausible is it that ``arr`` is already ``(N, 4 + num_classes)``?

    Judged on content rather than on shape. Comparing the two axis lengths is the
    obvious test and it is wrong: it only works when the model returns far more
    candidate boxes than channels, which fails for any model or any frame that
    yields a handful of detections. What is reliable is that class scores are
    sigmoid outputs in [0, 1] while box coordinates are pixels spanning tens or
    hundreds -- so the correct orientation is the one whose trailing columns look
    like probabilities and whose leading four look like coordinates.
    """
    if arr.ndim != 2 or arr.shape[1] < 5:
        return -1.0
    scores = arr[:, 4:]
    if scores.size == 0:
        return -1.0
    score = float(np.mean((scores >= -0.01) & (scores <= 1.01)))
    if arr[:, :4].size and float(np.ptp(arr[:, :4])) > 1.5:
        score += 0.25
    # Tie-break toward the common case of more detections than channels, which is
    # what decides an all-zero (no-detection) frame.
    if arr.shape[0] >= arr.shape[1]:
        score += 0.1
    return score


def _normalize_predictions(raw: np.ndarray) -> np.ndarray:
    """Collapse any of the known output layouts to ``(N, 4 + num_classes)``."""
    pred = raw[0] if raw.ndim == 3 else raw
    if pred.ndim != 2:
        raise ValueError(f"unexpected model output shape {raw.shape}")

    as_is, transposed = _layout_score(pred), _layout_score(pred.T)
    if max(as_is, transposed) < 0:
        raise ValueError(
            f"model output {raw.shape} does not look like YOLO detections; expected "
            "(1, 4+nc, N) or (1, N, 4+nc) with at least 5 channels"
        )
    return pred if as_is >= transposed else pred.T


class OnnxYoloDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            import onnxruntime as ort  # deferred: only this backend needs it
        except ImportError as exc:
            raise RuntimeError(
                "the onnx detector backend needs onnxruntime: "
                "pip install 'aimtrainer-bot[onnx]' (or onnxruntime-gpu / "
                "onnxruntime-directml for GPU inference)"
            ) from exc

        path = Path(cfg.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"model not found at {path}. Export one with "
                f"`python tools/export_yolo_onnx.py --weights yolov8n.pt --imgsz "
                f"{cfg.input_size} --out {path}`"
            )

        available = set(ort.get_available_providers())
        providers = [p for p in cfg.providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # One thread per stage beats parallelism here: the model is small and the loop
        # is latency-bound, so thread hand-off costs more than it saves.
        options.intra_op_num_threads = max(1, (__import__("os").cpu_count() or 4) // 2)

        if cfg.layout not in ("auto", "v8", "v5"):
            raise ValueError(f"detector.layout must be auto, v8 or v5 (got {cfg.layout!r})")

        self.cfg = cfg
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.providers = self.session.get_providers()
        self._dtype = (
            np.float16
            if "float16" in self.session.get_inputs()[0].type
            else np.float32
        )

    def describe(self) -> str:
        return (
            f"onnx {Path(self.cfg.model_path).name} @ {self.cfg.input_size}px "
            f"on {self.providers[0]} (layout {self.cfg.layout})"
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        size = self.cfg.input_size
        padded, scale, dx, dy = letterbox(frame_bgr, size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(self._dtype) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

        raw = self.session.run(None, {self.input_name: blob})[0]
        pred = _normalize_predictions(np.asarray(raw, dtype=np.float32))

        boxes = pred[:, :4]
        rest = pred[:, 4:]
        if rest.shape[1] == 0:
            return []
        # v5 exports carry an objectness column ahead of the class scores; fold it in
        # so a confident box of an uncertain class is scored honestly.
        if self.cfg.layout == "v5":
            has_objectness = rest.shape[1] > 1
        elif self.cfg.layout == "v8":
            has_objectness = False
        else:
            has_objectness = rest.shape[1] > 1 and _looks_like_objectness(rest)
        scores_all = rest[:, 1:] * rest[:, :1] if has_objectness else rest

        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(class_ids)), class_ids]

        keep = scores >= self.cfg.conf_threshold
        if self.cfg.class_ids:
            keep &= np.isin(class_ids, self.cfg.class_ids)
        if not keep.any():
            return []
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

        # xywh (letterboxed) -> xyxy (original frame)
        cx = (boxes[:, 0] - dx) / scale
        cy = (boxes[:, 1] - dy) / scale
        bw = boxes[:, 2] / scale
        bh = boxes[:, 3] / scale
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)

        idx = cv2.dnn.NMSBoxes(
            xyxy.tolist(),
            scores.astype(np.float32).tolist(),
            self.cfg.conf_threshold,
            self.cfg.nms_threshold,
        )
        if len(idx) == 0:
            return []

        out = []
        for i in np.asarray(idx).reshape(-1):
            # Aim at the upper third: for a humanoid target that is the head, and for
            # a sphere the box is square so it barely moves the point.
            out.append(
                Detection(
                    cx=float(cx[i]),
                    cy=float(cy[i] - bh[i] / 2 + bh[i] / 3) if bh[i] > 1.6 * bw[i] else float(cy[i]),
                    width=float(bw[i]),
                    height=float(bh[i]),
                    score=float(scores[i]),
                    class_id=int(class_ids[i]),
                )
            )
        return out


def _looks_like_objectness(rest: np.ndarray) -> bool:
    """Guess whether column 0 is a v5 objectness score.

    A v5 objectness column is in [0, 1] and, averaged over a model's full candidate
    set (typically thousands of rows, most of them background), sits well above the
    per-class columns beside it. That separation is what this measures.

    It is a heuristic and it cannot be otherwise: with one row and no knowledge of the
    class count, ``[obj=1.0, cls=0.9]`` and ``[cls0=1.0, cls1=0.9]`` are the same
    numbers. Set ``detector.layout`` explicitly if your model's scores come out wrong.
    """
    first, others = rest[:, 0], rest[:, 1:]
    if first.min() < 0.0 or first.max() > 1.0:
        return False
    return float(first.mean()) > float(others.mean()) * 1.5
