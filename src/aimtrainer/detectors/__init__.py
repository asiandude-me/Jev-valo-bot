"""Target detectors. All of them return :class:`Detection` in ROI pixel coordinates."""

from __future__ import annotations

from ..config import DetectorConfig
from .base import Detection, Detector


def build_detector(cfg: DetectorConfig) -> Detector:
    if cfg.backend == "color":
        from .color import ColorDetector

        return ColorDetector(cfg)
    if cfg.backend == "onnx":
        from .onnx_yolo import OnnxYoloDetector

        return OnnxYoloDetector(cfg)
    raise ValueError(f"unknown detector backend {cfg.backend!r} (expected 'color' or 'onnx')")


__all__ = ["Detection", "Detector", "build_detector"]
