"""Visualise what the detector sees. No input is ever sent from here.

This is where colour tuning happens: the mask panel shows exactly what the threshold
keeps, so a noisy or hollow mask is obvious rather than something you infer from the
bot missing shots.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .capture import build_capture, centered_region
from .config import Config
from .control import AimController
from .detectors import build_detector
from .runner import view_model
from .tracking import Tracker


def annotate(frame, detections, tracks, selected_id, center) -> np.ndarray:
    out = frame.copy()
    cx, cy = int(center[0]), int(center[1])
    cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)

    for d in detections:
        x0, y0 = int(d.cx - d.width / 2), int(d.cy - d.height / 2)
        cv2.rectangle(out, (x0, y0), (x0 + int(d.width), y0 + int(d.height)), (90, 90, 90), 1)

    for tr in tracks:
        chosen = tr.track_id == selected_id
        color = (60, 240, 60) if chosen else (200, 160, 40)
        cv2.circle(out, (int(tr.cx), int(tr.cy)), 4, color, -1)
        cv2.putText(out, f"#{tr.track_id}", (int(tr.cx) + 7, int(tr.cy) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        if chosen:
            cv2.line(out, (cx, cy), (int(tr.cx), int(tr.cy)), color, 1)
            # Velocity arrow, scaled to 100 ms of travel, so lead is visible.
            if tr.has_velocity:
                px, py = tr.predict(0.1)
                cv2.arrowedLine(out, (int(tr.cx), int(tr.cy)), (int(px), int(py)), (40, 200, 255), 1, tipLength=0.3)
    return out


def side_by_side(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame
    return np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])


def run_preview(cfg: Config, frames: int = 0, save_dir: str | None = None) -> int:
    region = centered_region(cfg.capture)
    capture = build_capture(cfg.capture, region)
    detector = build_detector(cfg.detector)
    tracker = Tracker(cfg.tracking)
    controller = AimController(cfg.aim, cfg.trigger, view_model(cfg))

    out_dir = Path(save_dir) if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"capture : {capture.describe()}")
    print(f"detector: {detector.describe()}")
    print("no input is sent in preview mode. q or ctrl-c to stop.\n")

    headless = False
    i = 0
    try:
        while frames <= 0 or i < frames:
            frame = capture.grab()
            if frame is None:
                continue
            now = time.perf_counter()
            detections = list(detector.detect(frame))
            tracks = tracker.update(detections, now)
            controller.step(tracks, region.center, now)

            mask = detector.mask_for(frame) if hasattr(detector, "mask_for") else None
            canvas = side_by_side(
                annotate(frame, detections, tracks, controller.selector.current_id, region.center),
                mask,
            )

            if out_dir:
                cv2.imwrite(str(out_dir / f"preview_{i:05d}.png"), canvas)
            if not headless:
                try:
                    cv2.imshow("aimtrainer preview (left: detections, right: mask)", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    # opencv-python-headless has no HighGUI. Keep going and save instead.
                    headless = True
                    if out_dir is None:
                        out_dir = Path("captures")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        print(f"no display available; writing frames to {out_dir}/")
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
        if not headless:
            cv2.destroyAllWindows()

    print(f"{i} frames previewed")
    return 0
