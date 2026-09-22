"""HSV threshold + connected components.

For an aim trainer this beats a neural detector on both axes: the targets are a flat
colour you chose yourself, so a threshold is *exact*, and it costs well under a
millisecond on a 320x320 ROI. The shape filters exist to reject the things that share
that colour by accident -- UI text, hit-marker particles, the crosshair itself.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..config import DetectorConfig, HSVRange
from .base import Detection


def build_mask(frame_bgr: np.ndarray, ranges: Sequence[HSVRange]) -> np.ndarray:
    """Union of the configured HSV bands as a uint8 0/255 mask.

    A band with ``h_min > h_max`` wraps around the hue circle, which is how you
    express red (``h_min=170, h_max=10``) in OpenCV's 0-179 hue space.
    """
    if not ranges:
        raise ValueError("at least one HSV range is required")
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        sv_lo = (r.s_min, r.v_min)
        sv_hi = (r.s_max, r.v_max)
        if r.h_min <= r.h_max:
            part = cv2.inRange(hsv, (r.h_min, *sv_lo), (r.h_max, *sv_hi))
        else:
            lo = cv2.inRange(hsv, (r.h_min, *sv_lo), (179, *sv_hi))
            hi = cv2.inRange(hsv, (0, *sv_lo), (r.h_max, *sv_hi))
            part = cv2.bitwise_or(lo, hi)
        mask = cv2.bitwise_or(mask, part)
    return mask


def clean_mask(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
    """Opening then closing: drop speckle, then re-fill holes.

    Order matters. Closing first would merge a speck into a target and inflate it;
    opening first removes the speck, and the close then only heals genuine holes --
    the crosshair sitting on top of a target, or a hit-marker overlay.
    """
    if open_kernel >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_kernel >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


class ColorDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        if not cfg.hsv_ranges:
            raise ValueError("detector.hsv_ranges is empty; nothing would ever match")

    def describe(self) -> str:
        bands = ", ".join(f"H{r.h_min}-{r.h_max}" for r in self.cfg.hsv_ranges)
        return f"color (HSV {bands})"

    def mask_for(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Exposed so ``preview`` can show exactly what the detector thresholds."""
        return clean_mask(
            build_mask(frame_bgr, self.cfg.hsv_ranges),
            self.cfg.open_kernel,
            self.cfg.close_kernel,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        mask = self.mask_for(frame_bgr)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        out: list[Detection] = []
        for i in range(1, count):  # label 0 is the background
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.cfg.min_area_px or area > self.cfg.max_area_px:
                continue
            if w == 0 or h == 0:
                continue
            if area / float(w * h) < self.cfg.min_extent:
                continue
            aspect = max(w / h, h / w)
            if aspect > self.cfg.max_aspect:
                continue

            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            # Tall blobs are humanoid-ish rather than spherical; bias the aim point
            # up into the upper third. For a round target this is a no-op.
            if h > 1.6 * w:
                cy = y + h / 3.0
            # Score by fill quality, so a clean disc outranks a ragged smear of the
            # same size when the selector has to break a tie.
            out.append(
                Detection(
                    cx=cx,
                    cy=cy,
                    width=float(w),
                    height=float(h),
                    score=min(1.0, area / float(w * h) / 0.785),
                )
            )
        return out
