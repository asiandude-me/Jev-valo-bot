"""Synthetic Aim Labs-like frames.

Lets the whole pipeline be benchmarked and regression-tested with no display, no game
and no Windows -- and, because the ground truth is known exactly, it also measures
detector recall and centre error rather than just latency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticTarget:
    x: float
    y: float
    radius: float


# Magenta, matching the default HSV band in configs/aimlabs_gridshot.yaml.
DEFAULT_TARGET_BGR = (200, 20, 220)
DEFAULT_BACKGROUND_BGR = (48, 42, 40)


def render_frame(
    width: int,
    height: int,
    targets: list[SyntheticTarget],
    background_bgr: tuple[int, int, int] = DEFAULT_BACKGROUND_BGR,
    target_bgr: tuple[int, int, int] = DEFAULT_TARGET_BGR,
    noise: float = 0.0,
    crosshair: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    frame = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    for t in targets:
        cv2.circle(frame, (int(round(t.x)), int(round(t.y))), int(round(t.radius)), target_bgr, -1, cv2.LINE_AA)
    if crosshair:
        # A green crosshair over the target is exactly the hole the mask close fills.
        cx, cy = width // 2, height // 2
        cv2.line(frame, (cx - 9, cy), (cx + 9, cy), (60, 255, 60), 2)
        cv2.line(frame, (cx, cy - 9), (cx, cy + 9), (60, 255, 60), 2)
    if noise > 0:
        rng = np.random.default_rng(seed)
        grain = rng.normal(0.0, noise, frame.shape)
        frame = np.clip(frame.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return frame


def orbiting_sequence(
    width: int,
    height: int,
    frames: int,
    radius: float = 18.0,
    orbit_px: float = 70.0,
    revolutions: float = 1.0,
    count: int = 3,
    noise: float = 0.0,
    seed: int | None = 7,
) -> tuple[list[np.ndarray], list[list[SyntheticTarget]]]:
    """Targets orbiting the crosshair: constant speed, known position every frame."""
    cx, cy = width / 2.0, height / 2.0
    images: list[np.ndarray] = []
    truth: list[list[SyntheticTarget]] = []
    for i in range(frames):
        phase = 2 * math.pi * revolutions * i / max(1, frames - 1)
        targets = [
            SyntheticTarget(
                x=cx + orbit_px * math.cos(phase + 2 * math.pi * k / count),
                y=cy + orbit_px * math.sin(phase + 2 * math.pi * k / count),
                radius=radius,
            )
            for k in range(count)
        ]
        truth.append(targets)
        images.append(render_frame(width, height, targets, noise=noise, seed=seed))
    return images, truth


def linear_sequence(
    width: int,
    height: int,
    frames: int,
    start: tuple[float, float],
    velocity_px_per_frame: tuple[float, float],
    radius: float = 18.0,
    noise: float = 0.0,
) -> tuple[list[np.ndarray], list[list[SyntheticTarget]]]:
    """A single target on a straight line -- the clean case for velocity estimation."""
    images: list[np.ndarray] = []
    truth: list[list[SyntheticTarget]] = []
    for i in range(frames):
        t = SyntheticTarget(
            x=start[0] + velocity_px_per_frame[0] * i,
            y=start[1] + velocity_px_per_frame[1] * i,
            radius=radius,
        )
        truth.append([t])
        images.append(render_frame(width, height, [t], noise=noise, crosshair=False))
    return images, truth
