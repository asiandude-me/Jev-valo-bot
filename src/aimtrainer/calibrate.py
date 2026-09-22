"""Measure ``deg_per_count`` empirically.

This is the one number that has to be right. Every other parameter trades speed for
stability; this one sets whether the controller's model of the world is true at all.
If it is 10% low the crosshair converges 10% short on every single correction, and no
amount of gain tuning fixes that -- it just makes the undershoot faster.

Method: lock a static target, move the mouse a known number of counts, see how far the
target slid, and solve the projection for the rotation that must have happened.

The probe size is found by search rather than taken from the config, because the
config value is exactly the thing being measured and may be wildly wrong -- a probe
sized from a bad guess either swings the target off screen or moves it so little that
detection noise swamps the result. A first coarse probe is grown or shrunk until it
lands, and its rough estimate then sizes the real trials. Those alternate direction so
that any one-way drift in the scene cancels instead of biasing the answer.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from enum import Enum

from .capture import Capture, Region, build_capture, centered_region
from .config import Config
from .detectors import build_detector
from .detectors.base import Detection
from .geometry import focal_length_px, solve_deg_per_count
from .mouse import MouseBackend, build_mouse

MIN_COUNTS = 15.0
MAX_COUNTS = 30000.0
MIN_USABLE_SHIFT_PX = 4.0     # below this, detection noise dominates the measurement
PROBE_FRACTION = 0.25         # aim each trial at this fraction of the ROI width
MAX_PROBE_ATTEMPTS = 8


class Outcome(Enum):
    OK = "ok"
    NO_TARGET = "no target visible"
    LOST = "target left the view"      # probe too large
    TOO_SMALL = "target barely moved"  # probe too small


@dataclass
class CalibrationResult:
    deg_per_count: float
    trials: list[float]
    spread_pct: float
    probe_counts: float = 0.0

    @property
    def ok(self) -> bool:
        return len(self.trials) >= 3 and self.spread_pct < 5.0


def _largest(detections: list[Detection]) -> Detection | None:
    return max(detections, key=lambda d: d.area) if detections else None


def _settled_target(
    capture: Capture, detector, attempts: int = 60, settle_s: float = 0.02
) -> Detection | None:
    """Wait for a detection that has stopped moving, so the trial starts from rest."""
    previous: Detection | None = None
    for _ in range(attempts):
        frame = capture.grab()
        if frame is None:
            time.sleep(settle_s)
            continue
        current = _largest(list(detector.detect(frame)))
        if current is None:
            previous = None
        elif previous is not None and abs(current.cx - previous.cx) < 1.5:
            return current
        else:
            previous = current
        time.sleep(settle_s)
    return previous


class _Probe:
    """One move-measure-undo cycle. Always restores the view, whatever the outcome."""

    def __init__(self, capture, detector, mouse, region: Region, focal: float, settle_ms: float):
        self.capture = capture
        self.detector = detector
        self.mouse = mouse
        self.region = region
        self.focal = focal
        self.settle_ms = settle_ms

    def run(self, counts: float, direction: int) -> tuple[Outcome, float | None]:
        before = _settled_target(self.capture, self.detector)
        if before is None:
            return Outcome.NO_TARGET, None

        step = int(round(counts * direction))
        if step == 0:
            return Outcome.TOO_SMALL, None

        self.mouse.move_relative(step, 0)
        time.sleep(self.settle_ms / 1000.0)
        after = _settled_target(self.capture, self.detector, attempts=20)

        # Undo before judging, so a failed probe never leaves the view rotated.
        self.mouse.move_relative(-step, 0)
        time.sleep(self.settle_ms / 1000.0)

        if after is None:
            return Outcome.LOST, None
        moved_px = abs(after.cx - before.cx)
        if moved_px <= MIN_USABLE_SHIFT_PX:
            return Outcome.TOO_SMALL, None
        if moved_px >= self.region.width * 0.9:
            # It moved further than a probe this size could explain, so we are almost
            # certainly looking at a *different* target now.
            return Outcome.LOST, None
        return Outcome.OK, solve_deg_per_count(
            before.cx, after.cx, float(step), self.region.width / 2.0, self.focal
        )


def _ideal_counts(deg_per_count: float, region: Region, focal: float) -> float:
    """Counts that should slide the target by ``PROBE_FRACTION`` of the ROI."""
    desired_deg = math.degrees(math.atan2(region.width * PROBE_FRACTION, focal))
    return min(MAX_COUNTS, max(MIN_COUNTS, desired_deg / max(deg_per_count, 1e-9)))


def _find_probe_size(probe: _Probe, start_counts: float, region: Region, focal: float) -> tuple[float, float]:
    """Grow or shrink the probe until one lands. Returns (counts, rough estimate)."""
    counts = start_counts
    last = Outcome.NO_TARGET
    for _ in range(MAX_PROBE_ATTEMPTS):
        outcome, estimate = probe.run(counts, 1)
        last = outcome
        if outcome is Outcome.OK and estimate:
            return _ideal_counts(abs(estimate), region, focal), abs(estimate)
        if outcome is Outcome.TOO_SMALL:
            counts = min(MAX_COUNTS, counts * 2.5)
        elif outcome is Outcome.LOST:
            counts = max(MIN_COUNTS, counts / 2.5)
        else:
            break  # nothing on screen at all; retrying a bigger swing will not help
    raise RuntimeError(_failure_hint(last))


def _failure_hint(outcome: Outcome) -> str:
    if outcome is Outcome.NO_TARGET:
        return (
            "no target visible. Check with `aimtrainer preview` that the detector's "
            "colour range matches your Aim Labs target colour, and that the trainer is "
            "on screen with a static target (a Microshot-style task works well)."
        )
    return (
        f"could not size a usable probe ({outcome.value}). Make sure exactly one static "
        "target is visible, that the trainer window has focus, and that the game is "
        "actually receiving synthetic mouse input."
    )


def calibrate(
    cfg: Config,
    trials: int = 8,
    settle_ms: float = 80.0,
    capture: Capture | None = None,
    mouse: MouseBackend | None = None,
) -> CalibrationResult:
    region = capture.region if capture else centered_region(cfg.capture)
    capture = capture or build_capture(cfg.capture, region)
    mouse = mouse or build_mouse()
    detector = build_detector(cfg.detector)
    focal = focal_length_px(cfg.capture.screen_width, cfg.aim.hfov_deg)

    results: list[float] = []
    try:
        probe = _Probe(capture, detector, mouse, region, focal, settle_ms)
        counts, rough = _find_probe_size(
            probe, _ideal_counts(cfg.aim.deg_per_count, region, focal), region, focal
        )
        results.append(rough)

        for i in range(trials):
            outcome, estimate = probe.run(counts, 1 if i % 2 == 0 else -1)
            if outcome is Outcome.OK and estimate:
                results.append(abs(estimate))
    finally:
        capture.close()

    if not results:
        raise RuntimeError(_failure_hint(Outcome.NO_TARGET))

    median = statistics.median(results)
    spread = (max(results) - min(results)) / median * 100.0 if median else 0.0
    return CalibrationResult(
        deg_per_count=median,
        trials=results,
        spread_pct=abs(spread),
        probe_counts=counts,
    )


def format_result(result: CalibrationResult, cfg: Config) -> str:
    lines = [
        f"deg_per_count = {result.deg_per_count:.6f}",
        f"  trials  : {len(result.trials)}  spread {result.spread_pct:.1f}%  "
        f"probe {result.probe_counts:.0f} counts",
        f"  current : {cfg.aim.deg_per_count:.6f} in config",
        f"  360 turn: {360.0 / result.deg_per_count:.0f} counts",
    ]
    if not result.ok:
        lines.append(
            "\n  WARNING: inconsistent results. Mouse acceleration ('Enhance pointer\n"
            "           precision' in Windows, or any in-game acceleration) breaks the\n"
            "           linear-device assumption this model depends on. Turn it off and\n"
            "           re-run before trusting the number above."
        )
    lines.append("\nPut this in your config:\n\naim:\n  deg_per_count: %.6f" % result.deg_per_count)
    return "\n".join(lines)
