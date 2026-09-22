"""Target selection and the closed-loop controller that turns pixels into counts.

Four ideas do most of the work here:

* **Dead-time compensation.** The frame you are looking at is one to three frames old,
  so some of the correction you already sent has not shown up in it yet. Acting on the
  raw error double-counts that motion, and the crosshair overshoots and rings -- badly
  enough that without this, gains above about 0.35 diverge outright. Subtracting the
  counts still in flight fixes it, and is worth more than any amount of gain tuning:
  in simulation at a two-frame delay it takes ``kp=0.9`` from *divergent* to settled in
  four frames. (Control theory calls this a Smith predictor.)
* **Stickiness.** Two targets a pixel apart in score would otherwise swap every frame
  and the crosshair would sit between them forever. The current target keeps its lock
  unless a rival is clearly better.
* **Lead.** Even with the loop delay compensated, the *target* has kept moving since
  the frame was taken. Aim along its velocity by roughly the end-to-end latency.
* **Sub-pixel accumulation.** Mouse counts are integers. Truncating a 0.4-count
  correction every tick means it never happens, and the crosshair parks just off
  centre forever. Carry the remainder.

The division of labour: compensation handles the crosshair's own unseen motion, lead
handles the target's constant velocity, and ``kd`` mops up what is left (acceleration,
detection jitter). Each one is aimed at a different source of error.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .config import AimConfig, TriggerConfig
from .geometry import ViewModel
from .tracking import Track


@dataclass
class AimCommand:
    dx_counts: int
    dy_counts: int
    should_fire: bool = False
    target_id: int | None = None
    error_px: float = 0.0


class CountAccumulator:
    """Integer mouse counts with the fractional remainder carried forward."""

    def __init__(self) -> None:
        self._rx = 0.0
        self._ry = 0.0

    def take(self, dx: float, dy: float) -> tuple[int, int]:
        self._rx += dx
        self._ry += dy
        # trunc, not round: round() would push the remainder past zero and make the
        # carry oscillate in sign on a steady sub-count command.
        ix, iy = int(math.trunc(self._rx)), int(math.trunc(self._ry))
        self._rx -= ix
        self._ry -= iy
        return ix, iy

    def reset(self) -> None:
        self._rx = self._ry = 0.0


class TargetSelector:
    """Picks one track to chase, with hysteresis so the choice is stable."""

    def __init__(self, cfg: AimConfig) -> None:
        self.cfg = cfg
        self.current_id: int | None = None

    def reset(self) -> None:
        self.current_id = None

    def select(self, tracks: list[Track], center: tuple[float, float]) -> Track | None:
        cx, cy = center
        best: tuple[float, Track] | None = None
        current: tuple[float, Track] | None = None

        for tr in tracks:
            dist = math.hypot(tr.cx - cx, tr.cy - cy)
            if dist > self.cfg.max_target_distance_px:
                continue
            # Lower is better. Detection confidence breaks ties between targets at
            # similar range without letting a distant-but-crisp blob outrank a near one.
            cost = dist / max(tr.score, 1e-3)
            if best is None or cost < best[0]:
                best = (cost, tr)
            if tr.track_id == self.current_id:
                current = (cost, tr)

        if best is None:
            self.current_id = None
            return None
        if current is not None and current[0] <= best[0] * self.cfg.stickiness:
            self.current_id = current[1].track_id
            return current[1]
        self.current_id = best[1].track_id
        return best[1]


class AimController:
    """PD control in mouse-count space.

    ``kp`` is the fraction of the remaining error closed per tick, so it is stable for
    any ``kp`` in (0, 1] regardless of frame rate -- unlike a gain in counts-per-pixel,
    which would need retuning every time the loop speed changed.

    ``compensation_frames`` should be about (end-to-end latency / frame interval);
    ``bench --live`` prints the latency to divide. Under-estimating it is mild and
    still far better than leaving it at zero, but over-estimating by more than a frame
    or two makes the loop sluggish and can stall it short of the target, so round down
    when unsure.
    """

    def __init__(self, aim: AimConfig, trigger: TriggerConfig, view: ViewModel) -> None:
        self.aim = aim
        self.trigger = trigger
        self.view = view
        self.selector = TargetSelector(aim)
        self._accum = CountAccumulator()
        # Counts issued but not yet visible in a captured frame. Bounded by the
        # configured delay, so old commands age out as the frames catch up.
        self._inflight: deque[tuple[int, int]] = deque(maxlen=max(1, aim.compensation_frames))
        self._prev_err: tuple[float, float] | None = None
        self._prev_id: int | None = None
        self._lock_frames = 0
        self._last_shot_t = -1e9

    def reset(self) -> None:
        self.selector.reset()
        self._accum.reset()
        self._inflight.clear()
        self._prev_err = None
        self._prev_id = None
        self._lock_frames = 0

    def step(self, tracks: list[Track], center: tuple[float, float], t: float) -> AimCommand:
        target = self.selector.select(tracks, center)
        if target is None:
            self.reset()
            return AimCommand(0, 0)

        # Switching targets invalidates the derivative and the carried remainder --
        # they describe the *old* error, and reusing them produces a kick. The
        # in-flight queue is about the crosshair, not the target, so it survives.
        if target.track_id != self._prev_id:
            self._prev_err = None
            self._prev_id = target.track_id
            self._lock_frames = 0
            self._accum.reset()

        lead = self.aim.lead_ms / 1000.0
        px, py = target.predict(lead) if target.has_velocity else (target.cx, target.cy)

        # The frame this came from is already a few frames old. Convert the error it
        # shows into counts, then subtract the counts we have issued since -- what is
        # left is the error that actually remains.
        observed_x, observed_y = self.view.pixel_error_to_counts(
            px - center[0], py - center[1]
        )
        err_cx = observed_x - sum(m[0] for m in self._inflight)
        err_cy = observed_y - sum(m[1] for m in self._inflight)

        # Back to pixels, because the deadzone and settle thresholds are pixel-sized
        # quantities that should mean the same thing wherever the target is on screen.
        eff_x, eff_y = self.view.counts_to_pixel_error(err_cx, err_cy)
        err_px = math.hypot(eff_x, eff_y)

        on_target = err_px <= max(self.trigger.deadzone_px, target.radius)
        self._lock_frames = self._lock_frames + 1 if on_target else 0

        if err_px <= self.aim.settle_px:
            self._prev_err = (err_cx, err_cy)
            self._record(0, 0)
            return AimCommand(0, 0, self._want_fire(t), target.track_id, err_px)

        # On the first tick of a lock there is no previous sample, so the derivative
        # is zero rather than the full error -- otherwise every new target starts with
        # a spurious kd-sized lurch (the classic derivative kick).
        if self._prev_err is None:
            dcx = dcy = 0.0
        else:
            dcx, dcy = err_cx - self._prev_err[0], err_cy - self._prev_err[1]
        self._prev_err = (err_cx, err_cy)

        move_x = self.aim.kp * err_cx + self.aim.kd * dcx
        move_y = self.aim.kp * err_cy + self.aim.kd * dcy

        # Clamp on the vector, not per axis: clamping x and y independently rotates a
        # long diagonal correction off the line to the target.
        magnitude = math.hypot(move_x, move_y)
        if magnitude > self.aim.max_counts_per_tick:
            k = self.aim.max_counts_per_tick / magnitude
            move_x, move_y = move_x * k, move_y * k

        dx, dy = self._accum.take(move_x, move_y)
        self._record(dx, dy)
        fire = self._want_fire(t) if on_target else False
        return AimCommand(dx, dy, fire, target.track_id, err_px)

    def _record(self, dx: int, dy: int) -> None:
        """Remember what the game will actually see.

        The integer counts are recorded, not the float command: the accumulator's
        carry means they differ, and it is the integers that move the view.
        Zero-movement ticks are recorded too, so the queue stays aligned with frames.
        """
        if self.aim.compensation_frames > 0:
            self._inflight.append((dx, dy))

    def _want_fire(self, t: float) -> bool:
        if not self.trigger.enabled:
            return False
        if self._lock_frames < self.trigger.min_lock_frames:
            return False
        if (t - self._last_shot_t) * 1000.0 < self.trigger.cooldown_ms:
            return False
        self._last_shot_t = t
        return True
