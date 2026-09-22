import math

import pytest

from aimtrainer.config import AimConfig, TrackingConfig, TriggerConfig
from aimtrainer.control import AimController, CountAccumulator, TargetSelector
from aimtrainer.detectors.base import Detection
from aimtrainer.geometry import ViewModel
from aimtrainer.tracking import Track, Tracker

CENTER = (160.0, 160.0)


def view():
    return ViewModel(1920, 1080, 103.0, 0.0245)


def controller(**aim_kwargs):
    # compensation_frames defaults to 0 here so each test opts into the behaviour it
    # is actually exercising; the compensation tests below set it explicitly.
    aim = AimConfig(**{"deg_per_count": 0.0245, "compensation_frames": 0, **aim_kwargs})
    return AimController(aim, TriggerConfig(), view())


def delayed_loop(ctrl, delay, start=(280.0, 160.0), steps=60, view_model=None):
    """Drive a controller through a loop where moves take `delay` frames to be seen.

    Returns the observed error magnitude after each tick. This is the real shape of
    the system: the frame the detector hands over is always a little out of date.
    """
    vm = view_model or view()
    true_pos = list(start)
    seen = [tuple(true_pos)] * (delay + 1)
    errors = []
    for i in range(steps):
        cmd = ctrl.step([track(seen[0][0], seen[0][1])], CENTER, i / 240.0)
        dx_px, dy_px = vm.counts_to_pixel_error(cmd.dx_counts, cmd.dy_counts)
        true_pos[0] -= dx_px
        true_pos[1] -= dy_px
        seen = seen[1:] + [tuple(true_pos)]
        errors.append(math.hypot(true_pos[0] - CENTER[0], true_pos[1] - CENTER[1]))
    return errors


def track(x, y, tid=1, vx=0.0, vy=0.0, hits=5):
    return Track(track_id=tid, cx=x, cy=y, width=30, height=30, score=1.0, vx=vx, vy=vy, hits=hits)


# --- sub-pixel accumulation -------------------------------------------------------

def test_accumulator_carries_fractions_until_they_add_up():
    acc = CountAccumulator()
    assert acc.take(0.4, 0.0) == (0, 0)
    assert acc.take(0.4, 0.0) == (0, 0)
    assert acc.take(0.4, 0.0) == (1, 0)


def test_accumulator_loses_nothing_over_many_ticks():
    """Without the carry, a steady sub-count command would truncate to zero forever."""
    acc = CountAccumulator()
    total = sum(acc.take(0.37, -0.21)[0] for _ in range(100))
    assert total == pytest.approx(37, abs=1)


def test_accumulator_does_not_oscillate_in_sign():
    acc = CountAccumulator()
    signs = {acc.take(0.3, 0.0)[0] for _ in range(30)}
    assert -1 not in signs


def test_accumulator_reset_drops_the_carry():
    acc = CountAccumulator()
    acc.take(0.9, 0.9)
    acc.reset()
    assert acc.take(0.05, 0.05) == (0, 0)


# --- target selection -------------------------------------------------------------

def test_selector_prefers_the_nearest_target():
    sel = TargetSelector(AimConfig())
    chosen = sel.select([track(300, 160, tid=1), track(180, 160, tid=2)], CENTER)
    assert chosen.track_id == 2


def test_selector_ignores_targets_beyond_the_range_limit():
    sel = TargetSelector(AimConfig(max_target_distance_px=50.0))
    assert sel.select([track(400, 400, tid=1)], CENTER) is None


def test_selector_sticks_to_the_current_target_against_a_marginal_rival():
    sel = TargetSelector(AimConfig(stickiness=1.35))
    assert sel.select([track(200, 160, tid=1)], CENTER).track_id == 1
    # A rival that is closer, but not 35% better, must not steal the lock.
    assert sel.select([track(200, 160, tid=1), track(195, 160, tid=2)], CENTER).track_id == 1


def test_selector_switches_when_a_rival_is_clearly_better():
    sel = TargetSelector(AimConfig(stickiness=1.35))
    sel.select([track(300, 160, tid=1)], CENTER)
    assert sel.select([track(300, 160, tid=1), track(170, 160, tid=2)], CENTER).track_id == 2


def test_selector_does_not_flicker_between_two_equal_targets():
    """Without hysteresis this alternates every frame and the crosshair sits between."""
    sel = TargetSelector(AimConfig())
    tracks = [track(160 - 40, 160, tid=1), track(160 + 40, 160, tid=2)]
    picks = {sel.select(tracks, CENTER).track_id for _ in range(20)}
    assert len(picks) == 1


def test_selector_releases_the_lock_when_everything_disappears():
    sel = TargetSelector(AimConfig())
    sel.select([track(200, 160, tid=1)], CENTER)
    assert sel.select([], CENTER) is None
    assert sel.current_id is None


# --- controller -------------------------------------------------------------------

def test_no_targets_means_no_movement():
    cmd = controller().step([], CENTER, 0.0)
    assert (cmd.dx_counts, cmd.dy_counts) == (0, 0)
    assert cmd.target_id is None


def test_movement_points_toward_the_target():
    right_down = controller().step([track(280, 240)], CENTER, 0.0)
    assert right_down.dx_counts > 0 and right_down.dy_counts > 0

    left_up = controller().step([track(40, 80)], CENTER, 0.0)
    assert left_up.dx_counts < 0 and left_up.dy_counts < 0


def test_the_crosshair_converges_on_a_static_target():
    ctrl = controller(kp=0.35, kd=0.0, lead_ms=0.0)
    vm = view()
    pos = [280.0, 240.0]
    errors = []
    for i in range(60):
        cmd = ctrl.step([track(pos[0], pos[1])], CENTER, i / 240.0)
        # Applying counts rotates the view, which slides the target the other way.
        dx_px, dy_px = vm.counts_to_pixel_error(cmd.dx_counts, cmd.dy_counts)
        pos[0] -= dx_px
        pos[1] -= dy_px
        errors.append(math.hypot(pos[0] - CENTER[0], pos[1] - CENTER[1]))

    assert errors[-1] < 2.5
    assert errors[-1] < errors[0] / 20
    assert all(b <= a + 1e-6 for a, b in zip(errors, errors[1:]))  # monotonic, no overshoot


def test_higher_gain_converges_faster():
    vm = view()

    def steps_to_settle(kp):
        ctrl = controller(kp=kp, kd=0.0, lead_ms=0.0)
        pos = [300.0, 160.0]
        for i in range(200):
            cmd = ctrl.step([track(pos[0], pos[1])], CENTER, i / 240.0)
            pos[0] -= vm.counts_to_pixel_error(cmd.dx_counts, cmd.dy_counts)[0]
            if abs(pos[0] - CENTER[0]) < 3.0:
                return i
        return 999

    assert steps_to_settle(0.6) < steps_to_settle(0.2)


def test_movement_is_clamped_and_the_clamp_preserves_direction():
    """Per-axis clamping would rotate a diagonal correction off the line to the target."""
    ctrl = controller(kp=1.0, kd=0.0, max_counts_per_tick=50.0, lead_ms=0.0)
    cmd = ctrl.step([track(160 + 600, 160 + 600)], CENTER, 0.0)
    magnitude = math.hypot(cmd.dx_counts, cmd.dy_counts)
    assert magnitude <= 51.0
    assert cmd.dx_counts == pytest.approx(cmd.dy_counts, rel=0.12)


def test_lead_aims_ahead_of_a_moving_target():
    fast = track(200, 160, vx=900.0, hits=10)
    ahead = controller(lead_ms=30.0, kd=0.0).step([fast], CENTER, 0.0)
    no_lead = controller(lead_ms=0.0, kd=0.0).step([fast], CENTER, 0.0)
    assert ahead.dx_counts > no_lead.dx_counts


def test_lead_is_ignored_for_a_target_with_no_velocity_history():
    stationary = Track(track_id=1, cx=200, cy=160, width=30, height=30, score=1.0, hits=1)
    cmd = controller(lead_ms=50.0).step([stationary], CENTER, 0.0)
    expected = controller(lead_ms=0.0).step([stationary], CENTER, 0.0)
    assert cmd.dx_counts == expected.dx_counts


def test_settled_aim_stops_issuing_corrections():
    cmd = controller(settle_px=5.0).step([track(162, 161)], CENTER, 0.0)
    assert (cmd.dx_counts, cmd.dy_counts) == (0, 0)
    assert cmd.error_px < 5.0


def test_switching_targets_does_not_carry_the_old_derivative():
    """A stale error term would produce a kick on the frame the lock changes."""
    ctrl = controller(kd=0.8, compensation_frames=0)
    for i in range(5):
        ctrl.step([track(300 - 10 * i, 160, tid=1)], CENTER, i / 240.0)
    switched = ctrl.step([track(175, 160, tid=2)], CENTER, 5 / 240.0)
    fresh = controller(kd=0.8, compensation_frames=0).step([track(175, 160, tid=2)], CENTER, 0.0)
    assert switched.dx_counts == fresh.dx_counts


def test_the_first_tick_of_a_lock_has_no_derivative_kick():
    """prev_err starts empty, so kd must contribute nothing on the opening frame."""
    with_kd = controller(kd=1.5, kp=0.4).step([track(260, 160)], CENTER, 0.0)
    without_kd = controller(kd=0.0, kp=0.4).step([track(260, 160)], CENTER, 0.0)
    assert with_kd.dx_counts == without_kd.dx_counts


# --- dead-time compensation -------------------------------------------------------

def uncompensated(**kw):
    return controller(compensation_frames=0, kd=0.0, lead_ms=0.0, **kw)


def compensated(frames=2, **kw):
    return controller(compensation_frames=frames, kd=0.0, lead_ms=0.0, **kw)


def test_without_compensation_a_delayed_loop_overshoots():
    """The baseline this whole mechanism exists to fix."""
    errors = delayed_loop(uncompensated(kp=0.6), delay=2)
    assert max(errors[3:]) > 40.0  # rings well past the target


def test_compensation_removes_the_overshoot():
    errors = delayed_loop(compensated(2, kp=0.6), delay=2)
    assert errors[-1] < 2.0
    assert all(b <= a + 1e-6 for a, b in zip(errors, errors[1:]))  # never overshoots


def test_compensation_makes_high_gain_usable():
    """kp=0.9 at a two-frame delay diverges uncompensated and settles compensated."""
    with_comp = delayed_loop(compensated(2, kp=0.9), delay=2)
    without = delayed_loop(uncompensated(kp=0.9), delay=2)
    assert with_comp[-1] < 2.5          # inside settle_px; it has converged
    assert without[-1] > with_comp[-1] * 10


def test_compensation_settles_faster_than_none_at_the_default_gain():
    def steps_to_settle(errors):
        for i, e in enumerate(errors):
            if all(x < 2.0 for x in errors[i:]):
                return i
        return len(errors)

    assert steps_to_settle(delayed_loop(compensated(2, kp=0.5), delay=2)) < steps_to_settle(
        delayed_loop(uncompensated(kp=0.5), delay=2)
    )


def test_under_compensating_still_beats_not_compensating():
    """Guessing the delay low is the safe direction to be wrong in."""
    under = delayed_loop(compensated(1, kp=0.6), delay=3)
    none = delayed_loop(uncompensated(kp=0.6), delay=3)
    assert max(under[3:]) < max(none[3:])


def test_over_compensating_rings_but_still_recovers():
    """Guessing high costs ringing -- which is why the docs say to round down.

    It must still converge rather than diverge, and it must be visibly worse than
    compensating correctly, or the guidance would be pointless.
    """
    too_much = delayed_loop(compensated(3, kp=0.5), delay=1, steps=120)
    correct = delayed_loop(compensated(1, kp=0.5), delay=1, steps=120)
    assert too_much[-1] < 5.0                     # converges
    assert max(too_much) > max(correct) + 1.0     # but rings on the way


def test_compensation_is_harmless_when_there_is_no_delay():
    errors = delayed_loop(compensated(2, kp=0.5), delay=0)
    assert errors[-1] < 2.0


def test_inflight_queue_records_the_integer_counts_actually_sent():
    """The carry means the float command and the counts the game sees differ."""
    ctrl = compensated(2, kp=0.5)
    cmd = ctrl.step([track(250, 160)], CENTER, 0.0)
    assert list(ctrl._inflight) == [(cmd.dx_counts, cmd.dy_counts)]


def test_settled_ticks_still_advance_the_inflight_queue():
    """Skipping them would misalign the queue with frames and mis-time the subtraction."""
    ctrl = compensated(2, kp=0.5, settle_px=5.0)
    ctrl.step([track(161, 160)], CENTER, 0.0)
    assert list(ctrl._inflight) == [(0, 0)]


def test_reset_clears_the_inflight_queue():
    ctrl = compensated(2, kp=0.5)
    ctrl.step([track(250, 160)], CENTER, 0.0)
    ctrl.reset()
    assert list(ctrl._inflight) == []


# --- trigger ----------------------------------------------------------------------

def test_trigger_is_off_by_default():
    ctrl = AimController(AimConfig(deg_per_count=0.0245), TriggerConfig(), view())
    assert ctrl.step([track(160, 160)], CENTER, 0.0).should_fire is False


def test_trigger_needs_the_lock_to_be_held():
    trig = TriggerConfig(enabled=True, deadzone_px=8.0, min_lock_frames=3, cooldown_ms=0.0)
    ctrl = AimController(AimConfig(deg_per_count=0.0245, settle_px=1.0), trig, view())
    fired = [ctrl.step([track(162, 160)], CENTER, i / 240.0).should_fire for i in range(5)]
    assert fired[:2] == [False, False]
    assert any(fired[2:])


def test_trigger_does_not_fire_off_target():
    trig = TriggerConfig(enabled=True, deadzone_px=4.0, min_lock_frames=1, cooldown_ms=0.0)
    ctrl = AimController(AimConfig(deg_per_count=0.0245, max_target_distance_px=1000.0), trig, view())
    assert not any(ctrl.step([track(400, 160)], CENTER, i / 240.0).should_fire for i in range(10))


def test_trigger_respects_the_cooldown():
    trig = TriggerConfig(enabled=True, deadzone_px=10.0, min_lock_frames=1, cooldown_ms=100.0)
    ctrl = AimController(AimConfig(deg_per_count=0.0245, settle_px=1.0), trig, view())
    shots = sum(ctrl.step([track(163, 160)], CENTER, i * 0.01).should_fire for i in range(50))
    assert 4 <= shots <= 6  # 500 ms at one shot per 100 ms


# --- integration ------------------------------------------------------------------

def test_tracker_and_controller_close_the_loop_on_a_real_detection_stream():
    tracker = Tracker(TrackingConfig())
    ctrl = controller(kp=0.4, kd=0.1, lead_ms=0.0)
    vm = view()
    pos = [250.0, 120.0]

    for i in range(80):
        detection = Detection(cx=pos[0], cy=pos[1], width=30, height=30)
        tracks = tracker.update([detection], i / 240.0)
        cmd = ctrl.step(tracks, CENTER, i / 240.0)
        dx_px, dy_px = vm.counts_to_pixel_error(cmd.dx_counts, cmd.dy_counts)
        pos[0] -= dx_px
        pos[1] -= dy_px

    assert math.hypot(pos[0] - CENTER[0], pos[1] - CENTER[1]) < 3.0
