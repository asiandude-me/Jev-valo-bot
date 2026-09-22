"""The calibrator is checked against a simulated world with a known sensitivity.

A fake display renders one static target; a fake mouse rotates that display's view by
a sensitivity the test chooses. If the calibrator recovers that number, the maths and
the procedure are both right.
"""

import math

import pytest

from aimtrainer.calibrate import calibrate, format_result
from aimtrainer.capture import Region
from aimtrainer.config import Config
from aimtrainer.geometry import focal_length_px
from aimtrainer.synthetic import SyntheticTarget, render_frame


class SimulatedWorld:
    """A target at a fixed world angle, seen through a view the mouse can rotate."""

    def __init__(self, cfg, deg_per_count, target_yaw_deg=3.0, noise=0.0):
        self.region = Region(0, 0, cfg.capture.roi_width, cfg.capture.roi_height)
        self.focal = focal_length_px(cfg.capture.screen_width, cfg.aim.hfov_deg)
        self.deg_per_count = deg_per_count
        self.target_yaw = target_yaw_deg
        self.view_yaw = 0.0
        self.noise = noise
        self.closed = False

    # -- capture side --
    def grab(self):
        relative = self.target_yaw - self.view_yaw
        x = self.region.width / 2.0 + self.focal * math.tan(math.radians(relative))
        y = self.region.height / 2.0
        if not 0 <= x < self.region.width:
            return render_frame(self.region.width, self.region.height, [], crosshair=False)
        return render_frame(
            self.region.width, self.region.height,
            [SyntheticTarget(x, y, 16.0)], noise=self.noise, seed=1, crosshair=False,
        )

    def close(self):
        self.closed = True

    def describe(self):
        return "simulated"

    # -- mouse side --
    def move_relative(self, dx, dy):
        self.view_yaw += dx * self.deg_per_count

    def click(self):
        pass


def cfg_with(deg_per_count=0.0245):
    cfg = Config()
    cfg.aim.deg_per_count = deg_per_count
    return cfg


@pytest.mark.parametrize("truth", [0.0245, 0.0140, 0.0480])
def test_calibration_recovers_a_known_sensitivity(truth):
    cfg = cfg_with(deg_per_count=0.03)  # deliberately a wrong starting guess
    world = SimulatedWorld(cfg, deg_per_count=truth)
    result = calibrate(cfg, trials=4, settle_ms=0.0, capture=world, mouse=world)
    assert result.deg_per_count == pytest.approx(truth, rel=0.03)


@pytest.mark.parametrize("guess", [0.002, 0.30])
def test_calibration_is_robust_to_a_bad_initial_guess(guess):
    """The config value IS the unknown, so the probe search must not depend on it.

    Too small a guess sizes a probe that swings the target off screen; too large a
    guess barely moves it. Both must be recovered from.
    """
    truth = 0.0245
    cfg = cfg_with(deg_per_count=guess)
    world = SimulatedWorld(cfg, deg_per_count=truth)
    result = calibrate(cfg, trials=4, settle_ms=0.0, capture=world, mouse=world)
    assert result.deg_per_count == pytest.approx(truth, rel=0.05)


def test_the_probe_settles_on_a_sensible_size():
    """It should end up sliding the target across a useful fraction of the ROI."""
    cfg = cfg_with(deg_per_count=0.002)
    world = SimulatedWorld(cfg, deg_per_count=0.0245)
    result = calibrate(cfg, trials=4, settle_ms=0.0, capture=world, mouse=world)
    shift_deg = result.probe_counts * 0.0245
    shift_px = world.focal * math.tan(math.radians(shift_deg))
    assert 0.15 * world.region.width < shift_px < 0.45 * world.region.width


def test_a_failed_probe_still_restores_the_view():
    """A probe that loses the target must undo itself, or the search drifts away."""
    cfg = cfg_with(deg_per_count=0.0001)  # forces an enormous first probe
    world = SimulatedWorld(cfg, deg_per_count=0.0245)
    calibrate(cfg, trials=2, settle_ms=0.0, capture=world, mouse=world)
    assert world.view_yaw == pytest.approx(0.0, abs=0.2)


def test_a_clean_run_reports_a_tight_spread_and_passes():
    cfg = cfg_with()
    world = SimulatedWorld(cfg, deg_per_count=0.0245)
    result = calibrate(cfg, trials=6, settle_ms=0.0, capture=world, mouse=world)
    assert result.ok
    assert result.spread_pct < 5.0
    assert len(result.trials) >= 3


def test_the_view_is_left_where_it_started():
    """Each trial undoes its own probe, so calibrating does not drift your aim."""
    cfg = cfg_with()
    world = SimulatedWorld(cfg, deg_per_count=0.0245)
    calibrate(cfg, trials=4, settle_ms=0.0, capture=world, mouse=world)
    assert world.view_yaw == pytest.approx(0.0, abs=0.2)


def test_trials_alternate_direction():
    """Alternating cancels any one-directional drift instead of baking it in."""
    cfg = cfg_with()
    moves = []

    world = SimulatedWorld(cfg, deg_per_count=0.0245)
    original = world.move_relative

    def recording(dx, dy):
        moves.append(dx)
        original(dx, dy)

    world.move_relative = recording
    calibrate(cfg, trials=4, settle_ms=0.0, capture=world, mouse=world)
    probes = [m for i, m in enumerate(moves) if i % 2 == 0]
    assert any(p > 0 for p in probes) and any(p < 0 for p in probes)


def test_capture_is_released_even_when_calibration_fails():
    cfg = cfg_with()

    class Blank(SimulatedWorld):
        def grab(self):
            return render_frame(self.region.width, self.region.height, [], crosshair=False)

    world = Blank(cfg, deg_per_count=0.0245)
    with pytest.raises(RuntimeError, match="no target visible"):
        calibrate(cfg, trials=2, settle_ms=0.0, capture=world, mouse=world)
    assert world.closed


def test_noise_does_not_break_the_estimate():
    truth = 0.0245
    cfg = cfg_with()
    world = SimulatedWorld(cfg, deg_per_count=truth, noise=6.0)
    result = calibrate(cfg, trials=6, settle_ms=0.0, capture=world, mouse=world)
    assert result.deg_per_count == pytest.approx(truth, rel=0.05)


def test_the_report_shows_the_value_and_how_to_apply_it():
    cfg = cfg_with()
    world = SimulatedWorld(cfg, deg_per_count=0.0245)
    text = format_result(calibrate(cfg, trials=4, settle_ms=0.0, capture=world, mouse=world), cfg)
    assert "deg_per_count" in text
    assert "aim:" in text  # copy-pasteable config snippet


def test_an_inconsistent_device_is_flagged_rather_than_silently_averaged():
    """Mouse acceleration breaks the linear model; the user has to be told."""
    cfg = cfg_with()

    class Accelerating(SimulatedWorld):
        def move_relative(self, dx, dy):
            # Larger moves rotate disproportionately far, as pointer accel does.
            self.view_yaw += dx * self.deg_per_count * (1.0 + 0.5 * (dx > 0))

    world = Accelerating(cfg, deg_per_count=0.0245)
    result = calibrate(cfg, trials=6, settle_ms=0.0, capture=world, mouse=world)
    assert not result.ok
    assert "acceleration" in format_result(result, cfg).lower()
