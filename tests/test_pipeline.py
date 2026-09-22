"""End-to-end: synthetic frames -> detector -> tracker -> controller -> mouse."""

import math

import pytest

from aimtrainer.capture import FrameSequenceCapture, Region, centered_region
from aimtrainer.config import CaptureConfig, Config
from aimtrainer.mouse import DryRunMouse
from aimtrainer.runner import Runner
from aimtrainer.stats import LoopStats, Stage
from aimtrainer.synthetic import SyntheticTarget, linear_sequence, orbiting_sequence, render_frame


def cfg_for(roi=320):
    cfg = Config()
    cfg.capture.roi_width = cfg.capture.roi_height = roi
    cfg.aim.lead_ms = 0.0
    return cfg


def runner_over(frames, cfg=None):
    cfg = cfg or cfg_for()
    region = Region(0, 0, cfg.capture.roi_width, cfg.capture.roi_height)
    mouse = DryRunMouse()
    runner = Runner(cfg, capture=FrameSequenceCapture(frames, region, loop=False), mouse=mouse)
    runner.enabled = True
    runner._guard_ok = True          # the guard is exercised in test_guard.py
    runner._last_guard_check = 1e18  # and must not re-query a real window here
    return runner, mouse


def test_region_is_centred_on_the_display():
    region = centered_region(CaptureConfig(screen_width=1920, screen_height=1080, roi_width=320, roi_height=320))
    assert (region.left, region.top) == (800, 380)
    assert region.center == (160.0, 160.0)
    assert region.to_screen(160, 160) == (960, 540)


def test_region_is_clamped_to_the_display():
    region = centered_region(CaptureConfig(screen_width=640, screen_height=480, roi_width=1920, roi_height=1080))
    assert (region.width, region.height) == (640, 480)


def test_frame_sequence_capture_stops_when_not_looping():
    region = Region(0, 0, 8, 8)
    frames = [render_frame(8, 8, [], crosshair=False)] * 2
    cap = FrameSequenceCapture(frames, region, loop=False)
    assert cap.grab() is not None and cap.grab() is not None
    assert cap.grab() is None


def test_frame_sequence_capture_rejects_an_empty_sequence():
    with pytest.raises(ValueError):
        FrameSequenceCapture([], Region(0, 0, 8, 8))


def test_the_loop_drives_the_mouse_toward_an_off_centre_target():
    frames = [render_frame(320, 320, [SyntheticTarget(250.0, 90.0, 18.0)], crosshair=False)] * 12
    runner, mouse = runner_over(frames)
    for _ in frames:
        runner.tick()
    dx, dy = mouse.total
    assert dx > 0 and dy < 0   # target is right of and above the crosshair


def test_the_loop_sends_nothing_when_there_is_no_target():
    frames = [render_frame(320, 320, [], crosshair=True)] * 10
    runner, mouse = runner_over(frames)
    for _ in frames:
        runner.tick()
    assert mouse.moves == []


def test_the_loop_sends_nothing_while_disabled():
    frames = [render_frame(320, 320, [SyntheticTarget(250.0, 90.0, 18.0)], crosshair=False)] * 10
    runner, mouse = runner_over(frames)
    runner.enabled = False
    for _ in frames:
        runner.tick()
    assert mouse.moves == []


def test_the_loop_sends_nothing_when_the_guard_says_no():
    frames = [render_frame(320, 320, [SyntheticTarget(250.0, 90.0, 18.0)], crosshair=False)] * 10
    runner, mouse = runner_over(frames)
    runner._guard_ok = False
    for _ in frames:
        runner.tick()
    assert mouse.moves == []


def test_the_loop_closes_on_the_target_when_the_view_actually_moves():
    """Feed back the mouse output into where the target is rendered next frame."""
    cfg = cfg_for()
    cfg.aim.kp = 0.5
    region = Region(0, 0, 320, 320)
    mouse = DryRunMouse()
    runner = Runner(cfg, capture=FrameSequenceCapture([render_frame(320, 320, [], crosshair=False)], region), mouse=mouse)
    runner.enabled = True
    runner._guard_ok = True
    runner._last_guard_check = 1e18

    from aimtrainer.runner import view_model

    vm = view_model(cfg)
    pos = [270.0, 70.0]
    for i in range(60):
        frame = render_frame(320, 320, [SyntheticTarget(pos[0], pos[1], 18.0)], crosshair=False)
        runner.capture = FrameSequenceCapture([frame], region)
        before = len(mouse.moves)
        runner.tick(now=i / 240.0)
        if len(mouse.moves) > before:
            dx_px, dy_px = vm.counts_to_pixel_error(*mouse.moves[-1])
            pos[0] -= dx_px
            pos[1] -= dy_px

    assert math.hypot(pos[0] - 160.0, pos[1] - 160.0) < 6.0


def test_a_moving_target_is_tracked_with_a_stable_identity():
    frames, truth = linear_sequence(320, 320, 40, start=(60.0, 160.0), velocity_px_per_frame=(5.0, 0.0))
    runner, _ = runner_over(frames)
    runner.enabled = False  # observe only; no feedback into the rendered position
    for i, _ in enumerate(frames):
        runner.tick(now=i / 240.0)
    assert len(runner.tracker.tracks) == 1
    assert runner.tracker.tracks[0].vx > 0


def test_stats_are_collected_for_every_stage():
    frames, _ = orbiting_sequence(320, 320, 20)
    runner, _ = runner_over(frames)
    for _ in frames:
        runner.tick()
    assert runner.stats.frames == len(frames)
    assert runner.stats.detections > 0
    for name in ("capture", "detect", "track", "control", "total"):
        assert runner.stats.stages[name].samples
    assert "fps" in runner.stats.report()


def test_a_none_frame_is_skipped_without_advancing_the_loop():
    """dxcam returns None for an unchanged desktop; that is not a frame."""
    region = Region(0, 0, 320, 320)

    class NoNewFrames:
        region = Region(0, 0, 320, 320)

        def grab(self):
            return None

        def close(self):
            pass

        def describe(self):
            return "none"

    runner, mouse = runner_over([render_frame(320, 320, [], crosshair=False)])
    runner.capture = NoNewFrames()
    runner.tick()
    assert runner.stats.frames == 0
    assert mouse.moves == []


# --- stats ------------------------------------------------------------------------

def test_percentiles_are_ordered_and_drawn_from_real_samples():
    stage = Stage("x")
    for v in range(1, 101):
        stage.add(float(v))
    assert stage.percentile(50) <= stage.percentile(95) <= stage.percentile(100)
    assert stage.percentile(50) in {50.0, 51.0}
    assert stage.mean == pytest.approx(50.5)


def test_an_empty_stage_reports_zero_rather_than_raising():
    assert Stage("x").percentile(95) == 0.0
    assert Stage("x").mean == 0.0
    assert "fps" in LoopStats("total").report()


def test_samples_are_bounded_so_a_long_run_does_not_grow_memory():
    stage = Stage("x")
    for v in range(5000):
        stage.add(float(v))
    assert len(stage.samples) == 600
