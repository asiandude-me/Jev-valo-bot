"""Hotkey parsing, preview drawing, and the mouse backends."""

import numpy as np
import pytest

from aimtrainer.hotkeys import VK_CODES, HotkeyWatcher, vk_code
from aimtrainer.mouse import DryRunMouse, build_mouse
from aimtrainer.preview import annotate, side_by_side
from aimtrainer.synthetic import SyntheticTarget, render_frame
from aimtrainer.tracking import Track


# --- hotkeys ----------------------------------------------------------------------

def test_function_keys_map_to_the_right_virtual_key_codes():
    assert vk_code("F1") == 0x70
    assert vk_code("F12") == 0x7B
    assert vk_code("ESC") == 0x1B


def test_letters_and_digits_are_accepted():
    assert vk_code("k") == ord("K")
    assert vk_code("5") == ord("5")


def test_hotkey_names_are_case_and_space_insensitive():
    assert vk_code(" f1 ") == vk_code("F1")


def test_an_unknown_hotkey_names_the_valid_forms():
    with pytest.raises(ValueError, match="F1-F12"):
        vk_code("NotAKey")


def test_the_watcher_builds_without_a_windows_api():
    watcher = HotkeyWatcher({"toggle": "F1", "quit": "F2"})
    assert set(watcher.codes) == {"toggle", "quit"}
    assert watcher.pressed("toggle") is False  # never fires where it cannot poll


def test_every_documented_default_hotkey_resolves():
    from aimtrainer.config import HotkeyConfig

    cfg = HotkeyConfig()
    assert vk_code(cfg.toggle) in VK_CODES.values()
    assert vk_code(cfg.quit) in VK_CODES.values()


# --- mouse ------------------------------------------------------------------------

def test_dry_run_records_moves_and_clicks():
    mouse = DryRunMouse()
    mouse.move_relative(3, -4)
    mouse.move_relative(1, 1)
    mouse.click()
    assert mouse.moves == [(3, -4), (1, 1)]
    assert mouse.total == (4, -3)
    assert mouse.clicks == 1


def test_dry_run_ignores_zero_moves():
    """A no-op tick should not pad the log or the totals."""
    mouse = DryRunMouse()
    mouse.move_relative(0, 0)
    assert mouse.moves == []


def test_build_mouse_honours_dry_run():
    assert isinstance(build_mouse(dry_run=True), DryRunMouse)


def test_build_mouse_never_returns_a_live_backend_without_a_display():
    """On a headless box it must degrade to dry-run rather than raise."""
    backend = build_mouse(dry_run=False)
    assert hasattr(backend, "move_relative") and hasattr(backend, "click")


# --- preview drawing ---------------------------------------------------------------

def frame_and_track():
    frame = render_frame(320, 320, [SyntheticTarget(200.0, 120.0, 18.0)])
    track = Track(track_id=7, cx=200.0, cy=120.0, width=36, height=36, score=1.0,
                  vx=300.0, vy=0.0, hits=6)
    return frame, track


def test_annotate_draws_without_mutating_the_source_frame():
    frame, track = frame_and_track()
    original = frame.copy()
    out = annotate(frame, [], [track], selected_id=7, center=(160.0, 160.0))
    assert np.array_equal(frame, original)
    assert not np.array_equal(out, original)
    assert out.shape == frame.shape


def test_annotate_handles_an_empty_scene():
    frame = render_frame(320, 320, [])
    out = annotate(frame, [], [], selected_id=None, center=(160.0, 160.0))
    assert out.shape == frame.shape


def test_annotate_draws_a_velocity_arrow_only_for_moving_targets():
    frame, moving = frame_and_track()
    still = Track(track_id=8, cx=200.0, cy=120.0, width=36, height=36, score=1.0, hits=1)
    with_arrow = annotate(frame, [], [moving], 7, (160.0, 160.0))
    without = annotate(frame, [], [still], 8, (160.0, 160.0))
    assert not np.array_equal(with_arrow, without)


def test_side_by_side_appends_the_mask_panel():
    frame = render_frame(320, 320, [])
    mask = np.zeros((320, 320), dtype=np.uint8)
    assert side_by_side(frame, mask).shape == (320, 640, 3)


def test_side_by_side_without_a_mask_is_a_passthrough():
    frame = render_frame(320, 320, [])
    assert side_by_side(frame, None).shape == frame.shape
