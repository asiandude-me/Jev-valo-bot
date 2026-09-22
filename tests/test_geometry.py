import math

import pytest

from aimtrainer.geometry import (
    ViewModel,
    angles_to_pixels,
    focal_length_px,
    pixels_to_angles,
    solve_deg_per_count,
)


def test_focal_length_matches_fov_at_screen_edge():
    f = focal_length_px(1920, 103.0)
    yaw, _ = pixels_to_angles(960, 0, f)
    assert yaw == pytest.approx(103.0 / 2, abs=1e-9)


def test_centre_has_zero_angle_and_is_symmetric():
    f = focal_length_px(1920, 103.0)
    assert pixels_to_angles(0, 0, f) == (0.0, 0.0)
    left, _ = pixels_to_angles(-200, 0, f)
    right, _ = pixels_to_angles(200, 0, f)
    assert left == pytest.approx(-right)


def test_pitch_uses_the_yawed_arm_not_the_raw_focal():
    """Off-centre pitch must be smaller than the naive atan2(dy, focal)."""
    f = focal_length_px(1920, 103.0)
    _, straight_ahead = pixels_to_angles(0, 300, f)
    _, off_axis = pixels_to_angles(800, 300, f)
    naive = math.degrees(math.atan2(300, f))
    assert straight_ahead == pytest.approx(naive)
    assert off_axis < straight_ahead


def test_angles_to_pixels_round_trips():
    f = focal_length_px(1920, 103.0)
    for dx, dy in [(0, 0), (120, -80), (-600, 350), (900, 500)]:
        yaw, pitch = pixels_to_angles(dx, dy, f)
        rx, ry = angles_to_pixels(yaw, pitch, f)
        assert rx == pytest.approx(dx, abs=1e-6)
        assert ry == pytest.approx(dy, abs=1e-6)


def test_projection_is_nonlinear_far_from_centre():
    """The whole reason for the exact model: linear scaling undershoots at the edge."""
    f = focal_length_px(1920, 103.0)
    near, _ = pixels_to_angles(100, 0, f)
    far, _ = pixels_to_angles(800, 0, f)
    linear_prediction = near * 8
    assert far < linear_prediction * 0.9


def test_counts_round_trip_through_the_view_model():
    vm = ViewModel(1920, 1080, 103.0, 0.0245)
    for dx, dy in [(50, 20), (-310, 140), (700, -260)]:
        cx, cy = vm.pixel_error_to_counts(dx, dy)
        rx, ry = vm.counts_to_pixel_error(cx, cy)
        assert rx == pytest.approx(dx, abs=1e-6)
        assert ry == pytest.approx(dy, abs=1e-6)


def test_vfov_is_consistent_with_a_16_9_display():
    vm = ViewModel(1920, 1080, 103.0, 0.0245)
    assert 65.0 < vm.vfov_deg < 75.0


def test_solve_deg_per_count_recovers_a_known_sensitivity():
    truth = 0.0245
    f = focal_length_px(1920, 103.0)
    center_x = 160.0
    counts = 300.0
    # A target 40px right of centre; moving the mouse right swings the view right,
    # so the target slides left by exactly counts * truth degrees.
    x_before = center_x + 40.0
    yaw_before = math.degrees(math.atan2(x_before - center_x, f))
    yaw_after = yaw_before - counts * truth
    x_after = center_x + f * math.tan(math.radians(yaw_after))

    assert solve_deg_per_count(x_before, x_after, counts, center_x, f) == pytest.approx(truth, rel=1e-9)


def test_solve_deg_per_count_is_sign_agnostic():
    truth = 0.03
    f = focal_length_px(1920, 103.0)
    cx = 160.0
    for counts in (250.0, -250.0):
        yaw_before = math.degrees(math.atan2(30.0, f))
        x_after = cx + f * math.tan(math.radians(yaw_before - counts * truth))
        got = solve_deg_per_count(cx + 30.0, x_after, counts, cx, f)
        assert got == pytest.approx(truth, rel=1e-9)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        focal_length_px(0, 103.0)
    with pytest.raises(ValueError):
        focal_length_px(1920, 180.0)
    with pytest.raises(ValueError):
        ViewModel(1920, 1080, 103.0, 0.0)
    with pytest.raises(ValueError):
        solve_deg_per_count(10, 20, 0, 0, 500)
