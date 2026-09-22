"""Projection maths: screen pixels <-> view angles <-> mouse counts.

A first-person camera is a rectilinear (pinhole) projection, so a pixel offset from
the crosshair does *not* map linearly to a rotation angle -- the error grows with the
tangent of the angle. At the edge of a 103 deg FOV the linear approximation is off by
more than 20%, which is the difference between landing on a target and sitting next
to it. Everything here uses the exact model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Degrees of rotation per mouse count, per unit of in-game sensitivity, for the two
# sensitivity conventions worth knowing. These are *starting points*: measure the real
# value with ``aimtrainer calibrate``.
YAW_CONSTANTS = {
    "valorant": 0.07,   # also Aim Labs' Valorant sensitivity profile
    "source": 0.022,    # Source engine / CS-family m_yaw
}


def focal_length_px(width_px: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels for a horizontal field of view.

    ``hfov_deg`` is the *horizontal* FOV across ``width_px``.
    """
    if width_px <= 0:
        raise ValueError("width_px must be positive")
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError("hfov_deg must be in (0, 180)")
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def pixels_to_angles(dx_px: float, dy_px: float, focal_px: float) -> tuple[float, float]:
    """Convert a pixel offset from the crosshair to (yaw, pitch) in degrees.

    The camera rotates yaw-first, then pitch, so the pitch arm is the distance from
    the eye to the point *after* the yaw rotation -- ``hypot(focal, dx)`` rather than
    ``focal``. Ignoring that under-rotates vertically on off-centre targets.
    """
    if focal_px <= 0:
        raise ValueError("focal_px must be positive")
    yaw = math.degrees(math.atan2(dx_px, focal_px))
    pitch = math.degrees(math.atan2(dy_px, math.hypot(focal_px, dx_px)))
    return yaw, pitch


def angles_to_pixels(yaw_deg: float, pitch_deg: float, focal_px: float) -> tuple[float, float]:
    """Inverse of :func:`pixels_to_angles`."""
    if abs(yaw_deg) >= 90.0 or abs(pitch_deg) >= 90.0:
        raise ValueError("angles must be within +/-90 degrees of the view axis")
    dx = focal_px * math.tan(math.radians(yaw_deg))
    dy = math.hypot(focal_px, dx) * math.tan(math.radians(pitch_deg))
    return dx, dy


@dataclass(frozen=True)
class ViewModel:
    """Everything needed to turn a pixel error into a mouse command.

    ``deg_per_count`` is the measured rotation per mouse count. It already folds in
    DPI and in-game sensitivity, which is why the calibrator solves for it directly
    instead of trying to reconstruct either.
    """

    screen_width: int
    screen_height: int
    hfov_deg: float
    deg_per_count: float

    def __post_init__(self) -> None:
        if self.deg_per_count <= 0:
            raise ValueError("deg_per_count must be positive")

    @property
    def focal_px(self) -> float:
        return focal_length_px(self.screen_width, self.hfov_deg)

    @property
    def vfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan2(self.screen_height / 2.0, self.focal_px))

    def pixel_error_to_counts(self, dx_px: float, dy_px: float) -> tuple[float, float]:
        """Mouse counts that would place the crosshair on a point ``(dx, dy)`` away.

        Returned as floats; the caller is responsible for accumulating the fractional
        part (see :class:`aimtrainer.control.CountAccumulator`).
        """
        yaw, pitch = pixels_to_angles(dx_px, dy_px, self.focal_px)
        return yaw / self.deg_per_count, pitch / self.deg_per_count

    def counts_to_pixel_error(self, dx_counts: float, dy_counts: float) -> tuple[float, float]:
        """Inverse of :meth:`pixel_error_to_counts`, used by the calibrator's checks."""
        return angles_to_pixels(
            dx_counts * self.deg_per_count,
            dy_counts * self.deg_per_count,
            self.focal_px,
        )


def solve_deg_per_count(
    x_before_px: float,
    x_after_px: float,
    counts_moved: float,
    center_x_px: float,
    focal_px: float,
) -> float:
    """Recover ``deg_per_count`` from one calibration move.

    A static target sits at ``x_before``; the mouse moves right by ``counts_moved``,
    which swings the view right and so slides the target *left* by the same angle.
    """
    if counts_moved == 0:
        raise ValueError("counts_moved must be non-zero")
    yaw_before = math.degrees(math.atan2(x_before_px - center_x_px, focal_px))
    yaw_after = math.degrees(math.atan2(x_after_px - center_x_px, focal_px))
    return (yaw_before - yaw_after) / counts_moved
