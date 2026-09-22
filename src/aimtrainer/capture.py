"""Screen capture, cropped to a region of interest around the crosshair.

The ROI is the single biggest lever on end-to-end latency. Grabbing 320x320 instead of
1920x1080 is ~35x less pixel traffic, and it shrinks every downstream stage with it.
Targets further from the crosshair than the ROI half-width are not worth chasing
anyway -- by the time you swung that far the trainer has spawned new ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np

from .config import CaptureConfig


@dataclass(frozen=True)
class Region:
    """An absolute screen rectangle, plus where the crosshair sits inside it."""

    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.left + x, self.top + y


def centered_region(cfg: CaptureConfig) -> Region:
    """ROI centred on the display, which is where the crosshair is."""
    w = min(cfg.roi_width, cfg.screen_width)
    h = min(cfg.roi_height, cfg.screen_height)
    return Region(
        left=(cfg.screen_width - w) // 2,
        top=(cfg.screen_height - h) // 2,
        width=w,
        height=h,
    )


class Capture(Protocol):
    region: Region

    def grab(self) -> np.ndarray | None:
        """Latest frame as HxWx3 BGR, or None if no new frame is ready."""

    def close(self) -> None:
        ...

    def describe(self) -> str:
        ...


class MSSCapture:
    """Cross-platform capture via mss. Works everywhere, ~2-5 ms for a small ROI."""

    def __init__(self, cfg: CaptureConfig, region: Region) -> None:
        import mss

        self.region = region
        self._cfg = cfg
        self._sct = mss.mss()
        self._box = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }

    def grab(self) -> np.ndarray:
        shot = self._sct.grab(self._box)
        # mss hands back BGRA; drop alpha without copying the whole buffer twice.
        return np.asarray(shot, dtype=np.uint8)[:, :, :3]

    def close(self) -> None:
        self._sct.close()

    def describe(self) -> str:
        return f"mss {self.region.width}x{self.region.height}"


class DXCamCapture:
    """Windows Desktop Duplication via dxcam. Sub-millisecond, and the right default.

    ``grab`` returns None when the desktop has not changed since the last call, which
    is genuinely useful: a static frame means no new information, so the loop can skip
    detection entirely and re-use the previous tracks.
    """

    def __init__(self, cfg: CaptureConfig, region: Region) -> None:
        import dxcam

        self.region = region
        self._camera = dxcam.create(output_idx=max(0, cfg.monitor - 1), output_color="BGR")
        if self._camera is None:
            raise RuntimeError("dxcam could not open the display")
        box = (region.left, region.top, region.left + region.width, region.top + region.height)
        self._camera.start(region=box, target_fps=cfg.target_fps, video_mode=True)

    def grab(self) -> np.ndarray | None:
        return self._camera.get_latest_frame()

    def close(self) -> None:
        try:
            self._camera.stop()
        except Exception:  # pragma: no cover - teardown races on the dxcam thread
            pass

    def describe(self) -> str:
        return f"dxcam {self.region.width}x{self.region.height}"


class FrameSequenceCapture:
    """Replays in-memory frames. Used by the tests and by ``bench --synthetic``."""

    def __init__(self, frames: Sequence[np.ndarray] | Iterable[np.ndarray], region: Region, loop: bool = True) -> None:
        self._frames = list(frames)
        if not self._frames:
            raise ValueError("no frames supplied")
        self.region = region
        self._loop = loop
        self._i = 0

    def grab(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            if not self._loop:
                return None
            self._i = 0
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._frames)

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return f"frames[{len(self._frames)}] {self.region.width}x{self.region.height}"


class CaptureUnavailableError(RuntimeError):
    """No screen-capture backend could be opened."""


def build_capture(cfg: CaptureConfig, region: Region | None = None) -> Capture:
    """Open the best available backend.

    ``auto`` prefers dxcam and falls back to mss, so the same config works on a
    Windows rig and on a Linux box you are only testing the pipeline on.
    """
    region = region or centered_region(cfg)
    backend = cfg.backend.lower()
    if backend not in ("auto", "mss", "dxcam"):
        raise ValueError(f"unknown capture backend {cfg.backend!r} (expected auto, dxcam or mss)")

    attempts: list[tuple[str, Exception]] = []
    order = ["dxcam", "mss"] if backend == "auto" else [backend]
    for name in order:
        try:
            return DXCamCapture(cfg, region) if name == "dxcam" else MSSCapture(cfg, region)
        except Exception as exc:
            attempts.append((name, exc))

    detail = "\n".join(f"  {name}: {type(exc).__name__}: {exc}" for name, exc in attempts)
    raise CaptureUnavailableError(
        f"could not open a screen-capture backend.\n{detail}\n"
        "  dxcam is Windows-only: pip install 'aimtrainer-bot[windows]'\n"
        "  mss works elsewhere but needs a real display (no X/Wayland session here?)."
    )
