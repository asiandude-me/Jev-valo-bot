#!/usr/bin/env python3
"""A virtual gridshot trainer, for testing the bot without Windows or a real game.

Renders targets to an X display, reads the mouse the way a game does (relative
motion, pointer re-centred every frame), rotates the view accordingly, and scores
clicks. That makes the whole loop real end to end: the bot's capture backend grabs
actual screen pixels and its mouse backend sends actual input.

    Xvfb :99 -screen 0 1920x1080x24 &
    DISPLAY=:99 python tools/simulate_aimlabs.py --seconds 20 --report run.json

Two modes:
  gridshot  three targets at a time; a hit respawns one elsewhere (the default)
  static    one motionless target -- what `aimtrainer calibrate` needs

The view model here is deliberately the same rectilinear projection the bot assumes,
so a correct bot should converge exactly. Its `--deg-per-count` is the ground truth
the calibrator has to recover.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from Xlib import X, display as xdisplay

BACKGROUND = (40, 36, 34)      # RGB
TARGET = (220, 20, 200)        # bright magenta, matching the bot's default HSV band
CROSSHAIR = (60, 255, 60)
BUTTON1_MASK = 1 << 8


@dataclass
class Target:
    yaw: float      # degrees, in world space
    pitch: float
    radius_px: float


@dataclass
class Scorecard:
    hits: int = 0
    misses: int = 0
    shots: int = 0
    time_to_hit: list[float] = field(default_factory=list)
    hit_offsets_px: list[float] = field(default_factory=list)
    frames: int = 0
    seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return 100.0 * self.hits / self.shots if self.shots else 0.0

    def as_dict(self) -> dict:
        ordered = sorted(self.time_to_hit)
        median = ordered[len(ordered) // 2] if ordered else None
        return {
            "shots": self.shots,
            "hits": self.hits,
            "misses": self.misses,
            "accuracy_pct": round(self.accuracy, 1),
            "hits_per_second": round(self.hits / self.seconds, 2) if self.seconds else 0.0,
            "median_time_to_hit_ms": round(median * 1000.0, 1) if median else None,
            "mean_hit_offset_px": (
                round(sum(self.hit_offsets_px) / len(self.hit_offsets_px), 2)
                if self.hit_offsets_px else None
            ),
            "render_fps": round(self.frames / self.seconds, 1) if self.seconds else 0.0,
            "seconds": round(self.seconds, 1),
        }


class VirtualTrainer:
    def __init__(self, args) -> None:
        self.display = xdisplay.Display()
        screen = self.display.screen()
        self.root = screen.root
        geom = self.root.get_geometry()
        self.width, self.height = geom.width, geom.height
        self.cx, self.cy = self.width // 2, self.height // 2
        self.gc = self.root.create_gc()

        self.focal = (self.width / 2.0) / math.tan(math.radians(args.hfov) / 2.0)
        self.deg_per_count = args.deg_per_count
        self.patch = args.patch
        self.spread = args.spread
        self.radius = args.radius
        self.rng = random.Random(args.seed)

        self.view_yaw = 0.0
        self.view_pitch = 0.0
        self.score = Scorecard()

        # Ask X to deliver button presses to us, so a fast click between two polls is
        # never missed the way a state poll would miss it.
        self.root.change_attributes(event_mask=X.ButtonPressMask)
        self.display.sync()

        if args.mode == "static":
            self.targets = [Target(yaw=2.5, pitch=0.0, radius_px=self.radius)]
        else:
            self.targets = [self._spawn() for _ in range(args.targets)]
        self.spawned_at = [time.perf_counter()] * len(self.targets)

    # -- world ---------------------------------------------------------------------
    def _spawn(self) -> Target:
        return Target(
            yaw=self.rng.uniform(-self.spread, self.spread),
            pitch=self.rng.uniform(-self.spread * 0.6, self.spread * 0.6),
            radius_px=self.radius,
        )

    def project(self, target: Target) -> tuple[float, float]:
        """World angles -> screen pixels, through the same projection the bot assumes."""
        rel_yaw = math.radians(target.yaw - self.view_yaw)
        rel_pitch = math.radians(target.pitch - self.view_pitch)
        dx = self.focal * math.tan(rel_yaw)
        dy = math.hypot(self.focal, dx) * math.tan(rel_pitch)
        return self.cx + dx, self.cy + dy

    def read_mouse(self) -> None:
        """Consume relative motion and re-centre the pointer, exactly as a game does."""
        pointer = self.root.query_pointer()
        dx, dy = pointer.root_x - self.cx, pointer.root_y - self.cy
        if dx or dy:
            self.view_yaw += dx * self.deg_per_count
            self.view_pitch += dy * self.deg_per_count
            self.root.warp_pointer(self.cx, self.cy)
            self.display.sync()

    def handle_clicks(self) -> None:
        while self.display.pending_events():
            event = self.display.next_event()
            if event.type != X.ButtonPress or event.detail != 1:
                continue
            self.score.shots += 1
            hit_index, offset = self._target_under_crosshair()
            if hit_index is None:
                self.score.misses += 1
                continue
            self.score.hits += 1
            self.score.hit_offsets_px.append(offset)
            self.score.time_to_hit.append(time.perf_counter() - self.spawned_at[hit_index])
            self.targets[hit_index] = self._spawn()
            self.spawned_at[hit_index] = time.perf_counter()

    def _target_under_crosshair(self) -> tuple[int | None, float]:
        best, best_offset = None, 1e9
        for i, target in enumerate(self.targets):
            x, y = self.project(target)
            offset = math.hypot(x - self.cx, y - self.cy)
            if offset <= target.radius_px and offset < best_offset:
                best, best_offset = i, offset
        return best, best_offset

    # -- rendering -----------------------------------------------------------------
    def render(self) -> None:
        """Repaint a patch around the crosshair.

        Only the bot's ROI actually matters, so painting a patch instead of the whole
        1920x1080 root keeps the simulator fast enough not to be the bottleneck.
        """
        size = self.patch
        frame = np.empty((size, size, 3), dtype=np.uint8)
        frame[:, :] = BACKGROUND
        origin_x, origin_y = self.cx - size // 2, self.cy - size // 2

        yy, xx = np.mgrid[0:size, 0:size]
        for target in self.targets:
            x, y = self.project(target)
            lx, ly = x - origin_x, y - origin_y
            if not (-target.radius_px < lx < size + target.radius_px):
                continue
            if not (-target.radius_px < ly < size + target.radius_px):
                continue
            frame[(xx - lx) ** 2 + (yy - ly) ** 2 <= target.radius_px ** 2] = TARGET

        half = size // 2
        frame[half - 1 : half + 2, half - 9 : half + 10] = CROSSHAIR
        frame[half - 9 : half + 10, half - 1 : half + 2] = CROSSHAIR

        self.root.put_pil_image(self.gc, origin_x, origin_y, Image.fromarray(frame))
        self.display.flush()

    # -- loop ----------------------------------------------------------------------
    def run(self, seconds: float, fps: float) -> Scorecard:
        self.root.warp_pointer(self.cx, self.cy)
        self.display.sync()
        start = time.perf_counter()
        interval = 1.0 / fps
        next_frame = start
        while True:
            now = time.perf_counter()
            if now - start >= seconds:
                break
            self.read_mouse()
            self.handle_clicks()
            self.render()
            self.score.frames += 1
            next_frame += interval
            sleep = next_frame - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_frame = time.perf_counter()
        self.score.seconds = time.perf_counter() - start
        return self.score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["gridshot", "static"], default="gridshot")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=144.0)
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--radius", type=float, default=20.0, help="target radius in pixels at centre")
    ap.add_argument("--spread", type=float, default=7.0, help="half-width of the spawn area, degrees")
    ap.add_argument("--hfov", type=float, default=103.0)
    ap.add_argument("--deg-per-count", type=float, default=0.0245,
                    help="ground truth the calibrator should recover")
    ap.add_argument("--patch", type=int, default=520, help="repainted square around the crosshair")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--report", help="write the scorecard here as JSON")
    args = ap.parse_args()

    trainer = VirtualTrainer(args)
    print(f"virtual {args.mode}: {trainer.width}x{trainer.height}, "
          f"deg_per_count={args.deg_per_count}, {args.seconds:.0f}s", flush=True)
    score = trainer.run(args.seconds, args.fps)

    report = score.as_dict()
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
