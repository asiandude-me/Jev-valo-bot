"""Frame-to-frame target association and velocity estimation.

Detections are anonymous; tracks are not. Identity is what makes two things possible:
leading a moving target (you need its velocity) and holding aim on one target instead
of flickering between two that are equally close (you need to know it is the *same*
one as last frame).

Association is greedy nearest-neighbour under a distance gate. Full Hungarian
assignment would be optimal, but with a handful of well-separated targets it produces
the same pairing for more code and more latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .config import TrackingConfig
from .detectors.base import Detection


@dataclass
class Track:
    track_id: int
    cx: float
    cy: float
    width: float
    height: float
    score: float
    vx: float = 0.0          # pixels per second
    vy: float = 0.0
    hits: int = 1
    missed: int = 0
    last_t: float = 0.0
    history: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def radius(self) -> float:
        """Half the smaller box side -- the radius the crosshair must land inside."""
        return 0.5 * min(self.width, self.height)

    @property
    def has_velocity(self) -> bool:
        return self.hits >= 2 and (self.vx or self.vy)

    def predict(self, dt: float) -> tuple[float, float]:
        """Where this target will be ``dt`` seconds from its last observation."""
        return self.cx + self.vx * dt, self.cy + self.vy * dt


class Tracker:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, detections: Sequence[Detection], t: float) -> list[Track]:
        """Fold this frame's detections in and return the live tracks."""
        gate_sq = self.cfg.max_assign_px ** 2
        unmatched = set(range(len(detections)))

        # Match the most confident tracks first: a well-established track should win a
        # contested detection over one that has been seen once.
        for track in sorted(self.tracks, key=lambda tr: (-tr.hits, tr.missed)):
            best_i, best_d = -1, gate_sq
            for i in unmatched:
                d = (detections[i].cx - track.cx) ** 2 + (detections[i].cy - track.cy) ** 2
                if d < best_d:
                    best_i, best_d = i, d
            if best_i < 0:
                track.missed += 1
                continue
            unmatched.discard(best_i)
            self._absorb(track, detections[best_i], t)

        for i in sorted(unmatched):
            d = detections[i]
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    cx=d.cx,
                    cy=d.cy,
                    width=d.width,
                    height=d.height,
                    score=d.score,
                    last_t=t,
                    history=[(t, d.cx, d.cy)],
                )
            )
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.missed <= self.cfg.max_missed_frames]
        return list(self.tracks)

    def _absorb(self, track: Track, det: Detection, t: float) -> None:
        dt = t - track.last_t
        # Guard the divide: duplicate timestamps (a repeated frame, a coarse clock)
        # would otherwise produce an infinite velocity and fling the crosshair.
        if dt > 1e-4 and track.hits >= self.cfg.min_hits_for_velocity - 1:
            a = self.cfg.velocity_alpha
            track.vx = a * ((det.cx - track.cx) / dt) + (1 - a) * track.vx
            track.vy = a * ((det.cy - track.cy) / dt) + (1 - a) * track.vy
        track.cx, track.cy = det.cx, det.cy
        track.width, track.height = det.width, det.height
        track.score = det.score
        track.hits += 1
        track.missed = 0
        track.last_t = t
        track.history.append((t, det.cx, det.cy))
        if len(track.history) > 16:
            del track.history[:-16]
