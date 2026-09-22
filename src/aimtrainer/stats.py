"""Rolling latency statistics for the loop stages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Stage:
    name: str
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=600))

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        # Nearest-rank: with a few hundred samples the interpolated variant says the
        # same thing, and this one never invents a value that was not measured.
        idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * len(ordered) + 0.5)) - 1))
        return ordered[idx]

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def summary(self) -> str:
        return f"{self.name} p50={self.percentile(50):.2f} p95={self.percentile(95):.2f}ms"


class LoopStats:
    def __init__(self, *names: str) -> None:
        self.stages = {n: Stage(n) for n in names}
        self.frames = 0
        self.detections = 0

    def add(self, name: str, ms: float) -> None:
        self.stages[name].add(ms)

    def report(self) -> str:
        parts = [s.summary() for s in self.stages.values() if s.samples]
        total = self.stages.get("total")
        fps = 1000.0 / total.mean if total and total.mean > 0 else 0.0
        parts.append(f"{fps:.0f} fps")
        return " | ".join(parts)
