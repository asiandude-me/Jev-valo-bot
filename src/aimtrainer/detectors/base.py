from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One target in ROI pixel coordinates.

    ``(cx, cy)`` is the aim point, not necessarily the geometric centre -- the colour
    detector biases it upward on tall boxes so you land on the head of a humanoid
    target rather than the chest.
    """

    cx: float
    cy: float
    width: float
    height: float
    score: float = 1.0
    class_id: int = 0

    @property
    def radius(self) -> float:
        return 0.5 * min(self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height


class Detector(Protocol):
    """Stateless per-frame target finder."""

    def detect(self, frame_bgr: np.ndarray) -> Sequence[Detection]:
        ...

    def describe(self) -> str:
        """One line naming the backend and the hardware it ended up on."""
        ...
