"""Latency and accuracy benchmark.

``--synthetic`` measures the pipeline against frames with known ground truth, so it
reports detector recall and centre error alongside timings, and runs anywhere.
``--live`` measures real capture from your display.

The number that matters is ``total`` p95: it is the age of the information the
controller acts on, and it is what ``aim.lead_ms`` should be set close to.
"""

from __future__ import annotations

import math
import time

from .capture import build_capture, centered_region
from .config import Config
from .control import AimController
from .detectors import build_detector
from .runner import view_model
from .stats import LoopStats
from .synthetic import orbiting_sequence
from .tracking import Tracker


def _percentiles(stats: LoopStats) -> str:
    return stats.report()


def bench_synthetic(cfg: Config, frames: int = 400) -> int:
    w, h = cfg.capture.roi_width, cfg.capture.roi_height
    images, truth = orbiting_sequence(w, h, frames, noise=3.0)
    detector = build_detector(cfg.detector)
    tracker = Tracker(cfg.tracking)
    controller = AimController(cfg.aim, cfg.trigger, view_model(cfg))
    stats = LoopStats("detect", "track", "control", "total")
    center = (w / 2.0, h / 2.0)

    matched = expected = 0
    center_errors: list[float] = []

    print(f"detector: {detector.describe()}")
    print(f"frames  : {frames} synthetic {w}x{h}\n")

    # One untimed pass so lazily-built OpenCV kernels and any GPU context are warm.
    for image in images[: min(20, len(images))]:
        detector.detect(image)

    for i, image in enumerate(images):
        t0 = time.perf_counter()
        detections = list(detector.detect(image))
        t1 = time.perf_counter()
        tracks = tracker.update(detections, i / 240.0)
        t2 = time.perf_counter()
        controller.step(tracks, center, i / 240.0)
        t3 = time.perf_counter()

        stats.add("detect", (t1 - t0) * 1000.0)
        stats.add("track", (t2 - t1) * 1000.0)
        stats.add("control", (t3 - t2) * 1000.0)
        stats.add("total", (t3 - t0) * 1000.0)

        expected += len(truth[i])
        for gt in truth[i]:
            nearest = min(
                (math.hypot(d.cx - gt.x, d.cy - gt.y) for d in detections), default=1e9
            )
            if nearest <= gt.radius:
                matched += 1
                center_errors.append(nearest)

    recall = 100.0 * matched / expected if expected else 0.0
    mean_err = sum(center_errors) / len(center_errors) if center_errors else float("nan")
    print(_percentiles(stats))
    print(f"recall: {recall:.1f}% ({matched}/{expected})   mean centre error: {mean_err:.2f}px")
    print(
        "\nnote: synthetic timings exclude screen capture. Run `bench --live` for the\n"
        "      end-to-end number to use for aim.lead_ms."
    )
    return 0


def bench_live(cfg: Config, frames: int = 400) -> int:
    region = centered_region(cfg.capture)
    capture = build_capture(cfg.capture, region)
    detector = build_detector(cfg.detector)
    stats = LoopStats("capture", "detect", "total")

    print(f"capture : {capture.describe()}")
    print(f"detector: {detector.describe()}")
    print(f"frames  : {frames}\n")

    seen = 0
    detected = 0
    try:
        while seen < frames:
            t0 = time.perf_counter()
            frame = capture.grab()
            t1 = time.perf_counter()
            if frame is None:
                continue
            detections = detector.detect(frame)
            t2 = time.perf_counter()
            stats.add("capture", (t1 - t0) * 1000.0)
            stats.add("detect", (t2 - t1) * 1000.0)
            stats.add("total", (t2 - t0) * 1000.0)
            seen += 1
            detected += len(detections)
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()

    print(_percentiles(stats))
    print(f"detections: {detected} over {seen} frames ({detected / max(1, seen):.2f}/frame)")
    total = stats.stages["total"]
    print(f"\nsuggested aim.lead_ms: {total.percentile(95):.0f}  (total p95, plus your display latency)")
    return 0
