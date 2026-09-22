"""The main loop: capture -> detect -> track -> select -> control -> input."""

from __future__ import annotations

import sys
import time

from . import guard
from .capture import Capture, Region, build_capture, centered_region
from .config import Config
from .control import AimController
from .detectors import build_detector
from .geometry import ViewModel
from .hotkeys import HotkeyWatcher
from .mouse import DryRunMouse, MouseBackend, build_mouse
from .stats import LoopStats
from .tracking import Tracker


def view_model(cfg: Config) -> ViewModel:
    return ViewModel(
        screen_width=cfg.capture.screen_width,
        screen_height=cfg.capture.screen_height,
        hfov_deg=cfg.aim.hfov_deg,
        deg_per_count=cfg.aim.deg_per_count,
    )


class Runner:
    def __init__(
        self,
        cfg: Config,
        capture: Capture | None = None,
        mouse: MouseBackend | None = None,
        dry_run: bool = False,
    ) -> None:
        self.cfg = cfg
        self.region: Region = capture.region if capture else centered_region(cfg.capture)
        self.capture = capture or build_capture(cfg.capture, self.region)
        self.mouse = mouse or build_mouse(dry_run=dry_run)
        self.detector = build_detector(cfg.detector)
        self.tracker = Tracker(cfg.tracking)
        self.controller = AimController(cfg.aim, cfg.trigger, view_model(cfg))
        self.stats = LoopStats("capture", "detect", "track", "control", "total")
        self.enabled = False
        self.running = True
        self._hotkeys = HotkeyWatcher({"toggle": cfg.hotkeys.toggle, "quit": cfg.hotkeys.quit})
        self._last_guard_check = 0.0
        self._guard_ok = False

    # -- one iteration, factored out so the tests can drive it deterministically ----
    def tick(self, now: float | None = None) -> None:
        t0 = time.perf_counter()
        now = t0 if now is None else now

        frame = self.capture.grab()
        t1 = time.perf_counter()
        if frame is None:
            # dxcam says "nothing changed"; no new information, so don't re-detect.
            return
        self.stats.add("capture", (t1 - t0) * 1000.0)

        detections = self.detector.detect(frame)
        t2 = time.perf_counter()
        self.stats.add("detect", (t2 - t1) * 1000.0)

        tracks = self.tracker.update(detections, now)
        t3 = time.perf_counter()
        self.stats.add("track", (t3 - t2) * 1000.0)

        command = self.controller.step(tracks, self.region.center, now)
        t4 = time.perf_counter()
        self.stats.add("control", (t4 - t3) * 1000.0)

        if self.enabled and self._input_permitted(now):
            if command.dx_counts or command.dy_counts:
                self.mouse.move_relative(command.dx_counts, command.dy_counts)
            if command.should_fire:
                self.mouse.click()

        self.stats.add("total", (time.perf_counter() - t0) * 1000.0)
        self.stats.frames += 1
        self.stats.detections += len(detections)

    def _input_permitted(self, now: float) -> bool:
        """Re-check the foreground window ~10x/second rather than every frame.

        The Win32 round trip is ~0.1 ms, which is a real fraction of a 4 ms budget,
        and no one alt-tabs faster than 100 ms.
        """
        if now - self._last_guard_check >= 0.1:
            self._last_guard_check = now
            window = guard.foreground_window()
            guard.assert_not_blocked(window)  # raises; never silently continues
            self._guard_ok = guard.is_allowed(window, self.cfg.guard)
        return self._guard_ok

    def run(self) -> int:
        cfg = self.cfg
        print(f"capture : {self.capture.describe()}")
        print(f"detector: {self.detector.describe()}")
        print(f"mouse   : {self.mouse.describe()}")
        print(f"view    : {cfg.aim.hfov_deg:.0f} deg hfov, {cfg.aim.deg_per_count:.5f} deg/count")

        window = guard.foreground_window()
        guard.assert_not_blocked(window)
        if not self._hotkeys.available:
            print(
                "\nnote: hotkeys and the foreground-window check are Windows-only.\n"
                "      On this platform the guard fails closed, so no input is sent.\n"
                "      Use `preview` and `bench` to work on the pipeline here."
            )
        print(f"\n{cfg.hotkeys.toggle}: toggle aiming   {cfg.hotkeys.quit}: quit")
        print("aiming is OFF until you toggle it.\n")

        next_report = time.perf_counter() + 2.0
        try:
            while self.running:
                if self._hotkeys.pressed("quit"):
                    break
                if self._hotkeys.pressed("toggle"):
                    self.enabled = not self.enabled
                    self.controller.reset()
                    self.tracker.reset()
                    print(f"aiming {'ON' if self.enabled else 'OFF'}")

                self.tick()

                now = time.perf_counter()
                if now >= next_report:
                    next_report = now + 2.0
                    state = "ON " if self.enabled else "OFF"
                    print(f"[{state}] {self.stats.report()}")
        except KeyboardInterrupt:
            pass
        except guard.BlockedApplicationError as exc:
            print(f"\nSTOP: {exc}", file=sys.stderr)
            return 2
        finally:
            self.capture.close()

        if isinstance(self.mouse, DryRunMouse):
            dx, dy = self.mouse.total
            print(f"\ndry run: {len(self.mouse.moves)} moves ({dx:+d}, {dy:+d} counts), {self.mouse.clicks} clicks")
        print(f"\n{self.stats.frames} frames | {self.stats.report()}")
        return 0
