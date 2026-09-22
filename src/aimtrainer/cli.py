"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, guard
from .capture import CaptureUnavailableError
from .config import Config

DEFAULT_CONFIG = "configs/aimlabs_gridshot.yaml"


class UserError(Exception):
    """A problem the user can fix. Reported as one line, without a traceback."""


def _load(path: str) -> Config:
    p = Path(path)
    if not p.is_file():
        if path == DEFAULT_CONFIG:
            print(f"no config at {path}; using built-in defaults", file=sys.stderr)
            return Config()
        raise UserError(f"config not found: {path}")
    try:
        return Config.load(p)
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    except ImportError as exc:
        raise UserError(f"{p}: reading YAML needs PyYAML (pip install PyYAML) -- {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="aimtrainer",
        description="Computer-vision agent for aim trainers (Aim Labs, Kovaak's).",
        epilog="Aim trainers only. Live competitive games are blocked; see aimtrainer/guard.py.",
    )
    ap.add_argument("--version", action="version", version=f"aimtrainer {__version__}")
    ap.add_argument("--traceback", action="store_true", help="show the full traceback on error")
    sub = ap.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML or JSON config file")
        return p

    run = with_config(sub.add_parser("run", help="run the aim loop"))
    run.add_argument("--dry-run", action="store_true", help="log mouse commands instead of sending them")

    prev = with_config(sub.add_parser("preview", help="show detections and the colour mask; sends no input"))
    prev.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until q)")
    prev.add_argument("--save", metavar="DIR", help="also write annotated frames here")

    cal = with_config(sub.add_parser("calibrate", help="measure aim.deg_per_count"))
    cal.add_argument("--trials", type=int, default=8)
    cal.add_argument("--settle-ms", type=float, default=80.0, help="wait after each probe move")

    ben = with_config(sub.add_parser("bench", help="latency and accuracy benchmark"))
    ben.add_argument("--frames", type=int, default=400)
    ben.add_argument("--live", action="store_true", help="benchmark real screen capture")

    with_config(sub.add_parser("doctor", help="check the environment and report what is available"))
    return ap


def cmd_doctor(cfg: Config) -> int:
    import platform

    print(f"aimtrainer {__version__} on {platform.system()} {platform.release()}, python {platform.python_version()}")

    def probe(label: str, fn) -> None:
        try:
            print(f"  {label:<22} {fn()}")
        except Exception as exc:
            print(f"  {label:<22} unavailable ({type(exc).__name__}: {exc})")

    print("\nbackends")
    probe("dxcam", lambda: __import__("dxcam").__name__ + " ok")
    probe("mss", lambda: "ok")
    probe("opencv", lambda: __import__("cv2").__version__)
    probe("onnxruntime", lambda: ", ".join(__import__("onnxruntime").get_available_providers()))

    print("\nguard")
    window = guard.foreground_window()
    if window is None:
        print("  foreground window   not readable on this platform -> input disabled (fail closed)")
    else:
        print(f"  foreground window   {window.title!r} ({window.process})")
        print(f"  blocked             {guard.is_blocked(window)}")
        print(f"  input allowed       {guard.is_allowed(window, cfg.guard)}")

    print("\naim model")
    from .runner import view_model

    vm = view_model(cfg)
    print(f"  focal               {vm.focal_px:.1f}px  ({cfg.aim.hfov_deg:.0f} deg h / {vm.vfov_deg:.1f} deg v)")
    print(f"  deg_per_count       {cfg.aim.deg_per_count:.6f}")
    print(f"  full 360 turn       {360.0 / cfg.aim.deg_per_count:.0f} counts")
    if cfg.aim.deg_per_count == 0.0245:
        print("  (that is the default -- run `aimtrainer calibrate` to measure yours)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except guard.BlockedApplicationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    except (UserError, CaptureUnavailableError, FileNotFoundError, RuntimeError, ImportError) as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        print("(re-run with --traceback for the full stack)", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    cfg = _load(args.config)

    if args.command == "run":
        from .runner import Runner

        return Runner(cfg, dry_run=args.dry_run).run()

    if args.command == "preview":
        from .preview import run_preview

        return run_preview(cfg, frames=args.frames, save_dir=args.save)

    if args.command == "calibrate":
        from .calibrate import calibrate, format_result

        guard.assert_not_blocked(guard.foreground_window())
        result = calibrate(cfg, trials=args.trials, settle_ms=args.settle_ms)
        print(format_result(result, cfg))
        return 0 if result.ok else 1

    if args.command == "bench":
        from .bench import bench_live, bench_synthetic

        return bench_live(cfg, args.frames) if args.live else bench_synthetic(cfg, args.frames)

    if args.command == "doctor":
        return cmd_doctor(cfg)

    raise UserError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
