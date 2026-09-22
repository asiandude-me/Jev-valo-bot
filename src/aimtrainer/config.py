"""Typed configuration, loaded from YAML or JSON.

Unknown keys are an error rather than a silent no-op -- a typo in a tuning file is
otherwise invisible and you spend an evening wondering why ``kp`` does nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class CaptureConfig:
    backend: str = "auto"          # auto | dxcam | mss
    monitor: int = 1               # 1-indexed, as mss numbers them
    roi_width: int = 320           # ROI side, centred on the crosshair
    roi_height: int = 320
    target_fps: int = 240          # dxcam capture rate ceiling
    screen_width: int = 1920       # full display, for the projection model
    screen_height: int = 1080


@dataclass
class HSVRange:
    """One inclusive HSV band. OpenCV hue is 0-179, not 0-359."""

    h_min: int
    h_max: int
    s_min: int = 120
    s_max: int = 255
    v_min: int = 120
    v_max: int = 255


@dataclass
class DetectorConfig:
    backend: str = "color"         # color | onnx
    # --- color backend ---
    hsv_ranges: list[HSVRange] = field(
        # Default: bright magenta/pink. Set your Aim Labs target colour to match, or
        # re-tune these with ``aimtrainer preview``.
        default_factory=lambda: [HSVRange(h_min=140, h_max=170)]
    )
    min_area_px: int = 40
    max_area_px: int = 40000
    min_extent: float = 0.45       # blob area / bounding-box area; a disc is ~0.785
    max_aspect: float = 2.5        # reject long thin artefacts (UI edges, trails)
    open_kernel: int = 3           # morphological opening, kills speckle
    close_kernel: int = 5          # closing, fills the crosshair hole in a target
    # --- onnx backend ---
    model_path: str = "models/targets.onnx"
    layout: str = "auto"           # auto | v8 (4+nc channels) | v5 (5+nc, objectness)
    input_size: int = 320
    conf_threshold: float = 0.45
    nms_threshold: float = 0.50
    providers: list[str] = field(
        default_factory=lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    class_ids: list[int] = field(default_factory=list)  # empty = keep every class


@dataclass
class TrackingConfig:
    max_assign_px: float = 90.0    # association gate; beyond this it's a new target
    max_missed_frames: int = 3     # drop a track after this many frames unseen
    velocity_alpha: float = 0.45   # EMA weight on new velocity samples
    min_hits_for_velocity: int = 2


@dataclass
class AimConfig:
    hfov_deg: float = 103.0
    deg_per_count: float = 0.0245  # MEASURE THIS: ``aimtrainer calibrate``
    kp: float = 0.50               # fraction of remaining error closed per tick
    kd: float = 0.08               # damping on error rate
    compensation_frames: int = 2   # frames of issued-but-unseen motion to subtract
    max_counts_per_tick: float = 220.0
    lead_ms: float = 8.0           # aim ahead along target velocity
    max_target_distance_px: float = 400.0   # ignore targets further than this
    stickiness: float = 1.35       # keep the current target unless a rival scores this much better
    settle_px: float = 2.0         # below this error, stop issuing corrections


@dataclass
class TriggerConfig:
    enabled: bool = False          # off by default; aim assist alone is the safer default
    deadzone_px: float = 6.0
    min_lock_frames: int = 2
    cooldown_ms: float = 90.0


@dataclass
class GuardConfig:
    # Substrings matched case-insensitively against the foreground window title and
    # process name. Empty means "never allow input" -- the gate fails closed.
    allow_window_substrings: list[str] = field(
        default_factory=lambda: ["aimlab", "aim lab", "kovaak", "fpsaimtrainer"]
    )
    require_foreground_match: bool = True


@dataclass
class HotkeyConfig:
    toggle: str = "F1"
    quit: str = "F2"


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    aim: AimConfig = field(default_factory=AimConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # deferred: only YAML configs need the dependency

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{p}: top level must be a mapping")
        return _build(cls, data, path=str(p))


def _build(cls: type, data: dict[str, Any], path: str, prefix: str = "") -> Any:
    """Recursively instantiate nested dataclasses, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = f"{path}:{prefix.rstrip('.')}" if prefix else path
        raise ValueError(
            f"{where}: unknown option(s) {sorted(unknown)}; "
            f"expected one of {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = known[name].type
        kwargs[name] = _coerce(ftype, value, path, f"{prefix}{name}.")
    return cls(**kwargs)


def _coerce(ftype: Any, value: Any, path: str, prefix: str) -> Any:
    # ``from __future__ import annotations`` leaves field types as strings, so match
    # on the names of the few nested types we actually have.
    type_name = ftype if isinstance(ftype, str) else getattr(ftype, "__name__", "")
    if "HSVRange" in type_name and isinstance(value, Sequence) and not isinstance(value, str):
        return [_build(HSVRange, v, path, prefix) for v in value]
    if isinstance(value, dict):
        nested = {c.__name__: c for c in _NESTED}
        for name, cls in nested.items():
            if name in type_name:
                return _build(cls, value, path, prefix)
    return value


_NESTED = [
    CaptureConfig,
    DetectorConfig,
    TrackingConfig,
    AimConfig,
    TriggerConfig,
    GuardConfig,
    HotkeyConfig,
    HSVRange,
]


def to_dict(obj: Any) -> Any:
    """Plain-data view of a config tree, for dumping and diffing."""
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj
