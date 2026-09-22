# aimtrainer-bot

A real-time computer-vision agent that plays **Aim Labs**: it captures a small region
of the screen, detects targets with an on-device model, predicts where they are going,
and drives the mouse to them.

Everything runs locally. No network calls, no cloud inference.

## Scope

This project targets **aim trainers only** — Aim Labs, Kovaak's, aimtrainer-style web
tasks. It is a testbed for low-latency perception + closed-loop control.

It deliberately **does not** support live multiplayer shooters. `guard.py` holds a
hard-coded block list of competitive online games; if one of them owns the foreground
window, the runner refuses to emit input and exits. That gate is not configurable, and
the input layer fails closed: if the foreground window can't be identified, or doesn't
match the allow list, no mouse events are sent.

Aim Labs has online leaderboards. Keep bot runs off them — use custom/unranked tasks.

## How it works

```
  capture (ROI around crosshair)     ~0.3-1.5 ms   dxcam / mss
      -> detect (targets in ROI)     ~0.5-6   ms   colour threshold | ONNX YOLO
      -> track  (associate + velocity)              nearest-neighbour + gated EMA
      -> select (stickiest best target)
      -> control (PD -> mouse counts)               dead-time compensated
      -> SendInput relative move (+ click)
```

The whole loop is budgeted so end-to-end latency stays under one display frame at
240 Hz. Capturing a 320x320 ROI instead of the full screen is where most of that comes
from: measured on a CPU-only container, detect + track + control is 0.49 ms at p50
with 100% recall and 0.37 px mean centre error on synthetic frames (`bench`).

### The part that actually decides whether it works

The frame you are looking at is one to three frames old, so some of the correction you
already sent has not appeared in it yet. Acting on the raw error double-counts that
motion: the crosshair overshoots, corrects, overshoots again. It is not subtle -- in
simulation at a two-frame delay, a gain of 0.6 rings past the target by 200 px and a
gain of 0.9 diverges outright.

The controller therefore keeps the counts it has issued but not yet seen and subtracts
them before acting (a Smith predictor). With that, the same 0.9 gain settles in four
frames with no overshoot at all. This matters far more than gain tuning, which is why
`aim.compensation_frames` is the second thing to set after calibration.

### Two detector backends

| backend | latency | when to use |
| --- | --- | --- |
| `color` | ~0.5 ms | default. Aim Labs targets are a flat, user-chosen colour — thresholding in HSV is exact and unbeatably cheap. |
| `onnx` | ~3-8 ms | drop in any YOLO-family `.onnx`. Runs on TensorRT / CUDA / DirectML / CPU via ONNX Runtime. Use this if you want a learned detector, or targets the colour path can't separate. Both v5 (`5+nc`, objectness) and v8/v11 (`4+nc`) output layouts decode; set `detector.layout` if auto-detection guesses wrong. |

The `color` backend is not a fallback — for this task it is both faster *and* more
accurate than a neural detector. The ONNX path exists because "run an open-weights
model on your own hardware" is the more general and more interesting setup.

## Install

```bash
python -m pip install -e .            # core: numpy, opencv, mss, pyyaml
python -m pip install -e ".[windows]" # + dxcam (much faster capture)
python -m pip install -e ".[onnx]"    # + onnxruntime (swap for onnxruntime-gpu / -directml)
```

Windows is the primary platform: relative mouse movement goes through `SendInput`,
which is what a raw-input game reads. The pure-logic layers (geometry, tracking,
control, detectors) are platform-independent and fully tested on Linux.

## Quick start

```bash
# 1. See what the detector sees, without touching the mouse.
python -m aimtrainer preview --config configs/aimlabs_gridshot.yaml

# 2. Measure your real mouse sensitivity (do this once, it matters more than any gain).
python -m aimtrainer calibrate --config configs/aimlabs_gridshot.yaml

# 3. Run it. F1 toggles aiming, F2 quits.
python -m aimtrainer run --config configs/aimlabs_gridshot.yaml

# 4. Latency numbers for your machine.
python -m aimtrainer bench --config configs/aimlabs_gridshot.yaml
```

`run` starts with aiming **disabled**; press F1 in Aim Labs to enable. Add `--dry-run`
to log the mouse commands it *would* send without sending them.

### Tuning, in the order that matters

1. **`aim.deg_per_count`** — run `calibrate`. A 10% error here caps your accuracy no
   matter what else you do.
2. **`detector.hsv_ranges`** — set your Aim Labs target colour to something saturated
   and unique (bright magenta works well), then `preview` until the mask is clean.
3. **`aim.compensation_frames`** — how many frames of your own motion are not yet
   visible. Roughly (end-to-end latency / frame interval); `bench --live` prints the
   latency. **Round down if unsure**: guessing low degrades gently, guessing high
   makes the loop ring and stall short of the target.
4. **`aim.kp`** — fraction of remaining error closed per tick. 0.5 is a sane start.
   With compensation set correctly you can push it toward 0.9; without it, anything
   above ~0.35 will ring.
5. **`aim.lead_ms`** — how far ahead to aim along the target's velocity. Set it near
   your measured end-to-end latency (`bench` prints it). Compensation handles the
   crosshair's unseen motion; this handles the target's.
6. **`aim.kd`** — small damping term for target acceleration and detection jitter.
   Leave it low; raising it mostly amplifies noise.

## Calibration, precisely

`deg_per_count` is degrees of view rotation per mouse count. The calibrator locks a
static target, issues a known horizontal move of N counts, re-detects, and solves

```
deg_per_count = (yaw(x_after) - yaw(x_before)) / N,    yaw(x) = atan2(x - cx, f)
```

where `f = (width/2) / tan(hfov/2)`. It repeats in both directions and reports the
median plus spread. A spread over ~2% usually means mouse acceleration is on somewhere
in Windows or in-game — turn it off, the model assumes a linear device.

Starting points if you want to skip calibration: `0.07 * sens` for a Valorant-style
sensitivity profile, `0.022 * sens` for a Source-style one. Verify them anyway.

## Config

See `configs/aimlabs_gridshot.yaml` — every field is commented. Sections: `capture`,
`detector`, `tracking`, `aim`, `trigger`, `guard`, `hotkeys`.

## Tests

```bash
python -m pytest -q
```

171 tests, no display or Windows API needed. They cover the projection maths, the
colour detector against synthetic frames, track association and velocity estimation,
dead-time compensation (including what happens when you mis-configure it), sub-pixel
accumulation, the guard's block list, and the full loop end to end. The calibrator is
checked against a simulated display with a known sensitivity, and the ONNX backend
against real ONNX Runtime sessions driven by purpose-built models.

Install the optional extras to run every test; the ONNX ones skip themselves if
`onnx` and `onnxruntime` are missing.

## Layout

```
src/aimtrainer/
  geometry.py      pixel offset <-> view angle <-> mouse counts
  capture.py       dxcam / mss / in-memory backends
  detectors/       color.py (HSV blobs), onnx_yolo.py (YOLO via ONNX Runtime)
  tracking.py      multi-target association + velocity
  control.py       target selection, PD + dead-time compensation, trigger logic
  mouse.py         SendInput / pynput / dry-run
  guard.py         foreground-window gate and block list
  runner.py        the loop, hotkeys, latency stats
tools/export_yolo_onnx.py   ultralytics -> onnx
```
