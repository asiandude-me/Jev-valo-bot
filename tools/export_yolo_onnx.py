#!/usr/bin/env python3
"""Export an Ultralytics YOLO checkpoint to ONNX for the `onnx` detector backend.

    pip install ultralytics
    python tools/export_yolo_onnx.py --weights yolov8n.pt --imgsz 320 --out models/targets.onnx

Use the smallest model that works -- on a 320px ROI, yolov8n is already overkill for
round targets, and every millisecond of inference is a millisecond of aim lag. If you
fine-tune on your own screenshots, export the fine-tuned `best.pt` the same way.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=320, help="must match detector.input_size")
    ap.add_argument("--out", default="models/targets.onnx")
    ap.add_argument("--half", action="store_true", help="fp16 weights (GPU providers only)")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    produced = Path(
        model.export(format="onnx", imgsz=args.imgsz, half=args.half, opset=args.opset, simplify=True)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != out.resolve():
        shutil.move(str(produced), out)
    print(f"wrote {out}")
    print(f"set detector.backend: onnx, detector.model_path: {out}, detector.input_size: {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
