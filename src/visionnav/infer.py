"""Run a pretrained YOLO object detector on one image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

import cv2
import torch
from ultralytics import YOLO


DEFAULT_MODEL = ROOT / "weights" / "yolo26n.pt"
DEFAULT_OUTPUT = ROOT / "outputs" / "detection.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "bus.jpg")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="CUDA device index (0) or cpu; defaults to CUDA when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.source}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(args.model))
    results = model.predict(
        source=str(args.source),
        device=args.device,
        imgsz=args.image_size,
        conf=args.confidence,
        verbose=False,
    )
    annotated = results[0].plot()
    if not cv2.imwrite(str(args.output), annotated):
        raise OSError(f"Could not write output image: {args.output}")

    count = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"Detections: {count}")
    print(f"Saved result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
