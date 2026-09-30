"""Download and prepare the public PASCAL VOC dataset inside this project."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

from ultralytics import settings  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402


def main() -> None:
    dataset_root = ROOT / "data" / "datasets"
    settings.update(
        {
            "datasets_dir": str(dataset_root),
            "weights_dir": str(ROOT / "weights"),
            "runs_dir": str(ROOT / "runs"),
        }
    )
    data = check_det_dataset("VOC.yaml")
    required = [
        dataset_root / "VOC" / "images" / "train2007",
        dataset_root / "VOC" / "images" / "train2012",
        dataset_root / "VOC" / "images" / "val2012",
        dataset_root / "VOC" / "images" / "val2007",
        dataset_root / "VOC" / "images" / "test2007",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("VOC preparation is incomplete:\n" + "\n".join(missing))
    print(f"VOC dataset ready under: {dataset_root / 'VOC'}")
    print(f"Official config path: {data['path']}")
    print("Protocol: train on VOC07 train + VOC12 train/val; validate on VOC07 val; test on VOC07 test.")


if __name__ == "__main__":
    main()
