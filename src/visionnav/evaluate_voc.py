"""Evaluate the trained detector on VOC07 validation and write a class report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

import torch  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402


MODEL_CANDIDATES = (
    ROOT / "runs" / "voc_yolo26n_baseline" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "runs" / "voc_yolo26n_baseline" / "weights" / "best.pt",
)
DEFAULT_MODEL = next((path for path in MODEL_CANDIDATES if path.is_file()), MODEL_CANDIDATES[0])
DATA_CONFIG = ROOT / "data" / "datasets" / "voc.yaml"
REPORT_DIR = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Model checkpoint (default: {DEFAULT_MODEL.relative_to(ROOT)})",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def project_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_CONFIG}")

    settings.update(
        {
            "datasets_dir": str(ROOT / "data" / "datasets"),
            "weights_dir": str(ROOT / "weights"),
            "runs_dir": str(ROOT / "runs"),
        }
    )
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(DATA_CONFIG),
        split="val",  # Keep this tool away from the already-used VOC07 test split.
        imgsz=args.image_size,
        batch=args.batch,
        device=device,
        project=str(ROOT / "outputs"),
        name="voc_yolo26n_val_audit",
        exist_ok=True,
        plots=True,
    )

    names = metrics.names
    class_ids_with_metrics = [int(class_id) for class_id in metrics.box.ap_class_index]
    per_class = []
    for class_id, class_name in sorted(names.items()):
        class_id = int(class_id)
        if class_id not in class_ids_with_metrics:
            per_class.append(
                {"class_id": class_id, "class": class_name, "precision": None, "recall": None, "map50": None, "map50_95": None}
            )
            continue
        values = metrics.box.class_result(class_ids_with_metrics.index(class_id))
        per_class.append(
            {
                "class_id": class_id,
                "class": class_name,
                "precision": float(values[0]),
                "recall": float(values[1]),
                "map50": float(values[2]),
                "map50_95": float(values[3]),
            }
        )

    results = {key: float(value) for key, value in metrics.results_dict.items()}
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": "VOC2007 val (validation only)",
        "model": project_relative_path(model_path),
        "dataset_config": project_relative_path(DATA_CONFIG),
        "device": torch.cuda.get_device_name(0) if device != "cpu" else "CPU",
        "image_size": args.image_size,
        "batch": args.batch,
        "aggregate_metrics": results,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "per_class_metrics": per_class,
        "warning": "Use this validation split for analysis/tuning. Do not use the previously evaluated VOC2007 test split to choose changes.",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "voc_yolo26n_val_audit.json"
    markdown_path = REPORT_DIR / "voc_yolo26n_val_audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    aggregate = metrics.box.mean_results()
    speed = payload["speed_ms_per_image"]
    rows = [
        "# VOC2007 Validation Audit",
        "",
        f"- Created: {payload['created_at']}",
        "- Split: VOC2007 val only (2,510 images); the VOC2007 test split is not used by this report.",
        f"- Model: `{model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path}`",
        f"- Device: {payload['device']}; image size: {args.image_size}; batch: {args.batch}",
        f"- Aggregate: Precision {aggregate[0]:.4f}, Recall {aggregate[1]:.4f}, mAP@0.5 {aggregate[2]:.4f}, mAP@0.5:0.95 {aggregate[3]:.4f}.",
        f"- Speed (ms/image): preprocess {speed.get('preprocess', 0):.3f}, inference {speed.get('inference', 0):.3f}, postprocess {speed.get('postprocess', 0):.3f}.",
        "- Per-class scores describe where to inspect examples; they do not by themselves establish why errors occur.",
        "- Do not tune against the VOC2007 test split, which has already been used for the previous final evaluation.",
        "",
        "| Class | Precision | Recall | AP50 | AP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in per_class:
        values = [row[key] for key in ("precision", "recall", "map50", "map50_95")]
        cells = ["—" if value is None else f"{value:.4f}" for value in values]
        rows.append(f"| {row['class']} | " + " | ".join(cells) + " |")
    rows.extend(
        [
            "",
            "Validation plots and prediction/label samples: `outputs/voc_yolo26n_val_audit/`.",
        ]
    )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Validation-only report: {markdown_path}")
    print(f"Machine-readable metrics: {json_path}")
    print(f"Validation plots: {ROOT / 'outputs' / 'voc_yolo26n_val_audit'}")


if __name__ == "__main__":
    main()
