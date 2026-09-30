"""Train house-grouped Robot@Home2 folds without using the held-out test home."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "datasets" / "RobotAtHome2" / "processed" / "yolo_cv_pilot"
DEFAULT_MODEL = ROOT / "weights" / "yolo26n.pt"
DEFAULT_PROJECT = ROOT / "runs" / "robotathome2_cv_smoke"
DEFAULT_SUMMARY = ROOT / "reports" / "robotathome2_cv_smoke_summary.md"
FOLDS = ("val_alma", "val_anto", "val_pare", "val_rx2")
METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=100,
                        help="Early-stopping epochs without mAP50 improvement (100 matches Ultralytics default).")
    parser.add_argument("--run-prefix", default="smoke",
                        help="Prefix for unique per-fold run directories, e.g. candidate_v1.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=("AdamW", "MuSGD", "SGD", "auto"), default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001429)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0)
    return parser.parse_args()


def render_summary(data: dict) -> str:
    rows = [
        f"# Robot@Home2 四折住宅分组训练：{data['run_prefix']}",
        "",
        f"- 开始时间：{data['started_at']}",
        f"- 权重：`{data['model']}`；配置：最多 epochs={data['epochs']}、patience={data['patience']}、imgsz={data['imgsz']}、batch={data['batch']}、optimizer={data['optimizer']}、lr0={data['lr0']}、warmup_bias_lr={data['warmup_bias_lr']}、device={data['device']}、seed={data['seed']}。",
        f"- 早停与 `best.pt` 选择指标：`{data['early_stopping_metric']}`；四折固定使用同一选择规则。",
        "- 每折只在各自 train 上训练、每轮用 val 验证；训练完成后再加载 `best.pt`，以统一参数独立运行一次 val。数据配置不含 test，sarmis 被排除。小数据集结果用于研究/诊断，不代表可部署性能。",
        "",
        "| 验证住宅 | train/val 图像 | 最佳 epoch | 独立验证 P/R | 独立 mAP50 | 独立 mAP50-95 | checkpoint mAP50 | 最佳权重 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fold in data["fold_results"]:
        metric = fold.get("standalone_validation_metrics") or fold["checkpoint_metrics"]
        checkpoint_metric = fold["checkpoint_metrics"]
        rows.append(
            f"| {fold['validation_home']} | {fold['train_images']}/{fold['val_images']} | "
            f"{fold.get('best_epoch', '—')} | {metric['precision']:.4f}/{metric['recall']:.4f} | {metric['mAP50']:.4f} | "
            f"{metric['mAP50_95']:.4f} | {checkpoint_metric['mAP50']:.4f} | `{fold['best_weights']}` |"
        )
    fold_metrics = [
        fold.get("standalone_validation_metrics") or fold["checkpoint_metrics"]
        for fold in data["fold_results"]
    ]
    if fold_metrics:
        count = len(fold_metrics)
        mean_p = sum(metric["precision"] for metric in fold_metrics) / count
        mean_r = sum(metric["recall"] for metric in fold_metrics) / count
        mean_ap50 = sum(metric["mAP50"] for metric in fold_metrics) / count
        mean_ap = sum(metric["mAP50_95"] for metric in fold_metrics) / count
        rows.append(
            f"| 等权宏平均（{count}/4 折） | — | — | {mean_p:.4f}/{mean_r:.4f} | "
            f"{mean_ap50:.4f} | {mean_ap:.4f} | — | — |"
        )
    rows.extend([
        "",
        "## 按留出住宅与类别的指标",
        "",
        "以下 P/R 是验证 PR 曲线上的类别 F1 最佳点；AP50/AP50–95 跨置信度阈值积分。阈值部署点需另在开发折上选取，并同时报告误报。",
        "",
        "| 留出住宅 | 类别 | P | R | AP50 | AP50–95 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for fold in data["fold_results"]:
        per_class = fold.get("standalone_per_class_metrics", {})
        for class_name in data.get("classes", []):
            metric = per_class.get(class_name)
            if metric:
                rows.append(
                    f"| {fold['validation_home']} | {class_name} | {metric['precision']:.4f} | "
                    f"{metric['recall']:.4f} | {metric['AP50']:.4f} | {metric['AP50_95']:.4f} |"
                )
    for class_name in data.get("classes", []):
        class_metrics = [
            fold.get("standalone_per_class_metrics", {}).get(class_name)
            for fold in data["fold_results"]
        ]
        class_metrics = [metric for metric in class_metrics if metric]
        if class_metrics:
            count = len(class_metrics)
            mean = {
                key: sum(metric[key] for metric in class_metrics) / count
                for key in ("precision", "recall", "AP50", "AP50_95")
            }
            rows.append(
                f"| 等权宏平均（{count}/4 折） | {class_name} | {mean['precision']:.4f} | "
                f"{mean['recall']:.4f} | {mean['AP50']:.4f} | {mean['AP50_95']:.4f} |"
            )
    rows.extend([
        "",
        "结果应按留出住宅分别理解。独立验证指标是本报告的统一比较口径；checkpoint 内嵌训练期指标仅作为交叉核对。每折样本仍少，不能据此宣称达到实用精度；四折差异用于发现住宅域偏移和难例。",
        "",
    ])
    return "\n".join(rows)


def add_macro_averages(data: dict) -> None:
    """Persist equal-weight fold means, keeping house-level results visible separately."""
    fold_metrics = [
        fold.get("standalone_validation_metrics") or fold.get("checkpoint_metrics")
        for fold in data.get("fold_results", [])
    ]
    fold_metrics = [metric for metric in fold_metrics if metric]
    if fold_metrics:
        keys = ("precision", "recall", "mAP50", "mAP50_95")
        data["macro_average"] = {
            key: sum(metric[key] for metric in fold_metrics) / len(fold_metrics)
            for key in keys
        }
        data["macro_average"]["folds_included"] = len(fold_metrics)
    per_class = {}
    for class_name in data.get("classes", []):
        values = [
            fold.get("standalone_per_class_metrics", {}).get(class_name)
            for fold in data.get("fold_results", [])
        ]
        values = [metric for metric in values if metric]
        if values:
            per_class[class_name] = {
                key: sum(metric[key] for metric in values) / len(values)
                for key in ("precision", "recall", "AP50", "AP50_95")
            }
            per_class[class_name]["folds_included"] = len(values)
    data["per_class_macro_average"] = per_class


def save_summary(data: dict, summary_path: Path) -> None:
    add_macro_averages(data)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_summary(data), encoding="utf-8")
    summary_path.with_suffix(".json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_checkpoint_metrics(checkpoint_path: Path) -> dict:
    """Read validation metrics embedded in the selected best.pt checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_metrics = checkpoint.get("train_metrics") or {}
    missing = [column for column in METRIC_COLUMNS.values() if column not in saved_metrics]
    if missing:
        raise ValueError(f"Best checkpoint lacks validation metrics {missing}: {checkpoint_path}")
    return {name: float(saved_metrics[column]) for name, column in METRIC_COLUMNS.items()}


def read_checkpoint_epoch(checkpoint_path: Path) -> int:
    """Resolve selected epoch; Ultralytics strips the checkpoint epoch after training."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    if epoch is not None and int(epoch) >= 0:
        return int(epoch) + 1

    saved_metrics = checkpoint.get("train_metrics") or {}
    columns = {
        "metrics/precision(B)": "metrics/precision(B)",
        "metrics/recall(B)": "metrics/recall(B)",
        "metrics/mAP50(B)": "metrics/mAP50(B)",
        "metrics/mAP50-95(B)": "metrics/mAP50-95(B)",
    }
    missing = [key for key in columns.values() if key not in saved_metrics]
    results_path = checkpoint_path.parents[1] / "results.csv"
    if missing or not results_path.is_file():
        raise ValueError(f"Cannot resolve best epoch from checkpoint/CSV: {checkpoint_path}")
    matches = []
    with results_path.open(encoding="utf-8-sig", newline="") as results_file:
        for row in csv.DictReader(results_file):
            if all(
                math.isclose(float(row[column]), float(saved_metrics[key]), rel_tol=0.0, abs_tol=2e-5)
                for column, key in columns.items()
            ):
                matches.append(int(row["epoch"]))
    if not matches:
        raise ValueError(f"Checkpoint validation metrics do not match any CSV epoch: {checkpoint_path}")
    # A repeated metric tie can save the later checkpoint; the retained row is the best available evidence.
    return matches[-1]


def standalone_class_metrics(validation, class_names: list[str]) -> dict:
    """Return class-wise P/R at best F1 plus threshold-integrated AP metrics."""
    box = validation.box
    result = {}
    for metric_index, class_index in enumerate(box.ap_class_index):
        class_name = class_names[int(class_index)]
        precision, recall, ap50, ap50_95 = box.class_result(metric_index)
        result[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "AP50": float(ap50),
            "AP50_95": float(ap50_95),
        }
    return result


def validate_fold_config(fold_root: Path, data_yaml: Path, manifest: dict) -> None:
    """Fail closed if a fold config points outside its train/val-only dataset."""
    import yaml

    dataset_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(dataset_config, dict) or "test" in dataset_config:
        raise ValueError(f"Fold {fold_root.name} data.yaml must be a mapping without a test key")
    resolved_fold_root = fold_root.resolve()
    declared_root = Path(dataset_config.get("path", "")).resolve()
    if declared_root != resolved_fold_root:
        raise ValueError(f"Fold {fold_root.name} data.yaml points outside its own data directory: {declared_root}")
    if dataset_config.get("names") != manifest.get("classes"):
        raise ValueError(f"Fold {fold_root.name} class names/order disagree between data.yaml and manifest")
    for split_name in ("train", "val"):
        split_value = dataset_config.get(split_name, "")
        if not isinstance(split_value, str) or not split_value:
            raise ValueError(f"Fold {fold_root.name} {split_name} path must be a non-empty string")
        split_path = Path(split_value)
        resolved_split_path = (
            split_path.resolve() if split_path.is_absolute()
            else (declared_root / split_path).resolve()
        )
        if not resolved_split_path.is_relative_to(resolved_fold_root):
            raise ValueError(f"Fold {fold_root.name} {split_name} path escapes its fold directory: {resolved_split_path}")
        split_manifest = manifest["splits"][split_name]
        expected_ids = {str(item["observation_id"]) for item in split_manifest["items"]}
        if resolved_split_path.is_dir():
            images = sorted(resolved_split_path.glob("*.*"))
            found_ids = {image.stem for image in images if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}}
            if found_ids != expected_ids:
                raise ValueError(f"Fold {fold_root.name} {split_name} image directory disagrees with manifest")
            for image in images:
                if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                    label = Path(str(image).replace("images", "labels", 1)).with_suffix(".txt")
                    if not label.is_file():
                        raise FileNotFoundError(f"Missing YOLO label for {image}: {label}")
        elif resolved_split_path.is_file():
            listed = [Path(line.strip()) for line in resolved_split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            resolved_images = [path.resolve() if path.is_absolute() else (resolved_split_path.parent / path).resolve() for path in listed]
            if len(resolved_images) != len(set(resolved_images)):
                raise ValueError(f"Fold {fold_root.name} {split_name} image list contains duplicates")
            if any(not image.is_relative_to(resolved_fold_root.parent / "images" / "all") or not image.is_file() for image in resolved_images):
                raise ValueError(f"Fold {fold_root.name} {split_name} image list contains a missing/out-of-dataset path")
            found_ids = {image.stem for image in resolved_images}
            if found_ids != expected_ids or len(resolved_images) != split_manifest["image_count"]:
                raise ValueError(f"Fold {fold_root.name} {split_name} image list disagrees with manifest")
            for image in resolved_images:
                label = (resolved_fold_root.parent / "labels" / "all" / f"{image.stem}.txt").resolve()
                if not label.is_file():
                    raise FileNotFoundError(f"Missing YOLO label for listed image {image}: {label}")
        else:
            raise FileNotFoundError(f"Fold {fold_root.name} {split_name} path does not exist: {resolved_split_path}")


def validate_fold_manifest(fold_name: str, manifest: dict) -> tuple[set[str], set[str]]:
    """Verify the declared and per-sample home groups contain no test leakage."""
    split_records = manifest.get("splits", {})
    if set(split_records) != {"train", "val"} or manifest.get("excluded_final_test_home") != "sarmis":
        raise ValueError(f"Fold {fold_name} must exclude sarmis and contain only train/val splits")
    train_homes = set(manifest.get("train_homes", []))
    split_homes = manifest.get("split_homes", {})
    val_homes = set(split_homes.get("val", []))
    if "sarmis" in train_homes or "sarmis" in val_homes or manifest.get("validation_home") == "sarmis":
        raise ValueError(f"Final test home sarmis must not enter fold {fold_name}")
    if set(split_homes.get("train", [])) != train_homes:
        raise ValueError(f"Fold {fold_name} has inconsistent train-home metadata")
    if manifest.get("validation_home") not in val_homes:
        raise ValueError(f"Fold {fold_name} has inconsistent validation-home metadata")
    if train_homes & val_homes:
        raise ValueError(f"Fold {fold_name} leaks a home between train and val")
    for split_name, expected_homes in (("train", train_homes), ("val", val_homes)):
        split_manifest = split_records[split_name]
        if set(split_manifest.get("homes", [])) != expected_homes:
            raise ValueError(f"Fold {fold_name} {split_name} split-home metadata is inconsistent")
        observed_homes = {item.get("home") for item in split_manifest.get("items", [])}
        if not observed_homes <= expected_homes or "sarmis" in observed_homes:
            raise ValueError(f"Fold {fold_name} {split_name} samples include an unexpected/final-test home")
    return train_homes, val_homes


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.patience < 0 or args.imgsz < 32 or args.batch < 1
            or args.workers < 0 or args.lr0 <= 0 or args.warmup_bias_lr < 0):
        raise ValueError("epochs/batch/lr0 must be positive; patience/workers/warmup_bias_lr cannot be negative; imgsz must be at least 32")
    if not args.run_prefix or any(character in args.run_prefix for character in "\\/:*?\"<>| "):
        raise ValueError("--run-prefix must be a non-empty path-safe token without spaces")
    data_root = args.data_root.resolve()
    model_path = args.model.resolve()
    project = args.project.resolve()
    summary_path = args.summary.resolve()
    if not data_root.is_relative_to(ROOT) or not project.is_relative_to(ROOT) or not summary_path.is_relative_to(ROOT):
        raise ValueError(f"Dataset, run outputs, and reports must remain under project root {ROOT}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    if summary_path.exists() or summary_path.with_suffix(".json").exists():
        raise FileExistsError(f"Summary exists; refusing to overwrite: {summary_path}")

    fold_inputs = []
    for fold_name in args.folds:
        fold_root = data_root / fold_name
        manifest_path = fold_root / "manifest.json"
        data_yaml = fold_root / "data.yaml"
        if not manifest_path.is_file() or not data_yaml.is_file():
            raise FileNotFoundError(f"Fold not prepared: {fold_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_homes, _ = validate_fold_manifest(fold_name, manifest)
        validate_fold_config(fold_root, data_yaml, manifest)

        run_name = f"{args.run_prefix}_{args.epochs}ep_{fold_name}"
        if (project / run_name).exists():
            raise FileExistsError(f"Run directory exists; refusing to reuse or overwrite: {project / run_name}")
        if (project / "standalone_val" / run_name).exists():
            raise FileExistsError(f"Standalone validation directory exists; refusing to reuse or overwrite: {project / 'standalone_val' / run_name}")
        fold_inputs.append((fold_name, fold_root, data_yaml, manifest, run_name))

    # Keep Ultralytics settings, font cache and plots within the user's project directory.
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "Ultralytics")
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    class MAp50DetectionTrainer(DetectionTrainer):
        """Use validation mAP50 consistently for early stopping and best.pt selection."""

        def validate(self):
            metrics, _ = super().validate()
            if metrics is None:
                return None, None
            metric_name = METRIC_COLUMNS["mAP50"]
            if metric_name not in metrics:
                raise KeyError(f"Validation metrics lack the configured early-stop key: {metric_name}")
            fitness = float(metrics[metric_name])
            if not math.isfinite(fitness):
                raise ValueError(f"Non-finite validation mAP50 cannot drive checkpoint selection: {fitness}")
            # BaseTrainer.validate() first updates this using Ultralytics' composite fitness;
            # replace it so EarlyStopping and save_model() both use the same explicit mAP50.
            previous_best = getattr(self, "best_map50", 0.0)
            self.best_map50 = max(previous_best, fitness)
            self.best_fitness = self.best_map50
            return metrics, fitness

    if str(args.device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    summary = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "model": str(model_path),
        "data_root": str(data_root),
        "project": str(project),
        "epochs": args.epochs,
        "patience": args.patience,
        "run_prefix": args.run_prefix,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(args.device),
        "workers": args.workers,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "warmup_bias_lr": args.warmup_bias_lr,
        "early_stopping_metric": "mAP50",
        "classes": fold_inputs[0][3]["classes"],
        "test_used": False,
        "fold_results": [],
    }
    save_summary(summary, summary_path)

    for fold_name, fold_root, data_yaml, manifest, run_name in fold_inputs:
        print(f"\n=== Training {fold_name}; train homes={manifest['train_homes']}; val={manifest['validation_home']} ===")
        model = YOLO(str(model_path))
        model.train(
            trainer=MAp50DetectionTrainer,
            data=str(data_yaml),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            optimizer=args.optimizer,
            lr0=args.lr0,
            warmup_bias_lr=args.warmup_bias_lr,
            deterministic=True,
            project=str(project),
            name=run_name,
            split="val",
            val=True,
            plots=True,
            cache=False,
            exist_ok=False,
        )
        save_dir = Path(model.trainer.save_dir).resolve()
        best_weights = save_dir / "weights" / "best.pt"
        if not (save_dir / "results.csv").is_file():
            raise FileNotFoundError(f"Training did not produce results.csv: {save_dir / 'results.csv'}")
        metrics = read_checkpoint_metrics(best_weights)
        row = {
            "fold": fold_name,
            "validation_home": manifest["validation_home"],
            "train_homes": manifest["train_homes"],
            "train_images": manifest["splits"]["train"]["image_count"],
            "val_images": manifest["splits"]["val"]["image_count"],
            "checkpoint_metrics": metrics,
            "best_epoch": read_checkpoint_epoch(best_weights),
            "run_dir": str(save_dir),
            "best_weights": str(best_weights),
        }
        if not Path(row["best_weights"]).is_file():
            raise FileNotFoundError(f"Training did not produce best.pt: {row['best_weights']}")
        summary["fold_results"].append(row)
        save_summary(summary, summary_path)

        # Re-evaluate every selected best checkpoint with one fixed, standalone protocol.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_model = YOLO(str(best_weights))
        validation = val_model.val(
            data=str(data_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            plots=False,
            project=str(project / "standalone_val"),
            name=run_name,
            exist_ok=False,
            verbose=False,
        )
        standalone = validation.results_dict
        row["standalone_validation_metrics"] = {
            "precision": float(standalone[METRIC_COLUMNS["precision"]]),
            "recall": float(standalone[METRIC_COLUMNS["recall"]]),
            "mAP50": float(standalone[METRIC_COLUMNS["mAP50"]]),
            "mAP50_95": float(standalone[METRIC_COLUMNS["mAP50_95"]]),
        }
        row["standalone_per_class_metrics"] = standalone_class_metrics(validation, manifest["classes"])
        row["standalone_validation_dir"] = str(validation.save_dir)
        save_summary(summary, summary_path)
        del val_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Fold summary: standalone={row['standalone_validation_metrics']}; checkpoint={metrics}; best={row['best_weights']}")

    summary["completed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    save_summary(summary, summary_path)
    print(f"Summary: {summary_path}")
    print(f"JSON: {summary_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
