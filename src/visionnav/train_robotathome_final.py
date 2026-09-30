"""Train the frozen final Robot@Home2 monocular detector on all dev observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from visionnav.prepare_robotathome_final import CLASSES, DEFAULT_OUTPUT, ROOT, sha256


CV_ROOT = ROOT / "runs" / "robotathome2_cv_candidate_v2"
CV_SUMMARY = ROOT / "reports" / "robotathome2_cv_candidate_v2.json"
FINAL_MODEL = ROOT / "weights" / "yolo26n.pt"
FINAL_RUNS = ROOT / "runs" / "robotathome2_final"
FINAL_SUMMARY = ROOT / "reports" / "robotathome2_final_model_v2.json"
FOLDS = ("alma", "anto", "pare", "rx2")
METRIC_COLUMN = "metrics/mAP50-95(B)"


def best_epoch_from_csv(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing v2 cross-validation results.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or METRIC_COLUMN not in rows[0]:
        raise ValueError(f"Missing selection metric {METRIC_COLUMN!r} in {path}")
    valid = []
    for index, row in enumerate(rows, 1):
        try:
            epoch = int(float(row.get("epoch", index)))
            metric = float(row[METRIC_COLUMN])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid epoch/metric row {index} in {path}") from exc
        if metric == metric and abs(metric) != float("inf"):
            valid.append((metric, epoch))
    if not valid:
        raise ValueError(f"No finite selection metrics in {path}")
    return max(valid, key=lambda pair: (pair[0], -pair[1]))[1]


def validate_training_inputs(data_root: Path) -> dict:
    if not data_root.is_relative_to(ROOT) or not data_root.is_dir():
        raise ValueError(f"Merged development data must exist inside the project: {data_root}")
    manifest_path = data_root / "manifest.json"
    data_path = data_root / "data.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if manifest.get("sarmis_included") is not False or manifest.get("audit", {}).get("excluded_home") != "sarmis":
        raise ValueError("Final development dataset manifest indicates sarmis contamination")
    if any(item.get("home") == "sarmis" for item in manifest.get("items", [])):
        raise ValueError("Final test home is present in development data")
    if manifest.get("classes") != list(CLASSES) or manifest.get("audit", {}).get("unique_observations") != len(manifest.get("items", [])):
        raise ValueError("Merged dataset manifest has invalid class order or sample count")
    if not isinstance(config, dict) or "test" in config or config.get("names") != list(CLASSES):
        raise ValueError("Final training data.yaml must have the frozen classes and no test key")
    resolved_root = Path(config.get("path", "")).resolve()
    if resolved_root != data_root.resolve() or config.get("train") != "images/train" or config.get("val") != "images/train":
        raise ValueError("Final training YAML must point to its own all-development train set only")
    ids = set()
    for item in manifest["items"]:
        observation_id = str(item.get("observation_id", ""))
        if not observation_id.isdigit() or observation_id in ids or item.get("home") not in FOLDS:
            raise ValueError(f"Duplicate/invalid development item or unexpected home: {item}")
        ids.add(observation_id)
        if item.get("home") == "sarmis":
            raise ValueError("Final test home is present in development data")
        image = data_root / "images" / "train" / f"{observation_id}.png"
        label = data_root / "labels" / "train" / f"{observation_id}.txt"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing merged development image/label for {observation_id}")
        if sha256(image) != item.get("image_sha256") or sha256(label) != item.get("label_sha256"):
            raise ValueError(f"Merged image/label hash differs from audited source for {observation_id}")
    if len(ids) != manifest["audit"]["unique_observations"]:
        raise ValueError("Merged image manifest count mismatch")
    cv_summary = json.loads(CV_SUMMARY.read_text(encoding="utf-8"))
    if cv_summary.get("test_used") is not False or cv_summary.get("run_prefix") != "candidate_v2":
        raise ValueError("Candidate-v2 training summary must show a completed development-only run")
    for key, expected in (("imgsz", 640), ("batch", 8), ("seed", 42), ("patience", 12), ("epochs", 50)):
        if cv_summary.get(key) != expected:
            raise ValueError(f"Candidate-v2 training config changed: expected {key}={expected}, got {cv_summary.get(key)}")
    best_epochs = {
        home: best_epoch_from_csv(CV_ROOT / f"candidate_v2_50ep_val_{home}" / "results.csv")
        for home in FOLDS
    }
    return {"manifest": manifest, "best_epochs": best_epochs, "config": cv_summary}


def render_report(data: dict) -> str:
    gpu = data.get("gpu", {})
    lines = [
        "# Robot@Home2 最终单目模型：candidate v2 全开发集重训",
        "",
        f"- 训练状态：{data['status']}",
        f"- 开始/结束时间：{data['started_at']} / {data.get('finished_at', '进行中')}",
        f"- 训练轮数：{data['epochs']}（四折最佳 epoch 的中位数；selection metric=`{METRIC_COLUMN}`）",
        f"- 四折最佳 epoch：`{data['cv_best_epochs']}`；最终权重：`{data.get('best_weights', '尚未生成')}`",
        f"- 开发集：{data['unique_development_observations']} 张去重 RGB 图像；住宅=`alma/anto/pare/rx2`；类别=`bottle/chair/table`。",
        f"- GPU：{gpu.get('name', '尚未记录')}；CUDA available={gpu.get('cuda_available', '尚未记录')}。",
        "- 数据边界：只使用四个开发住宅的 v2 train/val 并集；按 observation ID 去重；`sarmis` 未进入训练。",
        "- 指标解释：Ultralytics 需要配置 val 路径，本数据配置将 val 指向 train；`val=False` 关闭逐轮验证，但框架仍会在结束时做最终记账验证。因此任何该处指标都是训练集内拟合诊断，不是泛化成绩；本报告不以它选择 checkpoint。",
        "- 训练轮数只由开发集四折结果预先确定；sarmis 仅留给下一步的一次性最终评估，不据其结果回调当前模型。",
        "",
    ]
    if data.get("diagnostic_in_sample_metrics"):
        lines.extend([
            "## 训练集内诊断指标（不可当作验证/测试成绩）",
            "",
            "| Precision | Recall | mAP50 | mAP50-95 |",
            "|---:|---:|---:|---:|",
            "| " + " | ".join(f"{data['diagnostic_in_sample_metrics'][name]:.4f}" for name in ("precision", "recall", "mAP50", "mAP50_95")) + " |",
            "",
        ])
    if data.get("error"):
        lines.extend(["## 错误", "", f"`{data['error']}`", ""])
    return "\n".join(lines)


def persist(data: dict) -> None:
    FINAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    FINAL_SUMMARY.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    FINAL_SUMMARY.with_suffix(".md").write_text(render_report(data), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=None, help="Optional explicit override; default is median CV best epoch")
    parser.add_argument("--preflight-only", action="store_true", help="Validate inputs/config without starting training")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    validated = validate_training_inputs(data_root)
    cv_best_epochs = validated["best_epochs"]
    derived_epochs = int(statistics.median(cv_best_epochs.values()))
    epochs = args.epochs or derived_epochs
    run_name = f"final_v2_{epochs}ep"
    run_dir = FINAL_RUNS / run_name
    if not FINAL_MODEL.is_file():
        raise FileNotFoundError(f"Pretrained model missing: {FINAL_MODEL}")
    if run_dir.exists() or FINAL_SUMMARY.exists() or FINAL_SUMMARY.with_suffix(".md").exists():
        raise FileExistsError(f"Final run/report already exists; refusing overwrite: {run_dir} / {FINAL_SUMMARY}")
    base = {
        "status": "preflight_passed",
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "finished_at": None,
        "model": str(FINAL_MODEL),
        "model_sha256": sha256(FINAL_MODEL),
        "data_root": str(data_root),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "run_dir": str(run_dir),
        "best_weights": None,
        "cv_best_epochs": cv_best_epochs,
        "selection_metric": METRIC_COLUMN,
        "derived_epochs": derived_epochs,
        "epochs": epochs,
        "epochs_override": args.epochs,
        "unique_development_observations": len(validated["manifest"]["items"]),
        "classes": list(CLASSES),
        "imgsz": 640,
        "batch": 8,
        "patience": 100,
        "device": "0",
        "workers": 0,
        "seed": 42,
        "deterministic": True,
        "validation_mode": "val=False; final framework bookkeeping uses train data as val, metrics are in-sample diagnostic only",
        "test_used": False,
        "diagnostic_in_sample_metrics": None,
    }
    if args.preflight_only:
        print(json.dumps(base, indent=2, ensure_ascii=False))
        return

    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "Ultralytics")
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to train this final GPU run on CPU")
    base["gpu"] = {
        "cuda_available": True,
        "name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    base["status"] = "training"
    persist(base)
    started = time.perf_counter()
    try:
        model = YOLO(str(FINAL_MODEL))
        model.train(
            data=str(data_root / "data.yaml"),
            epochs=epochs,
            patience=100,
            imgsz=640,
            batch=8,
            device="0",
            workers=0,
            seed=42,
            deterministic=True,
            project=str(FINAL_RUNS),
            name=run_name,
            split="val",
            val=False,
            plots=False,
            cache=False,
            save=True,
            exist_ok=False,
        )
        save_dir = Path(model.trainer.save_dir).resolve()
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"
        results_csv = save_dir / "results.csv"
        if not best_weights.is_file() or not last_weights.is_file() or not results_csv.is_file():
            raise FileNotFoundError(f"Training did not produce required outputs under {save_dir}")
        import torch as torch_module
        checkpoint = torch_module.load(best_weights, map_location="cpu", weights_only=False)
        train_metrics = checkpoint.get("train_metrics") or {}
        metric_map = {
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
            "mAP50": "metrics/mAP50(B)",
            "mAP50_95": "metrics/mAP50-95(B)",
        }
        if all(key in train_metrics for key in metric_map.values()):
            base["diagnostic_in_sample_metrics"] = {name: float(train_metrics[key]) for name, key in metric_map.items()}
        base.update({
            "status": "complete",
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "run_dir": str(save_dir),
            "best_weights": str(best_weights),
            "best_weights_sha256": sha256(best_weights),
            "last_weights": str(last_weights),
            "last_weights_sha256": sha256(last_weights),
            "results_csv": str(results_csv),
            "results_csv_sha256": sha256(results_csv),
            "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
            "test_used": False,
        })
        del checkpoint
        persist(base)
        print(render_report(base))
    except Exception as exc:
        base.update({
            "status": "failed",
            "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "test_used": False,
        })
        persist(base)
        raise


if __name__ == "__main__":
    main()
