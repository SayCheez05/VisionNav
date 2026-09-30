"""Audit two completed Robot@Home2 LOHO runs and compare their val metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
FOLDS = ("val_alma", "val_anto", "val_pare", "val_rx2")
METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}
SAME_CONFIG = (
    "model", "data_root", "epochs", "patience", "batch", "device", "workers",
    "seed", "optimizer", "lr0", "warmup_bias_lr", "early_stopping_metric", "classes",
)
OVERALL = ("precision", "recall", "mAP50", "mAP50_95")
PER_CLASS = ("precision", "recall", "AP50", "AP50_95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports" / "robotathome2_cv_candidate_v3_640_map50.json")
    parser.add_argument("--candidate", type=Path, default=ROOT / "reports" / "robotathome2_cv_candidate_v3_960_map50.json")
    parser.add_argument("--output", type=Path, help="Write the Markdown report; omit to print only.")
    return parser.parse_args()


def number(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value}")
    return result


def audit_run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("completed_at"):
        raise ValueError(f"Four-fold run is not complete: {path}")
    if data.get("test_used") is not False:
        raise ValueError(f"Held-out test set was used or its usage is unknown: {path}")
    rows = data.get("fold_results", [])
    if len(rows) != 4 or {row["fold"] for row in rows} != set(FOLDS):
        raise ValueError(f"Expected exactly four distinct LOHO folds: {path}")
    if data.get("classes") != ["bottle", "chair", "table"]:
        raise ValueError(f"Unexpected class mapping: {path}")
    if data.get("early_stopping_metric") != "mAP50":
        raise ValueError(f"Unexpected checkpoint selection metric: {path}")

    import torch

    for row in rows:
        fold = row["fold"]
        run_dir = Path(row["run_dir"]).resolve()
        best_weights = Path(row["best_weights"]).resolve()
        if not run_dir.is_relative_to(ROOT / "runs") or best_weights != run_dir / "weights" / "best.pt":
            raise ValueError(f"Unexpected checkpoint location for {fold}: {best_weights}")
        results_path = run_dir / "results.csv"
        with results_path.open(encoding="utf-8-sig", newline="") as results_file:
            csv_rows = list(csv.DictReader(results_file))
        if not csv_rows:
            raise ValueError(f"Missing epoch results for {fold}: {results_path}")
        selected = [item for item in csv_rows if int(item["epoch"]) == row["best_epoch"]]
        if len(selected) != 1:
            raise ValueError(f"Best epoch not found exactly once for {fold}")
        csv_best = max(number(item[METRIC_COLUMNS["mAP50"]], fold) for item in csv_rows)
        selected_map50 = number(selected[0][METRIC_COLUMNS["mAP50"]], fold)
        if not math.isclose(selected_map50, csv_best, rel_tol=0, abs_tol=2e-5):
            raise ValueError(f"Selected epoch does not maximize mAP50 for {fold}")

        checkpoint = torch.load(best_weights, map_location="cpu", weights_only=False)
        saved = checkpoint.get("train_metrics") or {}
        for key, column in METRIC_COLUMNS.items():
            checkpoint_value = number(saved[column], f"{fold} checkpoint {key}")
            reported_value = number(row["checkpoint_metrics"][key], f"{fold} summary {key}")
            csv_value = number(selected[0][column], f"{fold} CSV {key}")
            if not math.isclose(checkpoint_value, reported_value, rel_tol=0, abs_tol=2e-5) or not math.isclose(
                checkpoint_value, csv_value, rel_tol=0, abs_tol=2e-5
            ):
                raise ValueError(f"Checkpoint, summary and CSV disagree for {fold} {key}")
            val_value = number(row["standalone_validation_metrics"][key], f"{fold} standalone {key}")
            if abs(val_value - checkpoint_value) > 0.005:
                raise ValueError(f"Independent validation differs from checkpoint for {fold} {key}")
        for class_name in data["classes"]:
            class_metrics = row["standalone_per_class_metrics"][class_name]
            for key in PER_CLASS:
                number(class_metrics[key], f"{fold} {class_name} {key}")
        if row["validation_home"] in row["train_homes"]:
            raise ValueError(f"Validation house appears in training houses for {fold}")

    for key in OVERALL:
        observed = sum(number(row["standalone_validation_metrics"][key], key) for row in rows) / 4
        if not math.isclose(observed, number(data["macro_average"][key], key), rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Macro average mismatch for {key}: {path}")
    for class_name in data["classes"]:
        for key in PER_CLASS:
            observed = sum(number(row["standalone_per_class_metrics"][class_name][key], key) for row in rows) / 4
            reported = number(data["per_class_macro_average"][class_name][key], key)
            if not math.isclose(observed, reported, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Per-class macro average mismatch for {class_name} {key}: {path}")
    return data


def compare(baseline: dict, candidate: dict) -> str:
    for key in SAME_CONFIG:
        if baseline[key] != candidate[key]:
            raise ValueError(f"Controlled comparison changed {key}: {baseline[key]!r} -> {candidate[key]!r}")
    baseline_folds = {row["fold"]: row for row in baseline["fold_results"]}
    candidate_folds = {row["fold"]: row for row in candidate["fold_results"]}
    for fold in FOLDS:
        before, after = baseline_folds[fold], candidate_folds[fold]
        for key in ("validation_home", "train_homes", "train_images", "val_images"):
            if before[key] != after[key]:
                raise ValueError(f"Controlled comparison changed {fold} {key}")

    lines = [
        "# Robot@Home2：输入分辨率四折 LOHO 对照",
        "",
        f"- 生成时间（Asia/Shanghai）：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}",
        f"- 基准：{baseline['run_prefix']}，imgsz={baseline['imgsz']}；候选：{candidate['run_prefix']}，imgsz={candidate['imgsz']}。",
        "- 两组均已完成四折；已核对住宅划分、关键配置、best.pt、CSV 最佳轮次、独立验证及等权宏平均。sarmis 未使用。",
        "- 差值为候选减基准；所有指标均来自各折 best.pt 的独立 val，不是训练中间轮次。",
        "",
        "| 留出住宅 | mAP50 基准→候选 | ΔmAP50 | Recall 基准→候选 | ΔRecall | bottle AP50 基准→候选 | Δbottle AP50 | bottle Recall 基准→候选 | Δbottle Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in FOLDS:
        before, after = baseline_folds[fold], candidate_folds[fold]
        old = before["standalone_validation_metrics"]
        new = after["standalone_validation_metrics"]
        old_bottle = before["standalone_per_class_metrics"]["bottle"]
        new_bottle = after["standalone_per_class_metrics"]["bottle"]
        lines.append(
            f"| {before['validation_home']} | {old['mAP50']:.4f}→{new['mAP50']:.4f} | {new['mAP50']-old['mAP50']:+.4f} | "
            f"{old['recall']:.4f}→{new['recall']:.4f} | {new['recall']-old['recall']:+.4f} | "
            f"{old_bottle['AP50']:.4f}→{new_bottle['AP50']:.4f} | {new_bottle['AP50']-old_bottle['AP50']:+.4f} | "
            f"{old_bottle['recall']:.4f}→{new_bottle['recall']:.4f} | {new_bottle['recall']-old_bottle['recall']:+.4f} |"
        )
    old = baseline["macro_average"]
    new = candidate["macro_average"]
    old_bottle = baseline["per_class_macro_average"]["bottle"]
    new_bottle = candidate["per_class_macro_average"]["bottle"]
    lines.append(
        f"| 四折等权平均 | {old['mAP50']:.4f}→{new['mAP50']:.4f} | {new['mAP50']-old['mAP50']:+.4f} | "
        f"{old['recall']:.4f}→{new['recall']:.4f} | {new['recall']-old['recall']:+.4f} | "
        f"{old_bottle['AP50']:.4f}→{new_bottle['AP50']:.4f} | {new_bottle['AP50']-old_bottle['AP50']:+.4f} | "
        f"{old_bottle['recall']:.4f}→{new_bottle['recall']:.4f} | {new_bottle['recall']-old_bottle['recall']:+.4f} |"
    )
    lines.extend(["", "| 总体宏平均指标 | 基准 | 候选 | 差值 |", "|---|---:|---:|---:|"])
    for key in OVERALL:
        lines.append(f"| {key} | {old[key]:.4f} | {new[key]:.4f} | {new[key]-old[key]:+.4f} |")
    lines.extend([
        "", "| 类别 | AP50 基准→候选 | ΔAP50 | AP50–95 基准→候选 | ΔAP50–95 | Recall 基准→候选 | ΔRecall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for class_name in baseline["classes"]:
        old_class = baseline["per_class_macro_average"][class_name]
        new_class = candidate["per_class_macro_average"][class_name]
        lines.append(
            f"| {class_name} | {old_class['AP50']:.4f}→{new_class['AP50']:.4f} | {new_class['AP50']-old_class['AP50']:+.4f} | "
            f"{old_class['AP50_95']:.4f}→{new_class['AP50_95']:.4f} | {new_class['AP50_95']-old_class['AP50_95']:+.4f} | "
            f"{old_class['recall']:.4f}→{new_class['recall']:.4f} | {new_class['recall']-old_class['recall']:+.4f} |"
        )
    lines.extend(["", "本对照只回答输入分辨率在当前数据和训练协议下的影响。只有四个住宅，单折退化不能被宏平均掩盖；后续还需检查误报、推理速度与瓶子按尺寸分层表现。", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = audit_run(args.baseline)
    candidate = audit_run(args.candidate)
    report = compare(baseline, candidate)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(report)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
