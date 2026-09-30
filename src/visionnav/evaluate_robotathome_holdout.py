"""Run exactly one final evaluation on the pre-reserved Robot@Home2 sarmis split."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import yaml

from visionnav.prepare_robotathome_final import CLASSES, DEFAULT_OUTPUT, ROOT, sha256, validate_labels
from visionnav.train_robotathome_final import FINAL_SUMMARY


PILOT_ROOT = ROOT / "data" / "datasets" / "RobotAtHome2" / "processed" / "yolo_pilot"
EVAL_RECORD = ROOT / "reports" / "robotathome2_sarmis_final_v2.json"
EVAL_PROJECT = ROOT / "runs" / "robotathome2_sarmis_final_eval"
EVAL_NAME = "v2_final_once"


def validate_holdout() -> dict:
    train_summary = json.loads(FINAL_SUMMARY.read_text(encoding="utf-8"))
    if train_summary.get("status") != "complete" or train_summary.get("test_used") is not False:
        raise ValueError("A complete final model trained without test use is required")
    weights = Path(train_summary["best_weights"]).resolve()
    if not weights.is_file() or sha256(weights) != train_summary.get("best_weights_sha256"):
        raise ValueError("Final checkpoint is missing or its SHA-256 differs from the frozen training report")
    if not weights.is_relative_to(ROOT):
        raise ValueError("Final checkpoint must remain within the project root")

    dev_manifest_path = DEFAULT_OUTPUT / "manifest.json"
    dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
    dev_ids = {str(item["observation_id"]) for item in dev_manifest.get("items", [])}
    if dev_manifest.get("sarmis_included") is not False or any(item.get("home") == "sarmis" for item in dev_manifest.get("items", [])):
        raise ValueError("Development set contains the reserved test home")

    pilot_manifest_path = PILOT_ROOT / "manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    test = pilot_manifest.get("splits", {}).get("test", {})
    test_items = test.get("items", [])
    test_ids = {str(item.get("observation_id", "")) for item in test_items}
    if pilot_manifest.get("classes") != list(CLASSES) or test.get("homes") != ["sarmis"]:
        raise ValueError("Reserved holdout must be the original sarmis test split with the frozen class mapping")
    if not test_items or len(test_ids) != len(test_items) or test_ids & dev_ids:
        raise ValueError("Holdout is empty, has duplicate IDs, or overlaps final development observations")

    data_path = PILOT_ROOT / "data.yaml"
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("names") != list(CLASSES):
        raise ValueError("Original Robot@Home2 data.yaml has a different class order")
    if Path(config.get("path", "")).resolve() != PILOT_ROOT.resolve() or config.get("test") != "images/test":
        raise ValueError("Original data.yaml test path is missing or escapes the reserved dataset")

    total_boxes = 0
    class_counts = {name: 0 for name in CLASSES}
    for item in test_items:
        observation_id = str(item["observation_id"])
        if item.get("home") != "sarmis":
            raise ValueError(f"Non-sarmis record in the reserved test split: {observation_id}")
        image_path = PILOT_ROOT / "images" / "test" / f"{observation_id}.png"
        label_path = PILOT_ROOT / "labels" / "test" / f"{observation_id}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing reserved holdout pair for observation {observation_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unreadable holdout RGB image: {image_path}")
        boxes, counts = validate_labels(label_path)
        source_counts = item.get("instances_by_class", {})
        expected_counts = {
            name: int(source_counts[name])
            for name in CLASSES
            if int(source_counts.get(name, 0)) > 0
        }
        if boxes != item.get("boxes") or counts != expected_counts:
            raise ValueError(f"Holdout labels disagree with the original manifest for {observation_id}")
        total_boxes += boxes
        for class_name, count in counts.items():
            class_counts[class_name] += count
    test_source_counts = test.get("instances_by_class", {})
    expected_split_counts = {
        name: int(test_source_counts[name])
        for name in CLASSES
        if int(test_source_counts.get(name, 0)) > 0
    }
    if len(test_items) != test.get("image_count") or class_counts != expected_split_counts:
        raise ValueError("Holdout image/class counts disagree with the preserved pilot manifest")
    if EVAL_RECORD.exists() or (EVAL_PROJECT / EVAL_NAME).exists():
        raise FileExistsError("The one-time sarmis evaluation has already been attempted or its output exists; refusing to rerun")
    if not EVAL_PROJECT.is_relative_to(ROOT):
        raise ValueError("Evaluation output must remain inside project root")

    return {
        "weights": str(weights),
        "weights_sha256": train_summary["best_weights_sha256"],
        "train_summary_sha256": sha256(FINAL_SUMMARY),
        "dev_manifest_sha256": sha256(dev_manifest_path),
        "test_manifest_sha256": sha256(pilot_manifest_path),
        "test_data_yaml_sha256": sha256(data_path),
        "test_data_yaml": str(data_path),
        "test_image_count": len(test_items),
        "test_box_count": total_boxes,
        "test_instances_by_class": class_counts,
        "test_home": "sarmis",
        "test_observation_ids": sorted(map(int, test_ids)),
        "dev_test_id_overlap": 0,
        "classes": list(CLASSES),
        "eval_run_dir": str(EVAL_PROJECT / EVAL_NAME),
    }


def render_report(data: dict) -> str:
    lines = [
        "# Robot@Home2 sarmis 最终留出评估（一次性）",
        "",
        f"- 状态：{data['status']}",
        f"- 评估时间：{data['started_at']} — {data.get('finished_at', '进行中')}",
        f"- 固定权重：`{data['weights']}`（SHA-256 `{data['weights_sha256']}`）",
        f"- 测试数据：住宅 `{data['test_home']}`，{data['test_image_count']} 张图、{data['test_box_count']} 个标注框；类别实例数 `{data['test_instances_by_class']}`。",
        f"- 与开发数据观测 ID 重叠：{data['dev_test_id_overlap']}。测试集来源清单：`{PILOT_ROOT / 'manifest.json'}`。",
        "- 此测试集仅评估一次。本次结果用于报告，不再用于调参或回头选择当前版本；当前基准只代表 Robot@Home2 保留住宅的数据分布，不代表真实机器人安全或用户相机域泛化。",
        "",
    ]
    if data.get("overall_metrics"):
        metrics = data["overall_metrics"]
        lines.extend([
            "## 总体指标",
            "",
            "| Precision | Recall | mAP50 | mAP50-95 |",
            "|---:|---:|---:|---:|",
            f"| {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['mAP50']:.4f} | {metrics['mAP50_95']:.4f} |",
            "",
            "## 类别指标",
            "",
            "| 类别 | Precision | Recall | AP50 | AP50-95 |",
            "|---|---:|---:|---:|---:|",
        ])
        for name in CLASSES:
            row = data["per_class_metrics"][name]
            lines.append(f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['AP50']:.4f} | {row['AP50_95']:.4f} |")
        lines.extend(["", f"- 推理耗时(ms/image)：`{data.get('inference_speed_ms_per_image', {})}`。", f"- 可视化/运行目录：`{data['eval_run_dir']}`。", ""])
    if data.get("error"):
        lines.extend(["## 评估失败", "", f"`{data['error']}`", "", "由于一次性测试锁已写入，脚本不会自动重跑；先核实评估是否已读取/输出任何测试指标，再决定是否需要新的独立测试集。", ""])
    return "\n".join(lines)


def persist(data: dict) -> None:
    EVAL_RECORD.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RECORD.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    EVAL_RECORD.with_suffix(".md").write_text(render_report(data), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate hashes, labels and split isolation without consuming the holdout")
    args = parser.parse_args()
    preflight = validate_holdout()
    if args.check_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        print("CHECK ONLY: sarmis metrics were not evaluated; the one-time test is still unused.")
        return

    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "Ultralytics")
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing the one-time final evaluation")
    model = YOLO(preflight["weights"])
    names = model.names
    ordered_names = [names[index] for index in range(len(names))] if isinstance(names, dict) else list(names)
    if ordered_names != list(CLASSES):
        raise ValueError(f"Frozen checkpoint class order mismatch: {ordered_names}")

    # This file is the irreversible-use guard: it is created immediately before the only call that evaluates sarmis.
    record = {
        **preflight,
        "status": "in_progress",
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "finished_at": None,
        "test_used": True,
        "device": torch.cuda.get_device_name(0),
        "imgsz": 640,
        "batch": 8,
        "confidence_threshold": 0.001,
        "nms_iou_threshold": 0.7,
        "split": "test",
    }
    persist(record)
    try:
        metrics = model.val(
            data=preflight["test_data_yaml"],
            split="test",
            imgsz=640,
            batch=8,
            device="0",
            workers=0,
            seed=42,
            deterministic=True,
            plots=True,
            conf=0.001,
            iou=0.7,
            project=str(EVAL_PROJECT),
            name=EVAL_NAME,
            exist_ok=False,
            verbose=True,
        )
        results = metrics.results_dict
        keys = {
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
            "mAP50": "metrics/mAP50(B)",
            "mAP50_95": "metrics/mAP50-95(B)",
        }
        record["overall_metrics"] = {name: float(results[key]) for name, key in keys.items()}
        class_to_result_index = {int(class_id): index for index, class_id in enumerate(metrics.box.ap_class_index)}
        per_class = {}
        for class_id, name in enumerate(CLASSES):
            if class_id not in class_to_result_index:
                per_class[name] = {"precision": 0.0, "recall": 0.0, "AP50": 0.0, "AP50_95": 0.0, "has_ground_truth": False}
                continue
            precision, recall, ap50, ap = metrics.box.class_result(class_to_result_index[class_id])
            per_class[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "AP50": float(ap50),
                "AP50_95": float(ap),
                "has_ground_truth": True,
            }
        record["per_class_metrics"] = per_class
        record["inference_speed_ms_per_image"] = {key: float(value) for key, value in metrics.speed.items()}
        record["eval_run_dir"] = str(Path(metrics.save_dir).resolve())
        record["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        record["status"] = "complete"
        persist(record)
        print(render_report(record))
    except Exception as exc:
        record["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        record["status"] = "failed_after_holdout_access"
        record["error"] = f"{type(exc).__name__}: {exc}"
        persist(record)
        raise


if __name__ == "__main__":
    main()
