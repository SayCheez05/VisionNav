"""Audit class-wise validation metrics and representative detection errors."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from visionnav.train_robotathome_cv import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT,
    FOLDS,
    ROOT,
    validate_fold_config,
    validate_fold_manifest,
)


DEFAULT_WEIGHTS_ROOT = DEFAULT_PROJECT
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "robotathome2_cv_error_audit"
DEFAULT_REPORT = ROOT / "reports" / "robotathome2_cv_error_audit.md"
DEFAULT_GALLERY_ROOT = ROOT / "outputs" / "robotathome2_cv_error_audit"
CLASSES = ("bottle", "chair", "table")
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
CANDIDATE_MAX_DET = 300
CONFIDENCE_THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25)
SIZE_BUCKETS = ("<2 px", "2–<8 px", "8–<16 px", "16–<24 px", "≥24 px")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--run-prefix", default="smoke_5ep",
                        help="Prefix used by the training run directories before the fold name.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prediction-conf", type=float, default=0.25,
                        help="Fixed confidence used only for qualitative error categorization and overlays.")
    parser.add_argument("--prediction-max-det", type=int, default=300,
                        help="Maximum retained predictions per image for the diagnostic matching pass.")
    parser.add_argument("--example-count", type=int, default=3)
    return parser.parse_args()


def yolo_label_boxes(label_path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        class_id = int(fields[0])
        xc, yc, box_width, box_height = map(float, fields[1:])
        if class_id not in range(len(CLASSES)) or not all(math.isfinite(x) for x in (xc, yc, box_width, box_height)):
            raise ValueError(f"Invalid class/coordinate at {label_path}:{line_number}")
        boxes.append({
            "class_id": class_id,
            "box": [
                (xc - box_width / 2) * width,
                (yc - box_height / 2) * height,
                (xc + box_width / 2) * width,
                (yc + box_height / 2) * height,
            ],
        })
    return boxes


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(
    ground_truth: list[dict],
    predictions: list[dict],
    available_gt: set[int],
    available_predictions: set[int],
    condition,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_index in available_gt:
        for prediction_index in available_predictions:
            overlap = box_iou(ground_truth[gt_index]["box"], predictions[prediction_index]["box"])
            if condition(ground_truth[gt_index], predictions[prediction_index], overlap):
                candidates.append((overlap, gt_index, prediction_index))
    candidates.sort(reverse=True)
    pairs = []
    for overlap, gt_index, prediction_index in candidates:
        if gt_index in available_gt and prediction_index in available_predictions:
            available_gt.remove(gt_index)
            available_predictions.remove(prediction_index)
            pairs.append((gt_index, prediction_index, overlap))
    return pairs


def categorize_errors(ground_truth: list[dict], predictions: list[dict]) -> dict:
    available_gt = set(range(len(ground_truth)))
    available_predictions = set(range(len(predictions)))
    categories = Counter()
    class_confusions = Counter()
    pairs = []

    correct = greedy_match(
        ground_truth, predictions, available_gt, available_predictions,
        lambda gt, pred, overlap: gt["class_id"] == pred["class_id"] and overlap >= 0.5,
    )
    categories["tp"] += len(correct)
    pairs.extend(("tp", gt_index, pred_index, overlap) for gt_index, pred_index, overlap in correct)

    wrong_class = greedy_match(
        ground_truth, predictions, available_gt, available_predictions,
        lambda gt, pred, overlap: gt["class_id"] != pred["class_id"] and overlap >= 0.5,
    )
    categories["class_mismatch"] += len(wrong_class)
    for gt_index, prediction_index, overlap in wrong_class:
        gt_class = CLASSES[ground_truth[gt_index]["class_id"]]
        predicted_class = CLASSES[predictions[prediction_index]["class_id"]]
        class_confusions[f"{gt_class}->{predicted_class}"] += 1
        pairs.append(("class_mismatch", gt_index, prediction_index, overlap))

    localization = greedy_match(
        ground_truth, predictions, available_gt, available_predictions,
        lambda gt, pred, overlap: gt["class_id"] == pred["class_id"] and 0.1 <= overlap < 0.5,
    )
    categories["localization_miss"] += len(localization)
    pairs.extend(("localization_miss", gt_index, prediction_index, overlap)
                 for gt_index, prediction_index, overlap in localization)

    categories["fn"] = len(available_gt)
    categories["fp"] = len(available_predictions)
    return {
        "counts": {key: int(categories[key]) for key in
                   ("tp", "fp", "fn", "class_mismatch", "localization_miss")},
        "class_confusions": dict(sorted(class_confusions.items())),
        "pairs": pairs,
    }


def gt_size_outcomes(ground_truth: list[dict], pairs: list[tuple[str, int, int, float]]) -> Counter:
    """Count each GT exactly once by original-pixel short side and match outcome."""
    outcome_by_gt = {}
    for category, gt_index, _, _ in pairs:
        if gt_index in outcome_by_gt or category not in ("tp", "class_mismatch", "localization_miss"):
            raise ValueError(f"Duplicate or unexpected GT matching outcome: {gt_index}, {category}")
        outcome_by_gt[gt_index] = category
    counts = Counter()
    for gt_index, item in enumerate(ground_truth):
        x1, y1, x2, y2 = item["box"]
        short_side = min(x2 - x1, y2 - y1)
        if short_side <= 0 or not math.isfinite(short_side):
            raise ValueError(f"Invalid GT short side: {short_side}")
        if short_side < 2:
            bucket = SIZE_BUCKETS[0]
        elif short_side < 8:
            bucket = SIZE_BUCKETS[1]
        elif short_side < 16:
            bucket = SIZE_BUCKETS[2]
        elif short_side < 24:
            bucket = SIZE_BUCKETS[3]
        else:
            bucket = SIZE_BUCKETS[4]
        class_name = CLASSES[item["class_id"]]
        counts[(class_name, bucket, "gt")] += 1
        counts[(class_name, bucket, outcome_by_gt.get(gt_index, "fn"))] += 1
    return counts


def size_outcome_rows(counts: Counter) -> list[dict]:
    rows = []
    for class_name in CLASSES:
        for bucket in SIZE_BUCKETS:
            row = {
                "class_name": class_name,
                "size_bucket": bucket,
                **{key: int(counts[(class_name, bucket, key)]) for key in
                   ("gt", "tp", "class_mismatch", "localization_miss", "fn")},
            }
            if row["gt"] != sum(row[key] for key in ("tp", "class_mismatch", "localization_miss", "fn")):
                raise ValueError(f"GT matching outcomes do not partition {class_name} {bucket}")
            row["recall_at_conf"] = row["tp"] / row["gt"] if row["gt"] else None
            rows.append(row)
    return rows


def match_prediction_item(
    result_path: str, image_by_id: dict[int, Path], item_by_id: dict[int, dict], seen: set[int],
) -> tuple[int, dict]:
    """Associate a streamed prediction with its manifest row by verified image path."""
    path = Path(result_path).resolve()
    observation_id = int(path.stem)
    expected_path = image_by_id.get(observation_id)
    if expected_path is None or path != expected_path.resolve() or observation_id in seen:
        raise ValueError(f"Prediction path is missing, duplicated, or outside the val manifest: {path}")
    seen.add(observation_id)
    return observation_id, item_by_id[observation_id]


def validation_files(fold_root: Path, data_root: Path, val_items: list[dict]) -> tuple[dict[int, Path], Path, str]:
    """Resolve both shared v3 files and legacy per-fold val directories."""
    expected_ids = {int(item["observation_id"]) for item in val_items}
    if len(expected_ids) != len(val_items):
        raise ValueError(f"Duplicate validation observation IDs in {fold_root}")
    val_list = fold_root / "val.txt"
    if val_list.is_file():
        paths = [Path(line.strip()).resolve() for line in val_list.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if any(not path.is_relative_to(data_root) for path in paths):
            raise ValueError(f"Validation list escapes the dataset root: {val_list}")
        label_dir = data_root / "labels" / "all"
        source = str(val_list)
    else:
        image_dir = fold_root / "images" / "val"
        paths = list(image_dir.glob("*.png"))
        label_dir = fold_root / "labels" / "val"
        source = str(image_dir)
    image_by_id = {int(path.stem): path for path in paths}
    if len(image_by_id) != len(paths) or set(image_by_id) != expected_ids:
        raise ValueError(f"Validation image IDs do not match manifest: {fold_root}")
    for observation_id, image_path in image_by_id.items():
        if not image_path.is_file() or not (label_dir / f"{observation_id}.txt").is_file():
            raise FileNotFoundError(f"Validation image/label pair missing: {image_path}")
    return image_by_id, label_dir, source


def draw_box(image: np.ndarray, box: list[float], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = max(18, y1 - 5)
    cv2.putText(image, label, (max(0, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def make_panel(image: np.ndarray, boxes: list[dict], title: str, prediction: bool) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    offset_x = (PANEL_WIDTH - resized.shape[1]) // 2
    offset_y = (PANEL_HEIGHT - resized.shape[0]) // 2
    canvas[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
    sx, sy = resized.shape[1] / width, resized.shape[0] / height
    for item in boxes:
        original = item["box"]
        scaled_box = [
            original[0] * sx + offset_x, original[1] * sy + offset_y,
            original[2] * sx + offset_x, original[3] * sy + offset_y,
        ]
        if prediction:
            label = f"{CLASSES[item['class_id']]} {item['confidence']:.4f}"
            color = (30, 30, 235)
        else:
            label = CLASSES[item["class_id"]]
            color = (30, 210, 30)
        draw_box(canvas, scaled_box, label, color)
    header = np.full((34, PANEL_WIDTH, 3), 245, dtype=np.uint8)
    cv2.putText(header, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
    return cv2.vconcat([header, canvas])


def save_examples(
    examples: list[dict], image_by_id: dict[int, Path], label_dir: Path,
    prediction_by_id: dict[int, list[dict]], output_path: Path, fold_name: str,
    prediction_conf: float,
) -> None:
    rows = []
    for example in examples:
        observation_id = example["observation_id"]
        image_path = image_by_id[observation_id]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot read validation image: {image_path}")
        height, width = image.shape[:2]
        gt_boxes = yolo_label_boxes(label_dir / f"{observation_id}.txt", width, height)
        predictions = prediction_by_id[observation_id]
        errors = example["error_counts"]
        caption = (f"{fold_name} | obs {observation_id} | errors: FP {errors['fp']}, FN {errors['fn']}, "
                   f"class {errors['class_mismatch']}, loc {errors['localization_miss']} | conf>={prediction_conf:.4f}")
        caption_bar = np.full((38, PANEL_WIDTH * 2, 3), 245, dtype=np.uint8)
        cv2.putText(caption_bar, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        gt_panel = make_panel(image, gt_boxes, "Ground truth", prediction=False)
        pred_panel = make_panel(image, predictions, "Prediction", prediction=True)
        rows.append(cv2.vconcat([caption_bar, cv2.hconcat([gt_panel, pred_panel])]))
    if not rows:
        raise ValueError(f"No examples selected for {fold_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not save example sheet: {output_path}")


def class_metrics(validation_metrics, names: list[str], manifest: dict) -> list[dict]:
    box_metrics = validation_metrics.box
    positions = {int(class_id): index for index, class_id in enumerate(box_metrics.ap_class_index)}
    support = manifest["splits"]["val"].get("instances_by_class", {})
    rows = []
    for class_id, name in enumerate(names):
        if class_id in positions:
            precision, recall, ap50, ap5095 = box_metrics.class_result(positions[class_id])
        else:
            precision = recall = ap50 = ap5095 = 0.0
        rows.append({
            "class_name": name,
            "ground_truth_instances": int(support.get(name, 0)),
            "precision": float(np.nan_to_num(precision)),
            "recall": float(np.nan_to_num(recall)),
            "AP50": float(np.nan_to_num(ap50)),
            "AP50_95": float(np.nan_to_num(ap5095)),
        })
    return rows


def render_report(data: dict) -> str:
    total_images = sum(fold["val_images"] for fold in data["fold_results"])
    count_at_005 = sum(fold["confidence_sweep"]["counts_by_threshold"]["0.005"]
                       for fold in data["fold_results"])
    count_at_025 = sum(fold["confidence_sweep"]["counts_by_threshold"]["0.25"]
                       for fold in data["fold_results"])
    candidate_count = sum(
        fold["confidence_sweep"]["mean_candidates_per_image"] * fold["val_images"]
        for fold in data["fold_results"]
    )
    capped_images = sum(fold["confidence_sweep"]["images_at_candidate_cap"]
                        for fold in data["fold_results"])
    p95_by_fold = ", ".join(
        f"{fold['validation_home']}={fold['confidence_sweep']['p95_confidence']:.4f}"
        for fold in data["fold_results"]
    )
    interpretation = (
        f"四折验证的是不同住宅域，类别实例数也不均衡；各折指标应分别阅读，不能视为同一模型的四个独立最终测试集。"
        f"固定诊断阈值 conf={data['prediction_conf']:.3f}、最多 {data['prediction_max_det']} 框/图的互斥错误计数受阈值和候选上限影响，不是标准 benchmark。"
        f"在四折合计 {total_images} 张 val 图中，conf≥0.25 保留 {count_at_025} 个候选，conf≥0.005 保留 {count_at_005} 个；"
        f"候选收集均值为 {candidate_count / max(total_images, 1):.1f} 个/图，达到候选上限的图像数为 {capped_images}，各住宅候选分数 P95 为 {p95_by_fold}。"
        "请优先复核 GT/预测图中的系统性漏检、类别混淆和框偏移，再决定是否补数据或修标注；不可仅为提高分数调阈值，也不能据小样本结果宣称可部署或查看 sarmis。"
    )
    lines = [
        "# Robot@Home2 四折类别与错误审计",
        "",
        f"- 时间：{data['started_at']}",
        f"- 模型：`{data['weights_root']}`；RTX 推理设备：`{data['device']}`；图像尺寸：{data['imgsz']}。",
        f"- 仅在每折对应的验证集上评估 best.pt；Ultralytics AP/P/R 使用验证曲线（val conf=0.001，NMS IoU=0.7）。sarmis 没有读取。",
        f"- 下方人工判读类别使用固定 conf={data['prediction_conf']:.3f}、NMS IoU=0.70、max_det={data['prediction_max_det']}；这组 TP/FP/FN 分类是可解释的诊断计数，不等同于 Ultralytics AP 指标或最优阈值性能。",
        "",
        "## 按住宅与类别的验证表现",
        "",
        "P/R 为 Ultralytics 验证曲线上 F1 最佳操作点；AP 是跨置信度阈值积分，更适合作为类别排序指标。每类实例数来自对应 val manifest。",
        "",
        "| 留出住宅 | 类别 | GT 实例 | P | R | AP50 | AP50-95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for fold in data["fold_results"]:
        for row in fold["class_metrics"]:
            lines.append(
                f"| {fold['validation_home']} | {row['class_name']} | {row['ground_truth_instances']} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | {row['AP50']:.3f} | {row['AP50_95']:.3f} |"
            )
    lines.extend([
        "",
        "## 置信度分布检查",
        "",
        f"每张图先按 conf≥{data['candidate_confidence_floor']:.3f} 收集最多 {data['candidate_max_det']} 个候选框，再统计置信度分布；候选框数接近上限意味着结果被 max_det 截断。此表用来说明为什么常见置信度阈值下可能没有输出，不是性能指标。",
        "",
        "| 留出住宅 | val 图像 | 每图候选框均值 | 候选分数 P95 | ≥0.005 | ≥0.01 | ≥0.05 | ≥0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fold in data["fold_results"]:
        sweep = fold["confidence_sweep"]
        count = sweep["counts_by_threshold"]
        lines.append(
            f"| {fold['validation_home']} | {fold['val_images']} | {sweep['mean_candidates_per_image']:.1f} | "
            f"{sweep['p95_confidence']:.4f} | {count['0.005']} | {count['0.01']} | {count['0.05']} | {count['0.25']} |"
        )
    lines.extend([
        "",
        "## 固定阈值下的错误类别",
        "",
        "下表采用互斥错误归类：同类 IoU≥0.50 为 TP；不同类别 IoU≥0.50 记类别混淆；同类 0.10≤IoU<0.50 记定位偏差；剩余未匹配目标/预测分别为 FN/FP。它用于挑选复核样例，不是正式 benchmark。",
        "",
        "| 留出住宅 | val 图像 | GT 框 | 预测框 | TP | 未匹配 FN | 未匹配 FP | 类别混淆 | 定位偏差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    total = Counter()
    for fold in data["fold_results"]:
        errors = fold["thresholded_error_counts"]
        total.update(errors)
        lines.append(
            f"| {fold['validation_home']} | {fold['val_images']} | {fold['ground_truth_boxes']} | "
            f"{fold['prediction_boxes']} | {errors['tp']} | {errors['fn']} | {errors['fp']} | "
            f"{errors['class_mismatch']} | {errors['localization_miss']} |"
        )
    lines.append(
        f"| 合计（out-of-fold） | {sum(f['val_images'] for f in data['fold_results'])} | "
        f"{sum(f['ground_truth_boxes'] for f in data['fold_results'])} | "
        f"{sum(f['prediction_boxes'] for f in data['fold_results'])} | {total['tp']} | {total['fn']} | "
        f"{total['fp']} | {total['class_mismatch']} | {total['localization_miss']} |"
    )
    lines.extend([
        "",
        "## 瓶子按原始目标尺寸分层的诊断召回率",
        "",
        f"尺寸为验证原图框的较短边，单位像素。下表在固定 conf={data['prediction_conf']:.3f}、同类 IoU≥0.50 下统计 TP/GT；类别混淆、定位偏差和未匹配目标都算漏检。该数值不是跨阈值 AP，也不是 Ultralytics 报告中的最优 F1 点 Recall。",
        "",
        "| 留出住宅 | 短边尺寸 | GT | 正确检出 | 类别混淆 | 定位偏差 | 未匹配 | 诊断 Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    pooled_size = Counter()
    for fold in data["fold_results"]:
        for row in fold["size_stratified"]:
            if row["class_name"] != "bottle":
                continue
            for key in ("gt", "tp", "class_mismatch", "localization_miss", "fn"):
                pooled_size[(row["class_name"], row["size_bucket"], key)] += row[key]
            recall_text = f"{row['recall_at_conf']:.3f}" if row["recall_at_conf"] is not None else "—"
            lines.append(
                f"| {fold['validation_home']} | {row['size_bucket']} | {row['gt']} | {row['tp']} | "
                f"{row['class_mismatch']} | {row['localization_miss']} | {row['fn']} | {recall_text} |"
            )
    for row in size_outcome_rows(pooled_size):
        if row["class_name"] == "bottle":
            recall_text = f"{row['recall_at_conf']:.3f}" if row["recall_at_conf"] is not None else "—"
            lines.append(
                f"| 合并验证住宅 | {row['size_bucket']} | {row['gt']} | {row['tp']} | "
                f"{row['class_mismatch']} | {row['localization_miss']} | {row['fn']} | {recall_text} |"
            )
    lines.extend([
        "",
        "## 类别混淆方向",
        "",
        "以下只统计固定阈值和 max_det 下、不同类别预测与 GT 的 IoU≥0.50 匹配；箭头左侧为 GT 类别，右侧为模型预测类别。",
        "",
        "| 留出住宅 | GT 类别 → 预测类别 | 数量 |",
        "|---|---|---:|",
    ])
    for fold in data["fold_results"]:
        if fold["class_confusions"]:
            for direction, count in fold["class_confusions"].items():
                lines.append(f"| {fold['validation_home']} | {direction} | {count} |")
        else:
            lines.append(f"| {fold['validation_home']} | 无 | 0 |")
    lines.extend(["", "## GT / prediction 复核图", ""])
    for fold in data["fold_results"]:
        lines.append(
            f"- {fold['validation_home']}：挑选 {len(fold['examples'])} 张固定阈值错误较多的 val 样例，"
            f"左侧为 GT 绿色框，右侧为预测红色框：`{fold['example_sheet']}`。"
        )
        for example in fold["examples"]:
            err = example["error_counts"]
            lines.append(
                f"  - observation {example['observation_id']}（{example['physical_room']}）："
                f"FP={err['fp']}、FN={err['fn']}、类别混淆={err['class_mismatch']}、定位偏差={err['localization_miss']}。"
            )
    lines.extend([
        "",
        "## 解读边界与建议",
        "",
        interpretation,
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.imgsz < 32 or args.batch < 1 or args.workers < 0 or args.example_count < 1 or args.prediction_max_det < 1:
        raise ValueError("imgsz, batch and example-count must be positive; workers cannot be negative")
    if not 0.0 <= args.prediction_conf < 1.0:
        raise ValueError("prediction confidence must be in [0, 1)")
    if not args.run_prefix or any(character in args.run_prefix for character in "\\/:*?\"<>| "):
        raise ValueError("--run-prefix must be a non-empty path-safe token without spaces")

    data_root = args.data_root.resolve()
    weights_root = args.weights_root.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    gallery_root = args.gallery_root.resolve()
    for path in (data_root, weights_root, output_root, report_path, gallery_root):
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Dataset, weights, and outputs must stay within project root {ROOT}: {path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if report_path.exists() or report_path.with_suffix(".json").exists():
        raise FileExistsError(f"Report exists; refusing to overwrite: {report_path}")
    for fold in args.folds:
        if (output_root / fold).exists():
            raise FileExistsError(f"Audit output already exists; refusing to overwrite: {output_root / fold}")
        gallery_path = gallery_root / f"{fold}_gt_vs_pred.jpg"
        if gallery_path.exists():
            raise FileExistsError(f"Example sheet exists; refusing to overwrite: {gallery_path}")

    prepared = []
    for fold_name in args.folds:
        fold_root = data_root / fold_name
        manifest_path = fold_root / "manifest.json"
        data_yaml = fold_root / "data.yaml"
        weights = weights_root / f"{args.run_prefix}_{fold_name}" / "weights" / "best.pt"
        if not manifest_path.is_file() or not data_yaml.is_file() or not weights.is_file():
            raise FileNotFoundError(f"Fold inputs missing for {fold_name}: {manifest_path}, {data_yaml}, {weights}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_fold_manifest(fold_name, manifest)
        validate_fold_config(fold_root, data_yaml, manifest)
        if manifest.get("classes") != list(CLASSES):
            raise ValueError(f"Unexpected class set/order in {fold_name}: {manifest.get('classes')}")
        prepared.append((fold_name, fold_root, data_yaml, weights, manifest))

    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "Ultralytics")
    import torch
    from ultralytics import YOLO

    if str(args.device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    candidate_confidence_floor = min(0.001, args.prediction_conf)
    output = {
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "weights_root": str(weights_root),
        "run_prefix": args.run_prefix,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "gallery_root": str(gallery_root),
        "device": str(args.device),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "validation_conf": 0.001,
        "validation_nms_iou": 0.7,
        "candidate_confidence_floor": candidate_confidence_floor,
        "candidate_max_det": CANDIDATE_MAX_DET,
        "confidence_sweep_thresholds": list(CONFIDENCE_THRESHOLDS),
        "prediction_conf": args.prediction_conf,
        "prediction_iou": 0.7,
        "prediction_max_det": args.prediction_max_det,
        "test_used": False,
        "fold_results": [],
    }

    for fold_name, fold_root, data_yaml, weights, manifest in prepared:
        print(f"\n=== Auditing {fold_name}; val={manifest['validation_home']} ===")
        model = YOLO(str(weights))
        names_map = model.names
        names = [names_map[index] for index in range(len(names_map))]
        if names != list(CLASSES):
            raise ValueError(f"Checkpoint class order mismatch for {fold_name}: {names}")

        val_items = manifest["splits"]["val"]["items"]
        item_by_id = {int(item["observation_id"]): item for item in val_items}
        image_by_id, label_dir, prediction_source = validation_files(fold_root, data_root, val_items)

        validation_metrics = model.val(
            data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, conf=0.001, iou=0.7,
            plots=True, project=str(output_root), name=fold_name,
            exist_ok=False, verbose=True,
        )
        per_class = class_metrics(validation_metrics, names, manifest)
        prediction_results = model.predict(
            source=prediction_source, imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, conf=candidate_confidence_floor,
            iou=0.7, max_det=CANDIDATE_MAX_DET, stream=True, save=False, verbose=False,
        )

        predictions_by_id = {}
        seen_predictions = set()
        per_image = []
        error_totals = Counter()
        size_error_totals = Counter()
        prediction_total = 0
        ground_truth_total = 0
        class_confusions = Counter()
        confidence_scores = []
        confidence_counts = Counter()
        images_at_candidate_cap = 0
        for result in prediction_results:
            observation_id, item = match_prediction_item(
                result.path, image_by_id, item_by_id, seen_predictions,
            )
            image = cv2.imread(str(image_by_id[observation_id]), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Could not decode validation image: {image_by_id[observation_id]}")
            height, width = image.shape[:2]
            gt_boxes = yolo_label_boxes(label_dir / f"{observation_id}.txt", width, height)
            candidate_boxes = []
            if result.boxes is not None and len(result.boxes):
                xyxy = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                candidate_boxes = [
                    {"class_id": int(class_id), "box": box.astype(float).tolist(), "confidence": float(conf)}
                    for box, class_id, conf in zip(xyxy, classes, confidences)
                ]
                candidate_boxes.sort(key=lambda row: row["confidence"], reverse=True)
            confidence_scores.extend(box["confidence"] for box in candidate_boxes)
            if len(candidate_boxes) >= CANDIDATE_MAX_DET:
                images_at_candidate_cap += 1
            for threshold in CONFIDENCE_THRESHOLDS:
                confidence_counts[f"{threshold:g}"] += sum(
                    box["confidence"] >= threshold for box in candidate_boxes
                )
            predicted_boxes = [box for box in candidate_boxes if box["confidence"] >= args.prediction_conf]
            predicted_boxes = predicted_boxes[:args.prediction_max_det]
            predictions_by_id[observation_id] = predicted_boxes
            image_errors = categorize_errors(gt_boxes, predicted_boxes)
            error_totals.update(image_errors["counts"])
            size_error_totals.update(gt_size_outcomes(gt_boxes, image_errors["pairs"]))
            class_confusions.update(image_errors["class_confusions"])
            ground_truth_total += len(gt_boxes)
            prediction_total += len(predicted_boxes)
            image_error_total = sum(image_errors["counts"][key] for key in
                                    ("fp", "fn", "class_mismatch", "localization_miss"))
            per_image.append({
                "observation_id": observation_id,
                "physical_room": item["physical_room"],
                "relative_room": item["relative_room"],
                "ground_truth_boxes": len(gt_boxes),
                "prediction_boxes": len(predicted_boxes),
                "error_counts": image_errors["counts"],
                "class_confusions": image_errors["class_confusions"],
                "error_score": image_error_total,
            })

        if seen_predictions != set(item_by_id):
            raise RuntimeError(f"Prediction/image count or IDs mismatch for {fold_name}")

        ranked = sorted(per_image, key=lambda row: (-row["error_score"], row["observation_id"]))
        examples = []
        selected_groups = set()
        for example in ranked:
            if len(examples) >= args.example_count:
                break
            group = example["relative_room"]
            if example["error_score"] > 0 and group not in selected_groups:
                examples.append(example)
                selected_groups.add(group)
        if len(examples) < args.example_count:
            chosen_ids = {item["observation_id"] for item in examples}
            examples.extend(row for row in ranked if row["observation_id"] not in chosen_ids
                            and len(examples) < args.example_count)

        example_sheet = gallery_root / f"{fold_name}_gt_vs_pred.jpg"
        save_examples(examples, image_by_id, label_dir, predictions_by_id,
                      example_sheet, fold_name, args.prediction_conf)
        size_rows = size_outcome_rows(size_error_totals)
        if sum(row["gt"] for row in size_rows) != ground_truth_total or any(
            sum(row[key] for row in size_rows) != error_totals[key]
            for key in ("tp", "class_mismatch", "localization_miss", "fn")
        ):
            raise RuntimeError(f"Size-stratified outcomes disagree with fixed-threshold error totals: {fold_name}")
        fold_row = {
            "fold": fold_name,
            "validation_home": manifest["validation_home"],
            "train_homes": manifest["train_homes"],
            "val_images": len(val_items),
            "ground_truth_boxes": ground_truth_total,
            "prediction_boxes": prediction_total,
            "class_metrics": per_class,
            "confidence_sweep": {
                "mean_candidates_per_image": len(confidence_scores) / len(val_items),
                "p95_confidence": float(np.quantile(confidence_scores, 0.95)) if confidence_scores else 0.0,
                "images_at_candidate_cap": images_at_candidate_cap,
                "counts_by_threshold": {f"{threshold:g}": int(confidence_counts[f"{threshold:g}"])
                                         for threshold in CONFIDENCE_THRESHOLDS},
            },
            "thresholded_error_counts": {key: int(error_totals[key]) for key in
                                          ("tp", "fp", "fn", "class_mismatch", "localization_miss")},
            "size_stratified": size_rows,
            "class_confusions": dict(sorted(class_confusions.items())),
            "examples": examples,
            "example_sheet": str(example_sheet),
            "run_dir": str(output_root / fold_name),
            "best_weights": str(weights),
        }
        output["fold_results"].append(fold_row)
        print(f"{fold_name}: class metrics ready; errors={fold_row['thresholded_error_counts']}; sheet={example_sheet}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(output), encoding="utf-8")
    report_path.with_suffix(".json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Report: {report_path}")
    print(f"JSON: {report_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
