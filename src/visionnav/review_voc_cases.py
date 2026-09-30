"""Create side-by-side ground-truth and prediction views for VOC val cases."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

import cv2
import torch
from ultralytics import YOLO


DEFAULT_MODEL = ROOT / "runs" / "detect" / "runs" / "voc_yolo26n_baseline" / "weights" / "best.pt"
DEFAULT_IMAGES = ROOT / "data" / "datasets" / "VOC" / "images" / "val2007"
DEFAULT_LABELS = ROOT / "data" / "datasets" / "VOC" / "labels" / "val2007"
DEFAULT_OUTPUT = ROOT / "outputs" / "voc_yolo26n_case_review"
DEFAULT_CASES = ("007624", "006628", "002965")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_ground_truth(label_path: Path, width: int, height: int) -> tuple[list, Counter]:
    boxes = []
    counts: Counter = Counter()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line}")
        class_id = int(parts[0])
        xc, yc, box_width, box_height = map(float, parts[1:])
        x1 = round((xc - box_width / 2) * width)
        y1 = round((yc - box_height / 2) * height)
        x2 = round((xc + box_width / 2) * width)
        y2 = round((yc + box_height / 2) * height)
        boxes.append((class_id, x1, y1, x2, y2))
        counts[class_id] += 1
    return boxes, counts


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_boxes(ground_truth: list, predictions: list, iou_threshold: float) -> tuple[list, list, list]:
    """Greedily match same-class boxes by confidence, with one GT per prediction."""
    unmatched_gt = set(range(len(ground_truth)))
    matched_predictions = set()
    matches = []
    for prediction_index in sorted(range(len(predictions)), key=lambda i: predictions[i][2], reverse=True):
        class_id, predicted_box, _ = predictions[prediction_index]
        candidates = [i for i in unmatched_gt if ground_truth[i][0] == class_id]
        if not candidates:
            continue
        gt_index = max(candidates, key=lambda i: box_iou(predicted_box, ground_truth[i][1:]))
        overlap = box_iou(predicted_box, ground_truth[gt_index][1:])
        if overlap >= iou_threshold:
            matches.append((gt_index, prediction_index, overlap))
            unmatched_gt.remove(gt_index)
            matched_predictions.add(prediction_index)
    unmatched_predictions = set(range(len(predictions))) - matched_predictions
    return matches, sorted(unmatched_gt), sorted(unmatched_predictions)


def draw_ground_truth(image, boxes: list, names: dict) -> object:
    canvas = image.copy()
    for class_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 190, 40), 2)
        cv2.putText(canvas, names[class_id], (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 130, 30), 1, cv2.LINE_AA)
    return canvas


def draw_predictions(image, result, names: dict) -> object:
    canvas = image.copy()
    if result.boxes is None:
        return canvas
    for box, class_id, confidence in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.cls.int().cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
    ):
        x1, y1, x2, y2 = map(round, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 230), 2)
        label = f"{names[class_id]} {confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 180), 1, cv2.LINE_AA)
    return canvas


def draw_error_map(image, ground_truth: list, predictions: list, matches: list,
                   missed_gt: list, false_predictions: list, names: dict,
                   focus_id: int | None = None) -> object:
    """Overlay matched and unmatched boxes with distinct colors for review."""
    canvas = image.copy()
    for index, (class_id, x1, y1, x2, y2) in enumerate(ground_truth):
        if focus_id is not None and class_id != focus_id:
            continue
        if index in missed_gt:
            color, label = (0, 220, 255), f"FN {names[class_id]}"
        else:
            color, label = (40, 190, 40), f"GT {names[class_id]}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if index in missed_gt:
            cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for index, (class_id, box, confidence) in enumerate(predictions):
        if focus_id is not None and class_id != focus_id:
            continue
        x1, y1, x2, y2 = map(round, box)
        if index in false_predictions:
            color, prefix = (20, 20, 230), "FP"
        else:
            color, prefix = (230, 180, 20), "TP"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if index in false_predictions:
            label = f"{prefix} {names[class_id]} {confidence:.2f}"
            cv2.putText(canvas, label, (x1, min(canvas.shape[0] - 4, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    canvas = cv2.copyMakeBorder(canvas, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(canvas, "GT match: green | TP prediction: cyan | FN: yellow | FP: red", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")

    model = YOLO(str(args.model))
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    args.output.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0) if str(args.device) != "cpu" else "CPU"
    report = [
        "# VOC07 val 样例框对照复核",
        "",
        f"- 权重：`{args.model}`",
        f"- 设备：`{args.device}`（{device_name}）",
        f"- 置信度阈值：{args.confidence}",
        f"- 匹配规则：同类别且 IoU ≥ {args.iou_threshold}；一个 GT 最多匹配一个预测框。",
        "- 绿色框：数据集人工标注（GT）；红色框：模型预测。",
        "- TP=成功匹配，FP=没有匹配到 GT 的预测框，FN=没有匹配到预测框的 GT。类别错误通常同时产生一个 FP 和一个 FN。",
        "",
    ]
    for stem in args.cases:
        image_path = args.images / f"{stem}.jpg"
        label_path = args.labels / f"{stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or label for case {stem}: {image_path}, {label_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        gt_boxes, gt_counts = read_ground_truth(label_path, width, height)
        result = model.predict(source=str(image_path), device=args.device, imgsz=args.image_size, conf=args.confidence, verbose=False)[0]
        pred_counts: Counter = Counter()
        prediction_boxes = []
        if result.boxes is not None:
            for box, class_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.int().cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                pred_counts[class_id] += 1
                prediction_boxes.append((class_id, tuple(box), confidence))
        matches, missed_gt, false_predictions = match_boxes(gt_boxes, prediction_boxes, args.iou_threshold)
        focus = {"007624": "pottedplant", "006628": "chair", "002965": "bottle"}.get(stem)
        target_id = next((class_id for class_id, class_name in names.items() if class_name == focus), None)
        error_view = draw_error_map(image, gt_boxes, prediction_boxes, matches, missed_gt, false_predictions, names, target_id)
        error_path = args.output / f"{stem}_errors.jpg"
        if not cv2.imwrite(str(error_path), error_view):
            raise OSError(f"Could not write error map: {error_path}")
        gt_view = draw_ground_truth(image, gt_boxes, names)
        pred_view = draw_predictions(image, result, names)
        panel = cv2.copyMakeBorder(gt_view, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        pred_panel = cv2.copyMakeBorder(pred_view, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(panel, "Ground truth (green)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 100, 30), 2, cv2.LINE_AA)
        cv2.putText(pred_panel, f"Prediction (conf >= {args.confidence:g})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 180), 2, cv2.LINE_AA)
        spacer = 16
        combined = cv2.hconcat([panel, cv2.copyMakeBorder(pred_panel, 0, 0, spacer, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))])
        output_image = args.output / f"{stem}_comparison.jpg"
        if not cv2.imwrite(str(output_image), combined):
            raise OSError(f"Could not write comparison image: {output_image}")

        image_link = f"../{args.output.relative_to(ROOT).as_posix()}/{stem}_comparison.jpg"
        error_link = f"../{args.output.relative_to(ROOT).as_posix()}/{stem}_errors.jpg"
        report.extend([f"## {stem}", "", f"![{stem} 标注框与预测框对照]({image_link})", "", f"[打开 {stem} 错误标记图]({error_link})", "", "| 重点类别 | GT | 预测 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|"])
        for class_id, class_name in names.items():
            if focus is None or class_name == focus:
                true_positives = sum(1 for gt_index, _, _ in matches if gt_boxes[gt_index][0] == class_id)
                false_positives = sum(1 for i in false_predictions if prediction_boxes[i][0] == class_id)
                false_negatives = sum(1 for i in missed_gt if gt_boxes[i][0] == class_id)
                report.append(f"| {class_name} | {gt_counts[class_id]} | {pred_counts[class_id]} | {true_positives} | {false_positives} | {false_negatives} |")
        report.extend(["", "重点类别未匹配项："])
        if target_id is None:
            report.append("- 本图没有预设重点类别；请结合并排图查看全部类别。")
        else:
            target_fp = [i for i in false_predictions if prediction_boxes[i][0] == target_id]
            target_fn = [i for i in missed_gt if gt_boxes[i][0] == target_id]
            if not target_fp and not target_fn:
                report.append("- 该类在当前阈值和 IoU 规则下没有未匹配项。")
            for i in target_fp:
                class_id, predicted_box, confidence = prediction_boxes[i]
                same_class_ious = [box_iou(predicted_box, box[1:]) for box in gt_boxes if box[0] == class_id]
                closest = max(same_class_ious, default=0.0)
                report.append(f"- FP 预测框：置信度 {confidence:.2f}，最接近同类 GT 的 IoU={closest:.2f}，坐标={tuple(round(v) for v in predicted_box)}。")
            for i in target_fn:
                class_id, x1, y1, x2, y2 = gt_boxes[i]
                same_class_ious = [box_iou(prediction[1], (x1, y1, x2, y2)) for prediction in prediction_boxes if prediction[0] == class_id]
                closest = max(same_class_ious, default=0.0)
                report.append(f"- FN 标注框：最接近同类预测的 IoU={closest:.2f}，坐标={(x1, y1, x2, y2)}。")
        report.append("")
        focus_id = target_id
        focus_tp = sum(1 for gt_index, _, _ in matches if gt_boxes[gt_index][0] == focus_id)
        focus_fp = sum(1 for i in false_predictions if prediction_boxes[i][0] == focus_id)
        focus_fn = sum(1 for i in missed_gt if gt_boxes[i][0] == focus_id)
        print(f"{stem} ({focus}): TP={focus_tp}, FP={focus_fp}, FN={focus_fn}; saved {output_image}")

    report.extend([
        "## 如何读这组图",
        "",
        "这里的 TP/FP/FN 是指定置信度阈值和 IoU 阈值下的逐框匹配结果，不等同于整套数据集的 AP/mAP。漏检是 FN；未匹配预测是 FP；同类别的框若 IoU 低于阈值，会按一个 FP 和一个 FN 处理，报告中的最近 IoU 可帮助识别是否属于定位偏差。",
        "错误标记图颜色：绿色=已匹配 GT，青色=已匹配预测，黄色=漏检 GT（FN），红色=未匹配预测（FP）。重叠框可能互相遮挡，需同时参考并排原图。",
        "遮挡、远小目标、密集相似物体可能增加判断难度，但单张图只能提供线索，不能直接证明错误原因。",
        "",
        "本复核仅抽查 VOC07 val，不用于训练或调参；VOC07 test 不在本阶段读取。",
        "",
    ])
    report_path = ROOT / "reports" / "voc_yolo26n_case_review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
