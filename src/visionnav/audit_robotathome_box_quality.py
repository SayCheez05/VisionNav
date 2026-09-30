"""Read-only audit of Robot@Home2 train/val boxes against their source masks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from visionnav.inspect_robotathome_masks import align_mask, decode_instance_mask, parse_mask_file
from visionnav.prepare_robotathome_yolo import (
    CLASSES,
    extract_yolo_boxes,
    load_instance_classes,
)
from visionnav.train_robotathome_cv import (
    DEFAULT_DATA_ROOT,
    ROOT,
    validate_fold_config,
    validate_fold_manifest,
)


FOLDS = ("val_alma", "val_anto", "val_pare", "val_rx2")
DEFAULT_RAW_ROOT = ROOT / "data" / "datasets" / "RobotAtHome2" / "raw"
DEFAULT_REPORT = ROOT / "reports" / "robotathome2_box_quality_audit.md"
DEFAULT_GALLERY = ROOT / "outputs" / "robotathome2_box_quality_extremes.jpg"
SMALL_SIDE_PX = 16.0
SMALL_AREA_FRACTION = 0.005
EXTREME_ASPECT_RATIO = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    return parser.parse_args()


def query_source_records(database: Path, observation_ids: set[int]) -> dict[int, dict]:
    """Read source file mappings for the manifest's development IDs only."""
    result: dict[int, dict] = {}
    uri = f"{database.resolve().as_uri()}?mode=ro"
    ordered_ids = sorted(observation_ids)
    with sqlite3.connect(uri, uri=True) as connection:
        for start in range(0, len(ordered_ids), 700):
            chunk = ordered_ids[start:start + 700]
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT DISTINCT o.id, o.home_session_id, h.name, m.new_path,
                                m.new_file_2, m.new_file_3, t.name
                FROM rh_lblrgbd AS o
                JOIN rh_homes AS h ON h.id = o.home_id
                JOIN rh2_old2new_rgbd_files AS m ON m.id = o.id
                JOIN rh_lblrgbd_labels AS l ON l.sensor_observation_id = o.id
                JOIN rh_object_types AS t ON t.id = l.object_type_id
                WHERE o.id IN ({placeholders}) AND t.name IN (?, ?, ?)
            """
            for observation_id, session_id, home, relative, image_name, mask_name, class_name in connection.execute(
                query, (*chunk, *CLASSES)
            ):
                record = result.setdefault(int(observation_id), {
                    "home_session_id": int(session_id),
                    "home": str(home),
                    "relative_room": str(relative),
                    "image_name": str(image_name),
                    "mask_name": str(mask_name),
                    "classes": set(),
                })
                metadata = (int(session_id), str(home), str(relative), str(image_name), str(mask_name))
                saved = (record["home_session_id"], record["home"], record["relative_room"],
                         record["image_name"], record["mask_name"])
                if metadata != saved:
                    raise ValueError(f"Conflicting source metadata for observation {observation_id}")
                record["classes"].add(str(class_name))
    missing = observation_ids - set(result)
    if missing:
        raise ValueError(f"Could not resolve {len(missing)} manifest observations in the source database")
    return result


def load_exported_boxes(label_path: Path, image_width: int, image_height: int) -> list[dict]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {label_path}:{line_number}")
        class_id = int(fields[0])
        xc, yc, width, height = map(float, fields[1:])
        if class_id not in range(len(CLASSES)) or not all(math.isfinite(x) for x in (xc, yc, width, height)):
            raise ValueError(f"Invalid class/coordinate at {label_path}:{line_number}")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"Out-of-range YOLO box at {label_path}:{line_number}")
        x1, y1 = (xc - width / 2) * image_width, (yc - height / 2) * image_height
        x2, y2 = (xc + width / 2) * image_width, (yc + height / 2) * image_height
        boxes.append({
            "class_id": class_id,
            "xc": xc,
            "yc": yc,
            "width": width,
            "height": height,
            "pixel_width": (x2 - x1),
            "pixel_height": (y2 - y1),
            "xyxy": [x1, y1, x2, y2],
        })
    return boxes


def validate_and_measure(
    data_root: Path,
    raw_root: Path,
    fold_manifests: dict[str, dict],
) -> tuple[list[dict], dict]:
    database = raw_root / "rh.db"
    rgb_root = raw_root / "files" / "rgbd"
    if not database.is_file() or not rgb_root.is_dir():
        raise FileNotFoundError(f"Expected Robot@Home2 raw data under {raw_root}")

    manifest_rows = {}
    for fold, manifest in fold_manifests.items():
        for split in ("train", "val"):
            for item in manifest["splits"][split]["items"]:
                observation_id = int(item["observation_id"])
                meta = {**item, "fold": fold, "split": split}
                old = manifest_rows.get((fold, split, observation_id))
                if old is not None and old != meta:
                    raise ValueError(f"Duplicate/conflicting manifest item: {fold}/{split}/{observation_id}")
                manifest_rows[(fold, split, observation_id)] = meta
    observation_ids = {key[2] for key in manifest_rows}
    sources = query_source_records(database, observation_ids)
    if any(row["home"].casefold() == "sarmis" for row in sources.values()):
        raise ValueError("A final-test home ID was unexpectedly present in the development manifests")
    instance_classes = load_instance_classes(database, sorted(observation_ids))

    measurements: list[dict] = []
    verified_observations = set()
    source_box_pixels: dict[int, list[int]] = {}
    split_summaries = {}
    for fold in FOLDS:
        fold_root = data_root / fold
        manifest = fold_manifests[fold]
        validate_fold_manifest(fold, manifest)
        validate_fold_config(fold_root, fold_root / "data.yaml", manifest)
        split_ids = {}
        for split in ("train", "val"):
            records = manifest["splits"][split]["items"]
            record_by_id = {int(item["observation_id"]): item for item in records}
            split_ids[split] = set(record_by_id)
            image_dir = fold_root / "images" / split
            label_dir = fold_root / "labels" / split
            image_paths = {int(path.stem): path for path in image_dir.glob("*.png")}
            label_paths = {int(path.stem): path for path in label_dir.glob("*.txt")}
            if set(image_paths) != set(record_by_id) or set(label_paths) != set(record_by_id):
                raise ValueError(f"Image/label/manifest IDs differ in {fold}/{split}")

            for observation_id, item in record_by_id.items():
                source = sources[observation_id]
                expected_meta = (item["home"], int(item["home_session_id"]), item["relative_room"])
                source_meta = (source["home"], source["home_session_id"], source["relative_room"])
                if source_meta != expected_meta:
                    raise ValueError(f"Source DB and manifest metadata differ for observation {observation_id}")
                relative = PurePosixPath(source["relative_room"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe source path in observation {observation_id}")
                source_dir = (rgb_root / Path(*relative.parts)).resolve()
                if not source_dir.is_relative_to(rgb_root.resolve()):
                    raise ValueError(f"Source path escapes RGBD root for observation {observation_id}")
                source_image = source_dir / source["image_name"]
                source_mask = source_dir / source["mask_name"]
                exported_image = image_paths[observation_id]
                exported_label = label_paths[observation_id]
                image = cv2.imread(str(exported_image), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Could not decode exported image: {exported_image}")
                image_height, image_width = image.shape[:2]
                if source_image.read_bytes() != exported_image.read_bytes():
                    raise ValueError(f"Exported RGB bytes differ from source for observation {observation_id}")

                exported_boxes = load_exported_boxes(exported_label, image_width, image_height)
                if len(exported_boxes) != int(item["boxes"]):
                    raise ValueError(f"Manifest box count differs for observation {observation_id}")
                for box in exported_boxes:
                    x1, y1, x2, y2 = box["xyxy"]
                    if x1 < -0.001 or y1 < -0.001 or x2 > image_width + 0.001 or y2 > image_height + 0.001:
                        raise ValueError(f"Box extends outside image for observation {observation_id}")

                if observation_id not in verified_observations:
                    labels, mask_values = parse_mask_file(source_mask)
                    aligned_mask, _ = align_mask(mask_values, (image_height, image_width))
                    regenerated, _ = extract_yolo_boxes(
                        source_mask, (image_height, image_width), instance_classes.get(observation_id, {})
                    )
                    pixel_counts = []
                    semantic_by_id = instance_classes.get(observation_id, {})
                    for local_id, _instance_name in labels:
                        if semantic_by_id.get(local_id) not in CLASSES:
                            continue
                        pixel_count = int(np.count_nonzero(decode_instance_mask(aligned_mask, local_id)))
                        if pixel_count:
                            pixel_counts.append(pixel_count)
                    if len(regenerated) != len(exported_boxes):
                        raise ValueError(f"Source-mask/export box count differs for observation {observation_id}")
                    if len(pixel_counts) != len(regenerated):
                        raise ValueError(f"Source-mask pixel counts do not match target boxes for observation {observation_id}")
                    for source_box, exported_box in zip(regenerated, exported_boxes):
                        if source_box[0] != exported_box["class_id"] or any(
                            not math.isclose(float(source_box[index]), exported_box[key], rel_tol=0.0, abs_tol=5.1e-7)
                            for index, key in zip(range(1, 5), ("xc", "yc", "width", "height"))
                        ):
                            raise ValueError(f"Source-mask/export geometry differs for observation {observation_id}")
                    # Keep the parsed/aligned mask check explicit: this catches corrupt or empty source matrices.
                    if not labels or aligned_mask.shape != image.shape[:2]:
                        raise ValueError(f"Invalid or unaligned source mask for observation {observation_id}")
                    source_box_pixels[observation_id] = pixel_counts
                    verified_observations.add(observation_id)

                expected_counts = item.get("instances_by_class", {})
                actual_counts = Counter(CLASSES[box["class_id"]] for box in exported_boxes)
                for class_name in CLASSES:
                    if actual_counts[class_name] != int(expected_counts.get(class_name, 0)):
                        raise ValueError(f"Per-class instance count differs for observation {observation_id}/{class_name}")
                room = item["physical_room"]
                for box_index, box in enumerate(exported_boxes):
                    pixel_width = box["pixel_width"]
                    pixel_height = box["pixel_height"]
                    short_side = min(pixel_width, pixel_height)
                    long_side = max(pixel_width, pixel_height)
                    area_fraction = box["width"] * box["height"]
                    ratio = pixel_width / pixel_height
                    x1, y1, x2, y2 = box["xyxy"]
                    touches_border = x1 <= 1.0 or y1 <= 1.0 or x2 >= image_width - 1.0 or y2 >= image_height - 1.0
                    measurements.append({
                        "fold": fold,
                        "split": split,
                        "observation_id": observation_id,
                        "home": item["home"],
                        "physical_room": room,
                        "relative_room": item["relative_room"],
                        "class_name": CLASSES[box["class_id"]],
                        "class_id": box["class_id"],
                        "box_index": box_index,
                        "image_width": image_width,
                        "image_height": image_height,
                        "pixel_width": pixel_width,
                        "pixel_height": pixel_height,
                        "short_side_px": short_side,
                        "long_side_px": long_side,
                        "area_fraction": area_fraction,
                        "aspect_ratio": ratio,
                        "mask_pixel_count": source_box_pixels[observation_id][box_index],
                        "mask_fill_fraction": source_box_pixels[observation_id][box_index] / max(1.0, pixel_width * pixel_height),
                        "touches_border": touches_border,
                        "image_path": str(exported_image),
                        "label_path": str(exported_label),
                    })
        if split_ids["train"] & split_ids["val"]:
            raise ValueError(f"Observation ID overlap in {fold} train/val")
        split_summaries[fold] = {
            split: {
                "image_count": len(manifest["splits"][split]["items"]),
                "homes": manifest["splits"][split]["homes"],
                "observation_ids": len(split_ids[split]),
            }
            for split in ("train", "val")
        }
    return measurements, {
        "verified_unique_observations": len(verified_observations),
        "manifest_unique_observations": len(observation_ids),
        "split_summaries": split_summaries,
    }


def summarize_boxes(measurements: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in measurements:
        groups[(item["fold"], item["split"], item["class_name"])].append(item)
    rows = []
    for fold in FOLDS:
        for split in ("train", "val"):
            for class_name in CLASSES:
                items = groups[(fold, split, class_name)]
                if not items:
                    rows.append({"fold": fold, "split": split, "class_name": class_name, "instances": 0})
                    continue
                short = np.asarray([x["short_side_px"] for x in items], dtype=np.float64)
                area = np.asarray([100.0 * x["area_fraction"] for x in items], dtype=np.float64)
                ratios = np.asarray([x["aspect_ratio"] for x in items], dtype=np.float64)
                rooms_with_class = {x["physical_room"] for x in items}
                all_rooms = {
                    x["physical_room"] for x in measurements
                    if x["fold"] == fold and x["split"] == split
                }
                rows.append({
                    "fold": fold,
                    "split": split,
                    "class_name": class_name,
                    "instances": len(items),
                    "rooms_with_class": len(rooms_with_class),
                    "rooms_total": len(all_rooms),
                    "short_side_p10_px": float(np.percentile(short, 10)),
                    "short_side_median_px": float(np.median(short)),
                    "small_side_lt16_count": int(np.sum(short < SMALL_SIDE_PX)),
                    "small_side_lt16_percent": float(100.0 * np.mean(short < SMALL_SIDE_PX)),
                    "area_lt0_5pct_count": int(np.sum(area < 100.0 * SMALL_AREA_FRACTION)),
                    "area_lt0_5pct_percent": float(100.0 * np.mean(area < 100.0 * SMALL_AREA_FRACTION)),
                    "area_median_percent": float(np.median(area)),
                    "mask_pixels_lt4_count": int(sum(x["mask_pixel_count"] < 4 for x in items)),
                    "single_pixel_mask_count": int(sum(x["mask_pixel_count"] == 1 for x in items)),
                    "mask_pixels_median": float(np.median([x["mask_pixel_count"] for x in items])),
                    "aspect_p10": float(np.percentile(ratios, 10)),
                    "aspect_p90": float(np.percentile(ratios, 90)),
                    "border_touch_count": int(sum(x["touches_border"] for x in items)),
                    "extreme_aspect_count": int(sum(x["aspect_ratio"] < 1.0 / EXTREME_ASPECT_RATIO
                                                     or x["aspect_ratio"] > EXTREME_ASPECT_RATIO for x in items)),
                })
    return rows


def make_gallery(measurements: list[dict], output_path: Path) -> list[dict]:
    val_items = [x for x in measurements if x["split"] == "val"]
    selected = []
    for class_name in CLASSES:
        candidates = [x for x in val_items if x["class_name"] == class_name]
        if not candidates:
            continue
        selected.append(("smallest short side", min(candidates, key=lambda x: (x["short_side_px"], x["observation_id"]))))
        selected.append(("largest area", max(candidates, key=lambda x: (x["area_fraction"], -x["observation_id"]))))

    panel_width, panel_height, header_height = 480, 300, 50
    panels = []
    colors = {"bottle": (255, 120, 20), "chair": (30, 180, 30), "table": (30, 30, 230)}
    for reason, item in selected:
        image = cv2.imread(item["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read gallery image: {item['image_path']}")
        h, w = image.shape[:2]
        scale = min(panel_width / w, panel_height / h)
        view = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        label_path = Path(item["label_path"])
        box_rows = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            class_id, xc, yc, bw, bh = int(fields[0]), *map(float, fields[1:])
            x1, y1 = round((xc - bw / 2) * w * scale), round((yc - bh / 2) * h * scale)
            x2, y2 = round((xc + bw / 2) * w * scale), round((yc + bh / 2) * h * scale)
            name = CLASSES[class_id]
            box_rows.append((x1, y1, x2, y2, name))
            cv2.rectangle(view, (x1, y1), (x2, y2), colors[name], 1)
        x1, y1, x2, y2, selected_name = box_rows[item["box_index"]]
        cv2.rectangle(view, (x1, y1), (x2, y2), colors[selected_name], 3)
        cv2.putText(view, "selected", (max(0, x1), max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, colors[selected_name], 1, cv2.LINE_AA)
        canvas = np.full((panel_height, panel_width, 3), 245, dtype=np.uint8)
        offset_x, offset_y = (panel_width - view.shape[1]) // 2, (panel_height - view.shape[0]) // 2
        canvas[offset_y:offset_y + view.shape[0], offset_x:offset_x + view.shape[1]] = view
        header = np.full((header_height, panel_width, 3), 248, dtype=np.uint8)
        line1 = f"{reason} | {item['class_name']} | {item['home']} | obs {item['observation_id']}"
        line2 = (f"box={item['pixel_width']:.1f}x{item['pixel_height']:.1f}px | mask={item['mask_pixel_count']}px | "
                 f"area={100*item['area_fraction']:.3f}% | aspect={item['aspect_ratio']:.2f}")
        cv2.putText(header, line1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(header, line2, (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(cv2.vconcat([header, canvas]))

    if not panels:
        raise ValueError("No validation boxes available for the gallery")
    while len(panels) % 2:
        panels.append(np.full_like(panels[0], 248))
    rows = [cv2.hconcat(panels[index:index + 2]) for index in range(0, len(panels), 2)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not save quality-audit gallery: {output_path}")
    return [
        {k: item[k] for k in ("home", "physical_room", "observation_id", "class_name", "short_side_px",
                              "pixel_width", "pixel_height", "mask_pixel_count", "area_fraction", "aspect_ratio")}
        | {"selection_reason": reason}
        for reason, item in selected
    ]


def render_report(data: dict) -> str:
    lines = [
        "# Robot@Home2 train/val 框尺度与来源一致性审计",
        "",
        f"- 时间：{data['started_at']}",
        f"- 审计范围：四个开发住宅折的 train/val；唯一源 observation 数 {data['source_check']['verified_unique_observations']}；没有查询或使用 sarmis。",
        f"- 来源校验：SQLite 只按 manifest 中的开发 observation ID 查询；核对 DB 房间/住宅元数据、原始 RGB 与导出 PNG 字节一致，并从原始像素 mask 重算全部 YOLO 框。",
        f"- 复核通过：{data['source_check']['verified_unique_observations']} 个唯一 observation 的类别、框数和坐标与导出标签匹配；共统计 {data['total_box_occurrences']} 个折内框实例（跨折会重复计数）。",
        "- 解释边界：短边 <16 px、面积 <0.5%、mask 像素少于 4、长宽比超出 1:5–5:1、贴图边缘均是人工复核提示，不等同错误标签；实例框由像素 mask 派生，长物体/遮挡目标可能天然产生极端值。",
        "",
        "## 四折 train/val 的类别尺度统计",
        "",
        "短边按原图像素计；面积为框面积/整图面积。每折分别统计，避免把不同折中的重复 observation 当成独立样本合并。",
        "",
        "| 折 | split | 类别 | 框数 | 有类别的物理房间 | 短边 P10/中位数(px) | 短边<16px | 面积<0.5% | mask<4像素 | mask像素中位数 | 面积中位数(%) | 长宽比P10–P90 | 贴边 | 极端比例 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["statistics"]:
        if row["instances"] == 0:
            lines.append(f"| {row['fold']} | {row['split']} | {row['class_name']} | 0 | 0/0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {row['fold']} | {row['split']} | {row['class_name']} | {row['instances']} | "
            f"{row['rooms_with_class']}/{row['rooms_total']} | {row['short_side_p10_px']:.1f}/{row['short_side_median_px']:.1f} | "
            f"{row['small_side_lt16_count']} ({row['small_side_lt16_percent']:.1f}%) | "
            f"{row['area_lt0_5pct_count']} ({row['area_lt0_5pct_percent']:.1f}%) | "
            f"{row['mask_pixels_lt4_count']} | {row['mask_pixels_median']:.0f} | {row['area_median_percent']:.3f} | "
            f"{row['aspect_p10']:.2f}–{row['aspect_p90']:.2f} | {row['border_touch_count']} | {row['extreme_aspect_count']} |"
        )
    lines.extend([
        "",
        "## 各留出住宅的房间覆盖",
        "",
        "下表只看每个住宅唯一对应的 val 折；房间数表示至少有一个该类实例框的物理房间数/该折验证房间总数。",
        "",
        "| 留出住宅 | val 图像 | 验证物理房间 | bottle 覆盖 | chair 覆盖 | table 覆盖 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for fold in FOLDS:
        val = data["source_check"]["split_summaries"][fold]["val"]
        rows = [x for x in data["statistics"] if x["fold"] == fold and x["split"] == "val"]
        total_rooms = max((x.get("rooms_total", 0) for x in rows), default=0)
        coverage = {x["class_name"]: f"{x.get('rooms_with_class', 0)}/{total_rooms}" for x in rows}
        lines.append(
            f"| {fold.removeprefix('val_')} | {val['image_count']} | {total_rooms} | "
            f"{coverage.get('bottle','0/0')} | {coverage.get('chair','0/0')} | {coverage.get('table','0/0')} |"
        )
    lines.extend([
        "",
        "## 极小/极大框样例",
        "",
        f"`{data['gallery_relative_path']}` 展示每类验证集短边最小和面积最大的样例；彩色框是导出的 GT 框。样例用于人工检查，不自动判为标注错误。",
        "",
        "## 结论与后续",
        "",
        data["interpretation"],
        "",
        "继续策略：仅对报告中短边很小、面积极小、长宽比极端或贴边的框做少量 RGB/原始 mask 人工复核；若框与 mask 一致，则将其视为目标外观/尺度困难，不修改标签。然后根据类别与房间覆盖，决定下一轮增加哪些住宅/房间样本。sarmis 继续封存。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    raw_root = args.raw_root.resolve()
    report_path = args.report.resolve()
    gallery_path = args.gallery.resolve()
    for path in (data_root, raw_root, report_path, gallery_path):
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Data and outputs must stay within project root {ROOT}: {path}")
    if report_path.exists() or report_path.with_suffix(".json").exists() or gallery_path.exists():
        raise FileExistsError("Audit report or gallery already exists; refusing to overwrite")

    fold_manifests = {}
    for fold in FOLDS:
        manifest_path = data_root / fold / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing fold manifest: {manifest_path}")
        fold_manifests[fold] = json.loads(manifest_path.read_text(encoding="utf-8"))

    measurements, source_check = validate_and_measure(data_root, raw_root, fold_manifests)
    statistics = summarize_boxes(measurements)
    small_count = sum(x["short_side_px"] < SMALL_SIDE_PX for x in measurements)
    tiny_area_count = sum(x["area_fraction"] < SMALL_AREA_FRACTION for x in measurements)
    extreme_count = sum(x["aspect_ratio"] < 1.0 / EXTREME_ASPECT_RATIO
                        or x["aspect_ratio"] > EXTREME_ASPECT_RATIO for x in measurements)
    border_count = sum(x["touches_border"] for x in measurements)
    single_pixel_masks = sum(x["mask_pixel_count"] == 1 for x in measurements)
    mask_pixels_lt4 = sum(x["mask_pixel_count"] < 4 for x in measurements)
    interpretation = (
        f"本次框/来源一致性检查未发现导出坐标、类别或 RGB 文件错配；{len(measurements)} 个折内框中，"
        f"{single_pixel_masks} 个来自单像素 source mask、{mask_pixels_lt4} 个 source mask 少于 4 像素。"
        f"尺度统计另将 {small_count}/{len(measurements)} "
        f"个框标为短边小于 16 px，{tiny_area_count} 个面积小于 0.5%，{extreme_count} 个长宽比超出 1:5–5:1，"
        f"{border_count} 个框触及图像边缘。它们是复核队列而非已确认错误；数据覆盖应按每折/房间表判断，不能因模型混淆直接断言类别映射错误。"
    )
    data = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": "Robot@Home2",
        "folds": list(FOLDS),
        "classes": list(CLASSES),
        "test_used": False,
        "source_check": source_check,
        "total_box_occurrences": len(measurements),
        "thresholds": {
            "small_side_px": SMALL_SIDE_PX,
            "small_area_fraction": SMALL_AREA_FRACTION,
            "extreme_aspect_ratio": EXTREME_ASPECT_RATIO,
        },
        "flagged_counts": {
            "small_side": small_count,
            "small_area": tiny_area_count,
            "mask_pixels_lt4": mask_pixels_lt4,
            "single_pixel_masks": single_pixel_masks,
            "extreme_aspect": extreme_count,
            "touches_border": border_count,
        },
        "statistics": statistics,
        "gallery_examples": [],
        "gallery_relative_path": str(gallery_path.relative_to(ROOT)).replace("\\", "/"),
        "interpretation": interpretation,
    }
    data["gallery_examples"] = make_gallery(measurements, gallery_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(data), encoding="utf-8")
    report_path.with_suffix(".json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Source-verified observations: {source_check['verified_unique_observations']}")
    print(f"Box occurrences: {len(measurements)}; flagged small={small_count}, tiny_area={tiny_area_count}, "
          f"extreme_aspect={extreme_count}, border={border_count}")
    print(f"Report: {report_path}")
    print(f"Gallery: {gallery_path}")


if __name__ == "__main__":
    main()
