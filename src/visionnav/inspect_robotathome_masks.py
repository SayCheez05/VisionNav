"""Preview Robot@Home2 per-instance bit-mask labels over one RGB image."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

import cv2
import numpy as np


DATA_ROOT = ROOT / "data" / "datasets" / "RobotAtHome2" / "raw"
SAMPLE_DIR = DATA_ROOT / "files" / "rgbd" / "session_1" / "alma" / "bathroom1" / "subsession_1"
DEFAULT_ID = "100000"
DEFAULT_OUTPUT = ROOT / "outputs" / "robotathome2_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=SAMPLE_DIR / f"{DEFAULT_ID}_intensity.png")
    parser.add_argument("--labels", type=Path, default=SAMPLE_DIR / f"{DEFAULT_ID}_labels.txt")
    parser.add_argument("--database", type=Path, default=DATA_ROOT / "rh.db")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_mask_file(path: Path) -> tuple[list[tuple[int, str]], np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"No mask data found in {path}")
    count = int(lines[0])
    labels = []
    for line in lines[1:1 + count]:
        local_id_text, name = line.split(maxsplit=1)
        labels.append((int(local_id_text), name))
    row_lines = lines[1 + count:]
    if not row_lines:
        raise ValueError(f"No mask matrix found in {path}")
    width = len(row_lines[0].split())
    mask_values = np.empty((len(row_lines), width), dtype=np.uint32)
    for row_index, line in enumerate(row_lines):
        row = np.fromstring(line, sep=" ", dtype=np.uint32)
        if row.size != width:
            raise ValueError(f"Mask row {row_index} has {row.size} values; expected {width}")
        mask_values[row_index] = row
    return labels, mask_values


def decode_instance_mask(mask_values: np.ndarray, local_id: int) -> np.ndarray:
    """Decode one 1-based instance bit, treating unrepresentable IDs as empty."""
    if local_id < 0:
        raise ValueError(f"local_id must be non-negative, got {local_id}")
    bit_width = np.iinfo(mask_values.dtype).bits
    if local_id >= bit_width:
        return np.zeros(mask_values.shape, dtype=bool)
    return (mask_values & np.array(1 << local_id, dtype=mask_values.dtype)) != 0


def align_mask(mask_values: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    if mask_values.shape == target_shape:
        return mask_values, "identity"
    rotated = np.rot90(mask_values, 1)
    if rotated.shape[0] == target_shape[0] and rotated.shape[1] == target_shape[1] + 4:
        return rotated[:, 2:-2], "rotate_counterclockwise_then_crop_2_columns_each_side"
    raise ValueError(
        f"Cannot safely align label matrix {mask_values.shape} to image {target_shape}; "
        "expected same dimensions or the verified Robot@Home2 90-degree rotation with 2-column crop."
    )


def read_semantic_names(database: Path, observation_id: int) -> dict[int, str]:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT l.local_id, t.name "
            "FROM rh_lblrgbd_labels AS l "
            "LEFT JOIN rh_object_types AS t ON t.id = l.object_type_id "
            "WHERE l.sensor_observation_id = ?",
            (observation_id,),
        ).fetchall()
    return {int(local_id): semantic_name for local_id, semantic_name in rows if semantic_name}


def stable_color(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    rgb = np.array(cv2.cvtColor(np.uint8([[[round(hue * 179), 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])
    return tuple(int(value) for value in rgb)


def main() -> None:
    args = parse_args()
    for path in (args.image, args.labels, args.database):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read RGB image: {args.image}")
    labels, mask_values = parse_mask_file(args.labels)
    height, width = image.shape[:2]
    mask_values, transform = align_mask(mask_values, (height, width))

    match = re.search(r"(\d+)_intensity$", args.image.stem)
    observation_id = int(match.group(1)) if match else int(args.image.stem)
    semantic_names = read_semantic_names(args.database, observation_id)
    overlay = image.copy()
    report_rows = []
    for local_id, instance_name in labels:
        # Robot@Home2 documents bit positions as 1-based: local_id 1 uses bit 1.
        instance_mask = decode_instance_mask(mask_values, local_id)
        ys, xs = np.nonzero(instance_mask)
        if not len(xs):
            report_rows.append((local_id, instance_name, semantic_names.get(local_id, "unknown"), 0, None))
            continue
        color = stable_color(local_id)
        overlay[instance_mask] = (0.55 * overlay[instance_mask] + 0.45 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        semantic_name = semantic_names.get(local_id, "unknown")
        cv2.putText(overlay, f"{semantic_name}:{local_id}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        report_rows.append((local_id, instance_name, semantic_name, int(len(xs)), (x1, y1, x2, y2)))

    header = np.full((28, width * 2, 3), 245, dtype=np.uint8)
    cv2.putText(header, f"Robot@Home2 | RGB (left) vs instance masks + derived boxes (right) | observation {observation_id}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    preview = cv2.vconcat([header, cv2.hconcat([image, overlay])])
    args.output.mkdir(parents=True, exist_ok=True)
    image_output = args.output / f"{observation_id}_mask_preview.jpg"
    if not cv2.imwrite(str(image_output), preview):
        raise OSError(f"Could not write preview image: {image_output}")

    report = [
        f"# Robot@Home2 样例标注检查：{observation_id}",
        "",
        f"- RGB 图像：`{args.image}`（{width}×{height}）",
        f"- 掩码标注：`{args.labels}`；采用对齐变换：`{transform}`。",
        "- 标签文件把多个实例编码为每像素的 bit mask；每个 local_id 对应一个实例，数据库提供其语义类别。",
        "- 本工具只生成预览和由像素掩码推导的矩形框，不写入/改动原始数据，也不生成训练集。",
        "",
        f"![样例掩码预览](./{image_output.name})",
        "",
        "| local_id | 实例名 | 数据库类别 | 掩码像素数 | 派生框（x1,y1,x2,y2） |",
        "|---:|---|---|---:|---|",
    ]
    for local_id, instance_name, semantic_name, pixels, box in report_rows:
        report.append(f"| {local_id} | {instance_name} | {semantic_name} | {pixels} | `{box}` |")
    report_path = args.output / f"{observation_id}_mask_preview.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Image: {width}x{height}; aligned mask: {mask_values.shape[1]}x{mask_values.shape[0]}; transform: {transform}; instances: {len(labels)}")
    print(f"Preview: {image_output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
