"""Build a small, reproducible YOLO detection pilot from Robot@Home2 masks."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from visionnav.inspect_robotathome_masks import align_mask, decode_instance_mask, parse_mask_file


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "datasets" / "RobotAtHome2" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "datasets" / "RobotAtHome2" / "processed" / "yolo_pilot"
CLASSES = ("bottle", "chair", "table")
SPLIT_SESSIONS = {"train": {0, 1, 2}, "val": {3}, "test": {4, 5, 6}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class-limit", type=int, default=30,
                        help="Maximum selected frames per class and split; this is a pilot, not a full dataset conversion.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_candidates(database: Path) -> dict[int, dict]:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    query = """
        SELECT DISTINCT o.id, o.home_session_id, h.name, m.new_path,
                        m.new_file_2, m.new_file_3, t.name
        FROM rh_lblrgbd AS o
        JOIN rh_homes AS h ON h.id = o.home_id
        JOIN rh2_old2new_rgbd_files AS m ON m.id = o.id
        JOIN rh_lblrgbd_labels AS l ON l.sensor_observation_id = o.id
        JOIN rh_object_types AS t ON t.id = l.object_type_id
        WHERE t.name IN (?, ?, ?)
        ORDER BY o.id, t.name
    """
    candidates: dict[int, dict] = {}
    with sqlite3.connect(uri, uri=True) as connection:
        for observation_id, session_id, home_name, new_path, image_name, label_name, class_name in connection.execute(query, CLASSES):
            item = candidates.setdefault(observation_id, {
                "id": int(observation_id),
                "session_id": int(session_id),
                "home": home_name,
                "new_path": new_path,
                "image_name": image_name,
                "label_name": label_name,
                "classes": set(),
            })
            item["classes"].add(class_name)
    return candidates


def load_instance_classes(database: Path, observation_ids: list[int]) -> dict[int, dict[int, str]]:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    mapping: dict[int, dict[int, str]] = defaultdict(dict)
    chunk_size = 800  # Stay below SQLite's common bind-variable limit.
    with sqlite3.connect(uri, uri=True) as connection:
        for start in range(0, len(observation_ids), chunk_size):
            chunk = observation_ids[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT l.sensor_observation_id, l.local_id, t.name
                FROM rh_lblrgbd_labels AS l
                JOIN rh_object_types AS t ON t.id = l.object_type_id
                WHERE l.sensor_observation_id IN ({placeholders})
            """
            for observation_id, local_id, class_name in connection.execute(query, chunk):
                mapping[int(observation_id)][int(local_id)] = class_name
    return mapping


def extract_yolo_boxes(label_path: Path, image_shape: tuple[int, int], instance_classes: dict[int, str]) -> tuple[list[tuple[int, float, float, float, float]], Counter]:
    labels, mask_values = parse_mask_file(label_path)
    mask_values, _ = align_mask(mask_values, image_shape)
    height, width = image_shape
    boxes = []
    counts: Counter = Counter()
    seen_local_ids = set()
    for local_id, _instance_name in labels:
        if local_id in seen_local_ids:
            raise ValueError(f"Duplicate local_id={local_id} in {label_path}")
        seen_local_ids.add(local_id)
        class_name = instance_classes.get(local_id)
        if class_name is None:
            raise ValueError(f"No database semantic mapping for local_id={local_id} in {label_path}")
        if class_name not in CLASSES:
            continue
        instance_mask = decode_instance_mask(mask_values, local_id)
        ys, xs = np.nonzero(instance_mask)
        if not len(xs):
            counts[f"empty_{class_name}"] += 1
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid box {(x1, y1, x2, y2)} in {label_path}")
        class_id = CLASSES.index(class_name)
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        boxes.append((class_id, xc, yc, box_width, box_height))
        counts[class_name] += 1
    return boxes, counts


def prepare_candidate(item: dict, raw_root: Path, database: Path) -> dict:
    rgb_root = raw_root / "files" / "rgbd"
    relative = PurePosixPath(item["new_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path in database for observation {item['id']}: {item['new_path']}")
    source_dir = rgb_root.joinpath(*relative.parts)
    source_image = source_dir / item["image_name"]
    source_label = source_dir / item["label_name"]
    if not source_image.is_file() or not source_label.is_file():
        raise FileNotFoundError(f"Missing pair for observation {item['id']}: {source_image}, {source_label}")
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode RGB image: {source_image}")
    height, width = image.shape[:2]
    instance_classes = load_instance_classes(database, [item["id"]]).get(item["id"], {})
    boxes, box_counts = extract_yolo_boxes(source_label, (height, width), instance_classes)
    return {
        "image": source_image,
        "label": source_label,
        "boxes": boxes,
        "box_counts": box_counts,
        "target_names": {CLASSES[class_id] for class_id, *_ in boxes},
    }


def write_preview(preview_candidates: dict[tuple[str, str], tuple[Path, Path]], output_path: Path) -> None:
    tile_width, tile_height, header_height = 240, 320, 26
    rows = []
    for split in ("train", "val", "test"):
        tiles = []
        for class_name in CLASSES:
            key = (split, class_name)
            tile = np.full((tile_height + header_height, tile_width, 3), 245, dtype=np.uint8)
            cv2.putText(tile, f"{split} / {class_name}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            if key in preview_candidates:
                image_path, label_path = preview_candidates[key]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Could not read preview image: {image_path}")
                image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    class_id, xc, yc, bw, bh = line.split()
                    class_id = int(class_id)
                    if CLASSES[class_id] != class_name:
                        continue
                    height, width = image.shape[:2]
                    x1 = round((float(xc) - float(bw) / 2) * width)
                    y1 = round((float(yc) - float(bh) / 2) * height)
                    x2 = round((float(xc) + float(bw) / 2) * width)
                    y2 = round((float(yc) + float(bh) / 2) * height)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (30, 30, 230), 2)
                tile[header_height:] = image
            tiles.append(tile)
        rows.append(cv2.hconcat(tiles))
    if not cv2.imwrite(str(output_path), cv2.vconcat(rows)):
        raise OSError(f"Could not write pilot preview: {output_path}")


def main() -> None:
    args = parse_args()
    database = args.raw_root / "rh.db"
    rgb_root = args.raw_root / "files" / "rgbd"
    if not database.is_file() or not rgb_root.is_dir():
        raise FileNotFoundError(f"Expected Robot@Home2 raw data under {args.raw_root}")
    if args.per_class_limit < 1:
        raise ValueError("--per-class-limit must be positive")
    output = args.output.resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Output or staging path already exists; refusing to overwrite: {output} / {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(database)
    sampled: dict[str, dict[int, dict]] = {split: {} for split in SPLIT_SESSIONS}
    selected_for: dict[tuple[str, int], set[str]] = defaultdict(set)
    rng = random.Random(args.seed)
    selection_report = {}
    prepared_by_id: dict[int, dict] = {}
    rejected_without_pixels: Counter = Counter()

    # Shuffle candidate order deterministically, then scan until the requested
    # number of frames has a visible, non-empty mask for the target class.
    for split, sessions in SPLIT_SESSIONS.items():
        selection_report[split] = {}
        for class_name in CLASSES:
            available = [
                item for item in candidates.values()
                if item["session_id"] in sessions and class_name in item["classes"]
            ]
            rng.shuffle(available)
            requested = min(args.per_class_limit, len(available))
            selected_count = 0
            inspected_count = 0
            for item in available:
                if selected_count >= requested:
                    break
                inspected_count += 1
                observation_id = item["id"]
                prepared = prepared_by_id.get(observation_id)
                if prepared is None:
                    prepared = prepare_candidate(item, args.raw_root, database)
                    prepared_by_id[observation_id] = prepared
                if class_name not in prepared["target_names"]:
                    rejected_without_pixels[class_name] += 1
                    continue
                sampled[split][observation_id] = item
                selected_for[(split, observation_id)].add(class_name)
                selected_count += 1
            selection_report[split][class_name] = {
                "database_candidate_frames": len(available),
                "requested_frames": requested,
                "selected_frames": selected_count,
                "candidates_inspected": inspected_count,
            }

    all_ids = [observation_id for rows in sampled.values() for observation_id in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Same observation was selected into more than one split")
    for split, rows in sampled.items():
        for observation_id in rows:
            missing_selected = selected_for[(split, observation_id)] - prepared_by_id[observation_id]["target_names"]
            if missing_selected:
                raise ValueError(f"Selected observation {observation_id} lacks masks for {sorted(missing_selected)}")

    staging.mkdir(parents=True)
    manifest = {
        "dataset": "Robot@Home2",
        "purpose": "small conversion pilot; not a full dataset",
        "seed": args.seed,
        "per_class_limit_per_split": args.per_class_limit,
        "classes": list(CLASSES),
        "split_sessions": {key: sorted(value) for key, value in SPLIT_SESSIONS.items()},
        "selection": selection_report,
        "candidates_skipped_without_pixels_by_class": dict(rejected_without_pixels),
        "splits": {},
        "notes": [
            "Mask conversion uses the verified counterclockwise rotation and two-column crop on each side.",
            "Each target instance mask becomes one axis-aligned YOLO box; empty masks are skipped and counted.",
            "No model was trained and no test metrics were computed in this preparation step.",
        ],
    }
    preview_candidates = {}
    for split, rows in sampled.items():
        image_dir = staging / "images" / split
        label_dir = staging / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        split_instances: Counter = Counter()
        split_items = []
        split_homes = set()
        for observation_id, item in sorted(rows.items()):
            record = prepared_by_id[observation_id]
            source_image = record["image"]
            boxes = record["boxes"]
            box_counts = record["box_counts"]
            target_names = record["target_names"]
            split_instances.update(box_counts)
            split_homes.add(item["home"])
            destination_image = image_dir / f"{observation_id}.png"
            destination_label = label_dir / f"{observation_id}.txt"
            shutil.copyfile(source_image, destination_image)
            destination_label.write_text(
                "".join(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n" for class_id, xc, yc, bw, bh in boxes),
                encoding="utf-8",
            )
            for class_name in target_names:
                preview_candidates.setdefault((split, class_name), (destination_image, destination_label))
            split_items.append({
                "observation_id": observation_id,
                "home": item["home"],
                "home_session_id": item["session_id"],
                "relative_room": item["new_path"],
                "selected_for": sorted(selected_for[(split, observation_id)]),
                "boxes": len(boxes),
                "instances_by_class": dict(box_counts),
            })
        manifest["splits"][split] = {
            "homes": sorted(split_homes),
            "image_count": len(split_items),
            "instances_by_class": dict(split_instances),
            "items": split_items,
        }

    yaml_path = staging / "data.yaml"
    yaml_path.write_text(
        f'path: "{output.as_posix()}"\n'
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names: [{', '.join(CLASSES)}]\n",
        encoding="utf-8",
    )
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    final_preview_candidates = {
        key: (output / image.relative_to(staging), output / label.relative_to(staging))
        for key, (image, label) in preview_candidates.items()
    }
    preview_path = ROOT / "outputs" / "robotathome2_yolo_pilot_preview.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    write_preview(final_preview_candidates, preview_path)
    print(f"Dataset: {output}")
    print(f"YAML: {output / 'data.yaml'}")
    print(f"Preview: {preview_path}")
    for split, summary in manifest["splits"].items():
        print(f"{split}: {summary['image_count']} images; homes={summary['homes']}; instances={summary['instances_by_class']}")


if __name__ == "__main__":
    main()
