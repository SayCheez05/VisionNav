"""Build a complete, home-grouped Robot@Home2 CV dataset with hard negatives."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2

from visionnav.prepare_robotathome_yolo import (
    CLASSES,
    ROOT,
    extract_yolo_boxes,
    load_candidates,
    load_instance_classes,
)


HOMES = ("alma", "anto", "pare", "rx2")
TEST_HOME = "sarmis"
RAW_ROOT = ROOT / "data" / "datasets" / "RobotAtHome2" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "datasets" / "RobotAtHome2" / "processed" / "yolo_cv_candidate_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative-ratio", type=float, default=0.20,
                        help="Selected target-absent frames per valid positive frame, balanced by room/acquisition.")
    return parser.parse_args()


def physical_room(item: dict) -> str:
    parts = PurePosixPath(item["new_path"]).parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected Robot@Home2 relative path: {item['new_path']!r}")
    return "/".join(parts[1:3])


def acquisition_group(item: dict) -> str:
    return item["new_path"]


def load_target_absent_candidates(database: Path) -> list[dict]:
    """Read observations with RGB/mask mappings and no labels for target classes."""
    uri = f"{database.resolve().as_uri()}?mode=ro"
    query = """
        SELECT o.id, o.home_session_id, h.name, m.new_path, m.new_file_2, m.new_file_3
        FROM rh_lblrgbd AS o
        JOIN rh_homes AS h ON h.id = o.home_id
        JOIN rh2_old2new_rgbd_files AS m ON m.id = o.id
        WHERE h.name IN (?, ?, ?, ?)
          AND NOT EXISTS (
              SELECT 1
              FROM rh_lblrgbd_labels AS l
              JOIN rh_object_types AS t ON t.id = l.object_type_id
              WHERE l.sensor_observation_id = o.id AND t.name IN (?, ?, ?)
          )
        ORDER BY h.name, m.new_path, o.id
    """
    items = []
    with sqlite3.connect(uri, uri=True) as connection:
        for observation_id, session_id, home, new_path, image_name, label_name in connection.execute(
            query, (*HOMES, *CLASSES)
        ):
            items.append({
                "id": int(observation_id),
                "session_id": int(session_id),
                "home": home,
                "new_path": new_path,
                "image_name": image_name,
                "label_name": label_name,
                "classes": set(),
            })
    return items


def allocate_evenly(capacities: dict[str, int], requested: int) -> dict[str, int]:
    """Allocate a quota round-robin so long rooms/acquisitions do not dominate."""
    allocation = {key: 0 for key in sorted(capacities)}
    remaining = min(max(0, requested), sum(capacities.values()))
    while remaining:
        changed = False
        for key in allocation:
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            break
    return allocation


def spaced_rows(items: list[dict], count: int) -> list[dict]:
    """Select evenly across observation-ID order to reduce adjacent-frame sampling."""
    rows = sorted(items, key=lambda item: item["id"])
    if count >= len(rows):
        return rows
    indices = [min(len(rows) - 1, int((index + 0.5) * len(rows) / count)) for index in range(count)]
    if len(set(indices)) != len(indices):
        raise ValueError("Systematic frame selection produced repeated indices")
    return [rows[index] for index in indices]


def select_negative_frames(positives_by_home: Counter, negatives: list[dict], ratio: float) -> tuple[list[dict], dict]:
    by_home_room: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in negatives:
        by_home_room[item["home"]][physical_room(item)][acquisition_group(item)].append(item)

    selected = []
    summary = {}
    for home in HOMES:
        target = round(positives_by_home[home] * ratio)
        rooms = by_home_room.get(home, {})
        room_capacities = {room: sum(len(rows) for rows in groups.values()) for room, groups in rooms.items()}
        room_quotas = allocate_evenly(room_capacities, target)
        home_selected = []
        selected_by_room = {}
        for room, room_quota in room_quotas.items():
            groups = rooms[room]
            group_capacities = {group: len(rows) for group, rows in groups.items()}
            group_quotas = allocate_evenly(group_capacities, room_quota)
            room_rows = []
            for group, quota in group_quotas.items():
                room_rows.extend(spaced_rows(groups[group], quota))
            home_selected.extend(room_rows)
            selected_by_room[room] = len(room_rows)
        if len(home_selected) < target:
            raise ValueError(f"Only {len(home_selected)} negative frames available for {home}; requested {target}")
        selected.extend(home_selected)
        summary[home] = {
            "valid_positive_frames": positives_by_home[home],
            "negative_target": target,
            "negative_selected": len(home_selected),
            "negative_by_physical_room": selected_by_room,
        }
    ids = [item["id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Negative-frame sampling selected duplicate observations")
    return selected, summary


def source_paths(item: dict, raw_root: Path) -> tuple[Path, Path]:
    relative = PurePosixPath(item["new_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe database path for observation {item['id']}: {item['new_path']!r}")
    source_dir = raw_root / "files" / "rgbd" / Path(*relative.parts)
    image_path = source_dir / item["image_name"]
    mask_path = source_dir / item["label_name"]
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing source RGB/mask for observation {item['id']}: {image_path}, {mask_path}")
    return image_path, mask_path


def prepare_records(raw_root: Path, database: Path, negative_ratio: float) -> tuple[dict[int, dict], dict]:
    candidates = load_candidates(database)
    positive_candidates = [item for item in candidates.values() if item["home"] in HOMES]
    positive_candidates.sort(key=lambda item: item["id"])
    class_mapping = load_instance_classes(database, [item["id"] for item in positive_candidates])

    records = {}
    valid_positive_by_home: Counter = Counter()
    positive_candidates_by_home: Counter = Counter()
    empty_mask_candidates_by_home: Counter = Counter()
    empty_mask_instances_by_class: Counter = Counter()
    positive_instances: Counter = Counter()
    for index, item in enumerate(positive_candidates, 1):
        image_path, mask_path = source_paths(item, raw_root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot decode source RGB for observation {item['id']}: {image_path}")
        boxes, class_counts = extract_yolo_boxes(mask_path, image.shape[:2], class_mapping.get(item["id"], {}))
        positive_candidates_by_home[item["home"]] += 1
        empty_diagnostics = {
            name: count for name, count in class_counts.items()
            if name.startswith("empty_") and count
        }
        for name, count in empty_diagnostics.items():
            empty_mask_instances_by_class[name.removeprefix("empty_")] += count
        if not boxes:
            empty_mask_candidates_by_home[item["home"]] += 1
            continue
        physical = physical_room(item)
        records[item["id"]] = {
            "id": item["id"],
            "home": item["home"],
            "session_id": item["session_id"],
            "new_path": item["new_path"],
            "physical_room": physical,
            "source_image": image_path,
            "boxes": boxes,
            "instances_by_class": {name: class_counts[name] for name in CLASSES if class_counts[name]},
            "source_mask_diagnostics": empty_diagnostics,
            "selected_for": sorted(name for name in CLASSES if class_counts[name]),
            "is_negative": False,
        }
        valid_positive_by_home[item["home"]] += 1
        positive_instances.update(class_counts)
        if index % 1000 == 0:
            print(f"Prepared positive observations: {index}/{len(positive_candidates)}")

    negative_candidates = load_target_absent_candidates(database)
    selected_negatives, negative_summary = select_negative_frames(
        valid_positive_by_home, negative_candidates, ratio=negative_ratio
    )
    for index, item in enumerate(selected_negatives, 1):
        image_path, _mask_path = source_paths(item, raw_root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot decode negative source RGB for observation {item['id']}: {image_path}")
        if item["id"] in records:
            raise ValueError(f"Observation selected as both positive and negative: {item['id']}")
        records[item["id"]] = {
            "id": item["id"],
            "home": item["home"],
            "session_id": item["session_id"],
            "new_path": item["new_path"],
            "physical_room": physical_room(item),
            "source_image": image_path,
            "boxes": [],
            "instances_by_class": {},
            "source_mask_diagnostics": {},
            "selected_for": [],
            "is_negative": True,
        }
        if index % 500 == 0:
            print(f"Prepared negative observations: {index}/{len(selected_negatives)}")

    summary = {
        "positive_candidate_frames_with_target_labels": dict(positive_candidates_by_home),
        "positive_frames_with_visible_target_masks": dict(valid_positive_by_home),
        "positive_candidates_without_visible_target_masks": dict(empty_mask_candidates_by_home),
        "empty_mask_instances_by_class": {name: empty_mask_instances_by_class[name] for name in CLASSES},
        "positive_instances_by_class": {name: positive_instances[name] for name in CLASSES},
        "target_absent_candidates_by_home": dict(Counter(item["home"] for item in negative_candidates)),
        "negative_sampling": negative_summary,
    }
    return records, summary


def write_dataset(records: dict[int, dict], summary: dict, output_root: Path, raw_root: Path, ratio: float) -> None:
    output_root = output_root.resolve()
    if not output_root.is_relative_to(ROOT):
        raise ValueError(f"Dataset output must remain under project root {ROOT}: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or staging.exists():
        raise FileExistsError(f"Output or staging path exists; refusing to overwrite: {output_root} / {staging}")

    by_home = {home: [] for home in HOMES}
    for item in records.values():
        if item["home"] not in by_home or item["home"] == TEST_HOME:
            raise ValueError(f"Unexpected/final-test home in development records: {item['home']!r}")
        by_home[item["home"]].append(item)

    try:
        image_dir = staging / "images" / "all"
        label_dir = staging / "labels" / "all"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        all_manifest_items = []
        frame_sizes = Counter()
        for index, observation_id in enumerate(sorted(records), 1):
            item = records[observation_id]
            image_path = item["source_image"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Cannot decode source image at export time: {image_path}")
            frame_sizes[(image.shape[1], image.shape[0])]+=1
            destination_image = image_dir / f"{observation_id}.png"
            destination_label = label_dir / f"{observation_id}.txt"
            shutil.copyfile(image_path, destination_image)
            destination_label.write_text(
                "".join(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n" for class_id, xc, yc, bw, bh in item["boxes"]),
                encoding="utf-8",
            )
            all_manifest_items.append({
                "observation_id": observation_id,
                "home": item["home"],
                "home_session_id": item["session_id"],
                "relative_room": item["new_path"],
                "physical_room": item["physical_room"],
                "is_negative": item["is_negative"],
                "boxes": len(item["boxes"]),
                "instances_by_class": item["instances_by_class"],
                "source_mask_diagnostics": item["source_mask_diagnostics"],
                "selected_for": item["selected_for"],
            })
            if index % 1000 == 0:
                print(f"Exported dataset images/labels: {index}/{len(records)}")

        for home in HOMES:
            home_dir = staging / f"val_{home}"
            final_home_dir = output_root / f"val_{home}"
            home_dir.mkdir()
            train_rows = [item for item in all_manifest_items if item["home"] != home]
            val_rows = [item for item in all_manifest_items if item["home"] == home]
            split_specs = {"train": train_rows, "val": val_rows}
            for split, rows in split_specs.items():
                image_list = home_dir / f"{split}.txt"
                image_list.write_text(
                    "".join(f"{(output_root / 'images' / 'all' / (str(row['observation_id']) + '.png')).as_posix()}\n" for row in rows),
                    encoding="utf-8",
                )
            fold_manifest = {
                "dataset": "Robot@Home2",
                "purpose": "expanded house-grouped leave-one-house-out development fold with target-absent frames",
                "version": "candidate_v3",
                "fold": f"val_{home}",
                "classes": list(CLASSES),
                "train_homes": [other for other in HOMES if other != home],
                "validation_home": home,
                "excluded_final_test_home": TEST_HOME,
                "split_homes": {"train": [other for other in HOMES if other != home], "val": [home]},
                "negative_ratio_target": ratio,
                "source_summary": summary,
                "splits": {},
                "notes": [
                    "All selected positive observations with visible target masks from the four development homes are included.",
                    "Target-absent frames are selected by round-robin physical room/acquisition groups and evenly spaced observation IDs.",
                    "A zero-byte YOLO label file is an intentional target-absent sample; objects outside the three target classes are ignored.",
                    "sarmis is excluded and must remain unused for model selection.",
                ],
            }
            for split, rows in split_specs.items():
                instances: Counter = Counter()
                positive_count = 0
                negative_count = 0
                for row in rows:
                    instances.update(row["instances_by_class"])
                    positive_count += not row["is_negative"]
                    negative_count += row["is_negative"]
                homes_for_split = fold_manifest["split_homes"][split]
                fold_manifest["splits"][split] = {
                    "homes": homes_for_split,
                    "image_count": len(rows),
                    "positive_image_count": positive_count,
                    "negative_image_count": negative_count,
                    "physical_rooms": sorted({row["physical_room"] for row in rows}),
                    "instances_by_class": {name: instances[name] for name in CLASSES},
                    "items": rows,
                }
            fold_manifest["all_dev_image_count"] = len(all_manifest_items)
            (home_dir / "manifest.json").write_text(
                json.dumps(fold_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (home_dir / "data.yaml").write_text(
                f'path: "{final_home_dir.as_posix()}"\n'
                f'train: "{(final_home_dir / "train.txt").as_posix()}"\n'
                f'val: "{(final_home_dir / "val.txt").as_posix()}"\n'
                f"names: [{', '.join(CLASSES)}]\n",
                encoding="utf-8",
            )

        root_manifest = {
            "dataset": "Robot@Home2",
            "version": "candidate_v3",
            "classes": list(CLASSES),
            "development_homes": list(HOMES),
            "excluded_final_test_home": TEST_HOME,
            "negative_ratio_target": ratio,
            "source_database": str((raw_root / "rh.db").resolve()),
            "image_size_width_height": {f"{width}x{height}": count for (width, height), count in frame_sizes.items()},
            "summary": summary,
            "total_unique_observations": len(records),
            "all_items": all_manifest_items,
            "notes": [
                "One canonical copy of each development RGB image and YOLO label is stored under images/all and labels/all.",
                "Each fold uses file lists referencing this canonical inventory to avoid duplicating image storage.",
                "Image lists are intentionally absolute paths under this project dataset root for Ultralytics.",
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(root_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"Dataset: {output_root}")
    print(f"Unique observations: {len(records)}; image sizes (width x height): {dict(frame_sizes)}")
    for home in HOMES:
        split = json.loads((output_root / f"val_{home}" / "manifest.json").read_text(encoding="utf-8"))["splits"]
        print(f"val_{home}: train={split['train']['image_count']} (positive={split['train']['positive_image_count']}, negative={split['train']['negative_image_count']}), "
              f"val={split['val']['image_count']} (positive={split['val']['positive_image_count']}, negative={split['val']['negative_image_count']})")


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    database = raw_root / "rh.db"
    rgb_root = raw_root / "files" / "rgbd"
    if not database.is_file() or not rgb_root.is_dir():
        raise FileNotFoundError(f"Expected Robot@Home2 source files under {raw_root}")
    if not 0.0 <= args.negative_ratio <= 1.0:
        raise ValueError("--negative-ratio must be between 0 and 1")
    print("Scanning and decoding all target-positive development observations...")
    records, summary = prepare_records(raw_root, database, args.negative_ratio)
    if any(home not in {item["home"] for item in records.values()} for home in HOMES):
        raise ValueError("At least one development home has no valid positive/negative observations")
    print("Selecting target-absent observations...")
    write_dataset(records, summary, args.output_root, raw_root, args.negative_ratio)


if __name__ == "__main__":
    main()
