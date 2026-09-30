"""Create house-grouped Robot@Home2 cross-validation datasets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from visionnav.prepare_robotathome_yolo import (
    CLASSES,
    DEFAULT_OUTPUT,
    RAW_ROOT,
    ROOT,
    load_candidates,
    prepare_candidate,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT.parent / "yolo_cv_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per-class-limit", type=int, default=30,
                        help="Maximum valid frames per class in each fold split; defaults to a small review pilot.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-home", default="sarmis", help="Excluded from every development fold.")
    parser.add_argument("--preview-prefix", default=None,
                        help="Prefix for preview image filenames; defaults to a version-specific prefix.")
    return parser.parse_args()


def physical_room(item: dict) -> str:
    parts = PurePosixPath(item["new_path"]).parts
    return "/".join(parts[1:3]) if len(parts) >= 3 else item["new_path"]


def balanced_candidate_order(items: list[dict], rng: random.Random) -> list[dict]:
    """Round-robin across physical rooms, then acquisition groups within each room."""
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        grouped[physical_room(item)][item["new_path"]].append(item)

    room_queues: dict[str, list[dict]] = {}
    for room, acquisition_groups in grouped.items():
        group_names = list(acquisition_groups)
        rng.shuffle(group_names)
        for group_name in group_names:
            rng.shuffle(acquisition_groups[group_name])
        room_queue = []
        while any(acquisition_groups[name] for name in group_names):
            for group_name in group_names:
                if acquisition_groups[group_name]:
                    room_queue.append(acquisition_groups[group_name].pop())
        room_queues[room] = room_queue

    room_names = list(room_queues)
    rng.shuffle(room_names)
    ordered = []
    while any(room_queues[room] for room in room_names):
        for room in room_names:
            if room_queues[room]:
                ordered.append(room_queues[room].pop())
    return ordered


def fold_specs(candidates: dict[int, dict], test_home: str) -> list[dict]:
    homes = {item["home"] for item in candidates.values()}
    if test_home not in homes:
        raise ValueError(f"Final test home {test_home!r} not found. Available homes: {sorted(homes)}")
    development_homes = sorted(homes - {test_home})
    if len(development_homes) < 2:
        raise ValueError("At least two development homes are required")
    return [
        {
            "name": f"val_{val_home}",
            "train_homes": sorted(set(development_homes) - {val_home}),
            "val_home": val_home,
        }
        for val_home in development_homes
    ]


def select_fold(
    spec: dict,
    candidates: dict[int, dict],
    prepared_cache: dict[int, dict],
    raw_root: Path,
    limit: int,
    seed: int,
) -> dict:
    split_homes = {"train": set(spec["train_homes"]), "val": {spec["val_home"]}}
    sampled: dict[str, dict[int, dict]] = {split: {} for split in split_homes}
    selected_for: dict[tuple[str, int], set[str]] = defaultdict(set)
    selection = {split: {} for split in split_homes}
    skipped_without_pixels: dict[str, dict[str, int]] = {split: {} for split in split_homes}

    for split_index, (split, homes) in enumerate(split_homes.items()):
        for class_index, class_name in enumerate(CLASSES):
            pool = [
                item for item in candidates.values()
                if item["home"] in homes and class_name in item["classes"]
            ]
            rng = random.Random(seed + split_index * 1009 + class_index * 97 + sum(map(ord, spec["name"])))
            ordered = balanced_candidate_order(pool, rng)
            requested = min(limit, len(ordered))
            selected_count = 0
            inspected = 0
            selected_by_room: Counter = Counter()
            skipped = 0
            for item in ordered:
                if selected_count >= requested:
                    break
                inspected += 1
                observation_id = item["id"]
                prepared = prepared_cache.get(observation_id)
                if prepared is None:
                    prepared = prepare_candidate(item, raw_root, raw_root / "rh.db")
                    prepared_cache[observation_id] = prepared
                if class_name not in prepared["target_names"]:
                    skipped += 1
                    continue
                sampled[split][observation_id] = item
                selected_for[(split, observation_id)].add(class_name)
                selected_count += 1
                selected_by_room[physical_room(item)] += 1
            selection[split][class_name] = {
                "database_candidate_frames": len(pool),
                "requested_frames": requested,
                "selected_frames": selected_count,
                "candidates_inspected": inspected,
                "candidates_skipped_without_pixels": skipped,
                "selected_frames_by_physical_room": dict(sorted(selected_by_room.items())),
            }
            skipped_without_pixels[split][class_name] = skipped
            if selected_count == 0:
                raise ValueError(f"Fold {spec['name']} {split}/{class_name} has no non-empty mask samples")

    ids_by_split = {split: set(rows) for split, rows in sampled.items()}
    if ids_by_split["train"] & ids_by_split["val"]:
        raise ValueError(f"Fold {spec['name']} has observation leakage between train and val")

    return {
        "spec": spec,
        "sampled": sampled,
        "selected_for": selected_for,
        "selection": selection,
        "skipped_without_pixels": skipped_without_pixels,
    }


def write_fold_preview(fold_root: Path, output_path: Path) -> None:
    tile_width, tile_height, header_height = 240, 320, 26
    rows = []
    for split in ("train", "val"):
        tiles = []
        for class_name in CLASSES:
            tile = np.full((tile_height + header_height, tile_width, 3), 245, dtype=np.uint8)
            cv2.putText(tile, f"{split} / {class_name}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            label_dir = fold_root / "labels" / split
            image_dir = fold_root / "images" / split
            candidate_labels = sorted(label_dir.glob("*.txt"), key=lambda path: int(path.stem))
            for label_path in candidate_labels:
                lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not any(int(line[0]) == CLASSES.index(class_name) for line in lines):
                    continue
                image_path = image_dir / f"{label_path.stem}.png"
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Could not read preview image: {image_path}")
                image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                height, width = image.shape[:2]
                for line in lines:
                    class_id, xc, yc, bw, bh = int(line[0]), *map(float, line[1:])
                    if CLASSES[class_id] != class_name:
                        continue
                    x1 = round((xc - bw / 2) * width)
                    y1 = round((yc - bh / 2) * height)
                    x2 = round((xc + bw / 2) * width)
                    y2 = round((yc + bh / 2) * height)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (30, 20, 230), 2)
                tile[header_height:] = image
                break
            tiles.append(tile)
        rows.append(cv2.hconcat(tiles))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.vconcat(rows)):
        raise OSError(f"Could not write fold preview: {output_path}")


def main() -> None:
    args = parse_args()
    if args.per_class_limit < 1:
        raise ValueError("--per-class-limit must be positive")
    database = args.raw_root / "rh.db"
    rgb_root = args.raw_root / "files" / "rgbd"
    if not database.is_file() or not rgb_root.is_dir():
        raise FileNotFoundError(f"Expected Robot@Home2 raw data under {args.raw_root}")

    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(ROOT):
        raise ValueError(f"All project outputs must remain under {ROOT}; got {output_root}")
    preview_prefix = args.preview_prefix
    if preview_prefix is None:
        suffix = "cv_pilot" if output_root.name == "yolo_cv_pilot" else output_root.name.removeprefix("yolo_")
        preview_prefix = f"robotathome2_{suffix}"
    if not preview_prefix or any(character in preview_prefix for character in "\\/:*?\"<>| "):
        raise ValueError("--preview-prefix must be a non-empty path-safe token without spaces")
    staging = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or staging.exists():
        raise FileExistsError(f"Output or staging path already exists; refusing to overwrite: {output_root} / {staging}")

    candidates = load_candidates(database)
    specs = fold_specs(candidates, args.test_home)
    preview_paths = {
        spec["name"]: ROOT / "outputs" / f"{preview_prefix}_{spec['name']}.jpg"
        for spec in specs
    }
    existing_previews = [path for path in preview_paths.values() if path.exists()]
    if existing_previews:
        raise FileExistsError(
            "Preview target already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in existing_previews)
        )
    prepared_cache: dict[int, dict] = {}
    folds = [
        select_fold(spec, candidates, prepared_cache, args.raw_root, args.per_class_limit, args.seed)
        for spec in specs
    ]

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    for fold in folds:
        spec = fold["spec"]
        fold_stage = staging / spec["name"]
        split_summaries = {}
        for split, rows in fold["sampled"].items():
            image_dir = fold_stage / "images" / split
            label_dir = fold_stage / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            instances: Counter = Counter()
            split_items = []
            for observation_id, item in sorted(rows.items()):
                record = prepared_cache[observation_id]
                instances.update(record["box_counts"])
                source_image = record["image"]
                shutil.copyfile(source_image, image_dir / f"{observation_id}.png")
                (label_dir / f"{observation_id}.txt").write_text(
                    "".join(
                        f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                        for class_id, xc, yc, bw, bh in record["boxes"]
                    ),
                    encoding="utf-8",
                )
                split_items.append({
                    "observation_id": observation_id,
                    "home": item["home"],
                    "home_session_id": item["session_id"],
                    "relative_room": item["new_path"],
                    "physical_room": physical_room(item),
                    "selected_for": sorted(fold["selected_for"][(split, observation_id)]),
                    "boxes": len(record["boxes"]),
                    "instances_by_class": dict(record["box_counts"]),
                })
            split_summaries[split] = {
                "homes": spec["train_homes"] if split == "train" else [spec["val_home"]],
                "image_count": len(split_items),
                "physical_rooms": sorted({item["physical_room"] for item in split_items}),
                "instances_by_class": dict(instances),
                "selection": fold["selection"][split],
                "items": split_items,
            }

        manifest = {
            "dataset": "Robot@Home2",
            "purpose": "house-grouped cross-validation pilot; no test split included",
            "fold": spec["name"],
            "seed": args.seed,
            "per_class_limit_per_split": args.per_class_limit,
            "classes": list(CLASSES),
            "train_homes": spec["train_homes"],
            "validation_home": spec["val_home"],
            "excluded_final_test_home": args.test_home,
            "split_homes": {
                "train": spec["train_homes"],
                "val": [spec["val_home"]],
            },
            "selection": fold["selection"],
            "candidates_skipped_without_pixels": fold["skipped_without_pixels"],
            "splits": split_summaries,
            "notes": [
                "Frames are selected by round-robin over physical rooms and acquisition groups, with deterministic random order.",
                "Only frames with a non-empty pixel mask for the class they were selected for count toward that class quota.",
                "Timestamp units are not verified; this version does not claim temporal de-duplication within an acquisition group.",
                "sarmis is excluded from all development folds and must be reserved for a final evaluation after model selection.",
            ],
        }
        fold_stage.mkdir(parents=True, exist_ok=True)
        names_inline = ", ".join(CLASSES)
        (fold_stage / "data.yaml").write_text(
            f'path: "{(output_root / spec["name"]).as_posix()}"\n'
            "train: images/train\nval: images/val\n"
            f"names: [{names_inline}]\n",
            encoding="utf-8",
        )
        (fold_stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    staging.rename(output_root)
    for fold in folds:
        spec = fold["spec"]
        preview = preview_paths[spec["name"]]
        write_fold_preview(output_root / spec["name"], preview)
        print(f"{spec['name']}: {output_root / spec['name']} | preview: {preview}")
        for split, summary in fold["sampled"].items():
            print(f"  {split}: {len(summary)} images; homes={spec['train_homes'] if split == 'train' else [spec['val_home']]}")
    print(f"All four folds exclude final test home {args.test_home!r}: {output_root}")


if __name__ == "__main__":
    main()
