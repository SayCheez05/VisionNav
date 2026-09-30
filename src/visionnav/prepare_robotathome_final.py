"""Merge the four audited Robot@Home2 development folds without duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "datasets" / "RobotAtHome2" / "processed" / "yolo_cv_candidate_v2"
DEFAULT_OUTPUT = ROOT / "data" / "datasets" / "RobotAtHome2" / "processed" / "yolo_final_v2_dev"
FOLDS = ("val_alma", "val_anto", "val_pare", "val_rx2")
HOMES = ("alma", "anto", "pare", "rx2")
CLASSES = ("bottle", "chair", "table")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_labels(path: Path) -> tuple[int, dict[str, int]]:
    counts: Counter[str] = Counter()
    line_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected 5 YOLO fields at {path}:{line_number}")
        try:
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}") from exc
        if class_id not in range(len(CLASSES)) or not all(math.isfinite(value) for value in box):
            raise ValueError(f"Invalid class or non-finite box at {path}:{line_number}")
        xc, yc, width, height = box
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Out-of-range YOLO box at {path}:{line_number}: {box}")
        counts[CLASSES[class_id]] += 1
        line_count += 1
    if line_count == 0:
        raise ValueError(f"Empty label file is not expected in the selected positive-frame dataset: {path}")
    return line_count, dict(counts)


def read_and_audit(source_root: Path) -> tuple[dict[str, dict], dict]:
    if not source_root.is_relative_to(ROOT):
        raise ValueError(f"Source data must remain under project root: {source_root}")
    observations: dict[str, dict] = {}
    source_manifest_hashes: dict[str, str] = {}
    fold_occurrences: Counter[str] = Counter()

    for fold_name in FOLDS:
        fold_root = source_root / fold_name
        manifest_path = fold_root / "manifest.json"
        data_path = fold_root / "data.yaml"
        if not manifest_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(f"Missing candidate-v2 fold manifest/config: {fold_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fold") != fold_name or manifest.get("classes") != list(CLASSES):
            raise ValueError(f"Fold identity or class order mismatch in {manifest_path}")
        if manifest.get("excluded_final_test_home") != "sarmis":
            raise ValueError(f"Fold does not explicitly exclude sarmis: {manifest_path}")
        split_records = manifest.get("splits", {})
        if set(split_records) != {"train", "val"}:
            raise ValueError(f"Expected only train/val in {manifest_path}")
        config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or "test" in config or config.get("names") != list(CLASSES):
            raise ValueError(f"Unsafe or inconsistent fold data.yaml: {data_path}")
        declared_root = Path(config.get("path", "")).resolve()
        if declared_root != fold_root.resolve():
            raise ValueError(f"Fold data.yaml path escapes/mismatches its fold: {data_path}")
        if config.get("train") != "images/train" or config.get("val") != "images/val":
            raise ValueError(f"Unexpected split paths in {data_path}")

        source_manifest_hashes[fold_name] = sha256(manifest_path)
        for split_name in ("train", "val"):
            split = split_records[split_name]
            if split.get("image_count") != len(split.get("items", [])):
                raise ValueError(f"Declared image count does not match items in {manifest_path}/{split_name}")
            observed_homes = set()
            for row in split["items"]:
                observation_id = str(row.get("observation_id", ""))
                if not observation_id.isdigit():
                    raise ValueError(f"Invalid observation ID in {manifest_path}: {observation_id!r}")
                home = row.get("home")
                if home == "sarmis" or home not in HOMES:
                    raise ValueError(f"Unexpected or holdout home in development fold: {home!r}")
                observed_homes.add(home)
                image_path = fold_root / "images" / split_name / f"{observation_id}.png"
                label_path = fold_root / "labels" / split_name / f"{observation_id}.txt"
                if not image_path.is_file() or not label_path.is_file():
                    raise FileNotFoundError(f"Missing image/label pair for observation {observation_id} in {fold_root}")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"Invalid RGB image: {image_path}")
                label_count, class_counts = validate_labels(label_path)
                source_counts = row.get("instances_by_class", {})
                expected_counts = {
                    name: int(source_counts[name])
                    for name in CLASSES
                    if int(source_counts.get(name, 0)) > 0
                }
                unexpected_source_classes = {
                    name for name in source_counts
                    if name not in CLASSES and not (name.startswith("empty_") and name[6:] in CLASSES)
                }
                if unexpected_source_classes:
                    raise ValueError(f"Unexpected source-mask class diagnostics for {observation_id}: {unexpected_source_classes}")
                if label_count != row.get("boxes") or class_counts != expected_counts:
                    raise ValueError(f"Label contents disagree with manifest for observation {observation_id}")
                current = {
                    "observation_id": int(observation_id),
                    "home": home,
                    "home_session_id": row.get("home_session_id"),
                    "relative_room": row.get("relative_room"),
                    "physical_room": row.get("physical_room"),
                    "boxes": label_count,
                    "instances_by_class": class_counts,
                    "source_manifest_instance_counts": source_counts,
                    "selected_for": sorted(set(row.get("selected_for", []))),
                    "image_sha256": sha256(image_path),
                    "label_sha256": sha256(label_path),
                    "source_image": image_path,
                    "source_label": label_path,
                    "source_occurrences": [],
                }
                existing = observations.get(observation_id)
                if existing:
                    identity_fields = (
                        "home", "home_session_id", "relative_room", "physical_room", "boxes",
                        "instances_by_class", "source_manifest_instance_counts", "image_sha256", "label_sha256",
                    )
                    if any(existing[field] != current[field] for field in identity_fields):
                        raise ValueError(f"Cross-fold observation conflict for ID {observation_id}")
                    existing["selected_for"] = sorted(set(existing["selected_for"]) | set(current["selected_for"]))
                    existing["source_occurrences"].append(f"{fold_name}/{split_name}")
                else:
                    current["source_occurrences"].append(f"{fold_name}/{split_name}")
                    observations[observation_id] = current
                fold_occurrences[split_name] += 1
            declared_homes = set(split.get("homes", []))
            if not observed_homes <= declared_homes or "sarmis" in declared_homes:
                raise ValueError(f"Split home metadata mismatch or sarmis leak in {manifest_path}/{split_name}")

    found_homes = {item["home"] for item in observations.values()}
    if found_homes != set(HOMES):
        raise ValueError(f"Expected all four development homes {HOMES}, found {sorted(found_homes)}")
    total_instances: Counter[str] = Counter()
    home_counts: Counter[str] = Counter()
    for item in observations.values():
        total_instances.update(item["instances_by_class"])
        home_counts[item["home"]] += 1
    audit = {
        "source_root": str(source_root),
        "folds": list(FOLDS),
        "classes": list(CLASSES),
        "excluded_home": "sarmis",
        "fold_split_occurrences": dict(fold_occurrences),
        "unique_observations": len(observations),
        "deduplicated_occurrences": sum(fold_occurrences.values()) - len(observations),
        "unique_images_by_home": dict(sorted(home_counts.items())),
        "unique_instances_by_class": {name: total_instances[name] for name in CLASSES},
        "source_manifest_sha256": source_manifest_hashes,
        "duplicate_consistency": "all repeated image and label SHA-256 values and source metadata match",
    }
    return observations, audit


def export_dataset(observations: dict[str, dict], audit: dict, output_root: Path) -> dict:
    if not output_root.is_relative_to(ROOT):
        raise ValueError(f"Output must remain under project root: {output_root}")
    staging = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or staging.exists():
        raise FileExistsError(f"Output/staging already exists; refusing to overwrite: {output_root} / {staging}")

    try:
        (staging / "images" / "train").mkdir(parents=True)
        (staging / "labels" / "train").mkdir(parents=True)
        materialized = []
        for observation_id in sorted(observations, key=int):
            item = observations[observation_id]
            filename = f"{observation_id}.png"
            shutil.copyfile(item["source_image"], staging / "images" / "train" / filename)
            shutil.copyfile(item["source_label"], staging / "labels" / "train" / f"{observation_id}.txt")
            materialized.append({key: item[key] for key in (
                "observation_id", "home", "home_session_id", "relative_room", "physical_room",
                "boxes", "instances_by_class", "source_manifest_instance_counts", "selected_for",
                "image_sha256", "label_sha256", "source_occurrences",
            )})
        output_manifest = {
            "dataset": "Robot@Home2",
            "purpose": "final monocular RGB training set from candidate-v2 development-home union; no holdout/test samples",
            "classes": list(CLASSES),
            "split_policy": "all unique candidate-v2 observations from alma/anto/pare/rx2; merge train and val appearances, deduplicated by observation_id",
            "in_sample_validation_note": "Ultralytics requires a val path for final bookkeeping; data.yaml val points to images/train. The resulting in-sample metrics are diagnostic only, never used as a performance estimate.",
            "sarmis_included": False,
            "audit": audit,
            "items": materialized,
        }
        (staging / "data.yaml").write_text(
            f'path: "{output_root.as_posix()}"\n'
            "train: images/train\nval: images/train\n"
            f"names: [{', '.join(CLASSES)}]\n",
            encoding="utf-8",
        )
        (staging / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        staging.rename(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Audit all folds without creating the merged dataset")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    observations, audit = read_and_audit(source_root)
    if args.dry_run:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    result = export_dataset(observations, audit, output_root)
    print(json.dumps({key: result[key] for key in ("purpose", "classes", "sarmis_included", "audit")}, indent=2, ensure_ascii=False))
    print(f"Merged development data: {output_root}")


if __name__ == "__main__":
    main()
