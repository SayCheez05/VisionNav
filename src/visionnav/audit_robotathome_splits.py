"""Audit Robot@Home2 class coverage and house-grouped validation options."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path, PurePosixPath

from visionnav.prepare_robotathome_yolo import (
    CLASSES,
    DEFAULT_OUTPUT,
    RAW_ROOT,
    ROOT,
    load_candidates,
    prepare_candidate,
)


DEFAULT_MANIFEST = DEFAULT_OUTPUT / "manifest.json"
DEFAULT_REPORT = ROOT / "reports" / "robotathome2_split_audit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-per-class-home", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--test-home", default="sarmis")
    return parser.parse_args()


def physical_room(path_text: str) -> str:
    parts = PurePosixPath(path_text).parts
    if len(parts) >= 3:
        return "/".join(parts[1:3])  # home/room; session and subsession are separate acquisition groups.
    return path_text


def database_coverage(candidates: dict[int, dict], homes: set[str]) -> dict[str, dict]:
    result = {}
    for class_name in CLASSES:
        rows = [
            item for item in candidates.values()
            if item["home"] in homes and class_name in item["classes"]
        ]
        rooms = sorted({physical_room(item["new_path"]) for item in rows})
        result[class_name] = {
            "database_candidate_frames": len(rows),
            "physical_rooms": rooms,
            "physical_room_count": len(rooms),
        }
    return result


def audit_mask_samples(
    candidates: dict[int, dict],
    raw_root: Path,
    sample_limit: int,
    seed: int,
) -> dict[str, dict[str, dict]]:
    homes = sorted({item["home"] for item in candidates.values()})
    prepared_cache: dict[int, dict] = {}
    audit = defaultdict(dict)
    database = raw_root / "rh.db"

    for home in homes:
        for class_name in CLASSES:
            pool = sorted(
                (
                    item for item in candidates.values()
                    if item["home"] == home and class_name in item["classes"]
                ),
                key=lambda item: item["id"],
            )
            stable_seed = seed + sum(ord(char) for char in f"{home}|{class_name}")
            chosen = random.Random(stable_seed).sample(pool, min(sample_limit, len(pool)))
            valid_frames = 0
            valid_instances = 0
            sampled_rooms = set()
            for item in chosen:
                if item["id"] not in prepared_cache:
                    prepared_cache[item["id"]] = prepare_candidate(item, raw_root, database)
                prepared = prepared_cache[item["id"]]
                valid_frames += class_name in prepared["target_names"]
                valid_instances += prepared["box_counts"].get(class_name, 0)
                sampled_rooms.add(physical_room(item["new_path"]))
            audit[home][class_name] = {
                "database_candidate_frames": len(pool),
                "physical_room_count": len({physical_room(item["new_path"]) for item in pool}),
                "sampled_frames": len(chosen),
                "mask_valid_frames": valid_frames,
                "mask_valid_rate": valid_frames / len(chosen) if chosen else 0.0,
                "valid_instances_in_sample": valid_instances,
                "sampled_physical_rooms": sorted(sampled_rooms),
                "sampled_observation_ids": [item["id"] for item in chosen],
            }
    return dict(audit)


def grouped_folds(candidates: dict[int, dict], test_home: str) -> list[dict]:
    all_homes = {item["home"] for item in candidates.values()}
    if test_home not in all_homes:
        raise ValueError(f"Requested final test home {test_home!r} is not in Robot@Home2 labels: {sorted(all_homes)}")
    development_homes = sorted(all_homes - {test_home})
    if len(development_homes) < 2:
        raise ValueError("At least two development homes are required for grouped validation")

    folds = []
    for val_home in development_homes:
        train_homes = set(development_homes) - {val_home}
        folds.append({
            "validation_home": val_home,
            "train_homes": sorted(train_homes),
            "train": database_coverage(candidates, train_homes),
            "val": database_coverage(candidates, {val_home}),
        })
    return folds


def timestamp_metadata(database: Path) -> dict:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(rh_lblrgbd)")}
        if "time_stamp" not in columns:
            return {"column_present": False}
        count, non_null, distinct, minimum, maximum, storage_type = connection.execute(
            "SELECT COUNT(*), COUNT(time_stamp), COUNT(DISTINCT time_stamp), "
            "MIN(time_stamp), MAX(time_stamp), typeof(MIN(time_stamp)) FROM rh_lblrgbd"
        ).fetchone()
    return {
        "column_present": True,
        "rows": count,
        "non_null": non_null,
        "distinct_values": distinct,
        "minimum": minimum,
        "maximum": maximum,
        "storage_type": storage_type,
        "unit_and_frame_spacing_verified": False,
    }


def pilot_summary(manifest: dict) -> dict:
    result = {}
    for split, entry in manifest["splits"].items():
        items = entry["items"]
        selection = manifest["selection"][split]
        result[split] = {
            "image_count": entry["image_count"],
            "homes": entry["homes"],
            "distinct_acquisition_groups": len({item["relative_room"] for item in items}),
            "distinct_physical_rooms": len({physical_room(item["relative_room"]) for item in items}),
            "classes": {},
        }
        for class_name in CLASSES:
            row = selection[class_name]
            result[split]["classes"][class_name] = {
                **row,
                "database_candidates_without_nonempty_mask_seen": row["candidates_inspected"] - row["selected_frames"],
            }
    return result


def frame_counts(coverage: dict) -> str:
    return ", ".join(f"{class_name} {coverage[class_name]['database_candidate_frames']}" for class_name in CLASSES)


def room_counts(coverage: dict) -> str:
    return ", ".join(f"{class_name} {coverage[class_name]['physical_room_count']}" for class_name in CLASSES)


def build_report(data: dict) -> str:
    lines = [
        "# Robot@Home2 数据覆盖与分组验证审计",
        "",
        f"- 审计日期：{data['audit_date']}",
        f"- 目标类别：{', '.join(data['classes'])}",
        f"- 像素掩码分层抽查：每个住宅/类别最多 {data['mask_sample_limit_per_home_class']} 帧，seed={data['seed']}；此项是抽样估计，不是全量掩码扫描。",
        "- 计数来源：SQLite 类别标注帧和路径元数据；‘物理房间’按 `home/room` 归并不同 session 与 subsession。",
        "- 数据库 `time_stamp` 存为整数；本次尚未校准其单位/相邻帧间隔，因此未用该值做抽帧。",
        "",
        "## 现有 pilot 覆盖",
        "",
        "| split | 图像 | 住宅 | acquisition groups | 物理房间 | 每类抽样检查（候选/目标帧/检到空掩码） |",
        "|---|---:|---|---:|---:|---|",
    ]
    for split, summary in data["current_pilot"].items():
        class_stats = []
        for class_name in data["classes"]:
            stat = summary["classes"][class_name]
            class_stats.append(
                f"{class_name} {stat['database_candidate_frames']}/{stat['selected_frames']}/"
                f"{stat['database_candidates_without_nonempty_mask_seen']}"
            )
        lines.append(
            f"| {split} | {summary['image_count']} | {', '.join(summary['homes'])} | "
            f"{summary['distinct_acquisition_groups']} | {summary['distinct_physical_rooms']} | "
            f"{'; '.join(class_stats)} |"
        )

    lines.extend([
        "",
        "空掩码列只统计转换器在达到抽样目标前实际检查过的候选，不是整库空掩码总量。val 当前全部来自 rx2；`chair` 的候选仅分布在一个物理房间。",
        "",
        "## 各住宅候选量与像素掩码抽查",
        "",
        "`DB候选` 是 SQLite 声称含该类别的不同观察帧数；`掩码有效样本` 是固定随机种子抽取的最多 30 帧中，实际出现非空目标像素掩码的数量。",
        "",
        "| 住宅 | 类别 | DB候选帧 | 物理房间 | 掩码有效样本 | 有效率 | 抽样覆盖房间 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for home, classes in data["home_class_audit"].items():
        for class_name in data["classes"]:
            stat = classes[class_name]
            lines.append(
                f"| {home} | {class_name} | {stat['database_candidate_frames']} | "
                f"{stat['physical_room_count']} | {stat['mask_valid_frames']}/{stat['sampled_frames']} | "
                f"{stat['mask_valid_rate']:.0%} | {', '.join(stat['sampled_physical_rooms'])} |"
            )

    lines.extend([
        "",
        "## 保留 sarmis 为最终 test 的开发期留一住宅验证",
        "",
        "以下每折仅用三个住宅训练、另一个开发住宅验证；sarmis 不参与折内抽样、训练或调参。帧数是 DB 候选量，真实可用数量须以像素掩码过滤后的清单为准。",
        "",
        "| 验证住宅 | 训练住宅 | train 候选帧（bottle/chair/table） | val 候选帧（bottle/chair/table） | val 物理房间数（b/c/t） |",
        "|---|---|---|---|---|",
    ])
    for fold in data["grouped_validation_folds"]:
        rooms = ", ".join(str(fold["val"][name]["physical_room_count"]) for name in data["classes"])
        lines.append(
            f"| {fold['validation_home']} | {', '.join(fold['train_homes'])} | "
            f"{frame_counts(fold['train'])} | {frame_counts(fold['val'])} | {rooms} |"
        )

    lines.extend([
        "",
        "## 建议",
        "",
        "1. 开发阶段采用按住宅分组的 4 折留一验证（alma、anto、pare、rx2 依次留作 val），聚合多个住宅的验证表现；每折都按实际非空像素掩码重新筛选样本。",
        "2. 固定 sarmis 为最终测试住宅，在模型和超参数定稿前不读取其预测指标。",
        "3. 抽样按类别与物理房间分层；先校准 `time_stamp` 单位/帧间隔，或使用图像近重复检测，再限制同一采集组内的相近帧。检查少量叠框图后再增加训练帧数。单纯增加 epoch 无法弥补场景覆盖不足。",
        "4. 对 rx2 chair 等候选空掩码偏多的类别，按实际 mask-valid 样本计量，不直接把 SQLite 候选数当作可训练样本数。",
        "",
        "本报告只审计数据覆盖、分组候选和掩码抽样；没有训练模型，也没有生成或读取 Robot@Home2 test 预测。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.sample_per_class_home < 1:
        raise ValueError("--sample-per-class-home must be positive")
    database = args.raw_root / "rh.db"
    if not database.is_file():
        raise FileNotFoundError(f"Robot@Home2 database not found: {database}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Pilot manifest not found: {args.manifest}")

    candidates = load_candidates(database)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    home_audit = audit_mask_samples(candidates, args.raw_root, args.sample_per_class_home, args.seed)
    folds = grouped_folds(candidates, args.test_home)
    data = {
        "audit_date": date.today().isoformat(),
        "classes": list(CLASSES),
        "mask_sample_limit_per_home_class": args.sample_per_class_home,
        "seed": args.seed,
        "test_home": args.test_home,
        "development_homes": sorted({item["home"] for item in candidates.values()} - {args.test_home}),
        "current_pilot": pilot_summary(manifest),
        "home_class_audit": home_audit,
        "grouped_validation_folds": folds,
        "timestamp_metadata": timestamp_metadata(database),
    }
    report = build_report(data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.report.with_suffix(".json")
    args.report.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Report: {args.report.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    for home, classes in home_audit.items():
        summary = ", ".join(
            f"{name}={classes[name]['mask_valid_frames']}/{classes[name]['sampled_frames']}"
            for name in CLASSES
        )
        print(f"{home}: mask-valid sampled frames {summary}")
    print(f"Development homes: {', '.join(data['development_homes'])}; final test home: {args.test_home}")
    print(f"Grouped validation folds: {len(folds)}")


if __name__ == "__main__":
    main()
