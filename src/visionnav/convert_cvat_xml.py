"""Convert a CVAT image-annotation XML export and matching images to YOLO labels."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="CVAT annotations.xml")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing the exact source images")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for paired images and YOLO labels")
    return parser.parse_args()


def load_annotations(path: Path) -> ET.Element:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            try:
                xml_bytes = archive.read("annotations.xml")
            except KeyError as error:
                raise ValueError(f"ZIP archive does not contain annotations.xml: {path}") from error
        return ET.fromstring(xml_bytes)
    return ET.parse(path).getroot()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new path: {args.output}")

    root = load_annotations(args.annotations)
    labels = [node.findtext("name") for node in root.findall("./meta/task/labels/label")]
    if not labels or any(label is None for label in labels):
        raise ValueError("Could not read the ordered class names from CVAT XML metadata")
    class_to_id = {label: index for index, label in enumerate(labels)}
    image_nodes = root.findall("image")
    if not image_nodes:
        raise ValueError("CVAT XML contains no image annotations")
    if not any(image_node.findall("box") for image_node in image_nodes):
        raise ValueError("CVAT XML contains images but no bounding-box annotations")

    image_out = args.output / "images"
    label_out = args.output / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    box_count = 0

    for image_node in image_nodes:
        name = image_node.get("name", "")
        if not name or Path(name).name != name:
            raise ValueError(f"Unsafe or missing image filename in XML: {name!r}")
        source = args.images / name
        if not source.is_file():
            raise FileNotFoundError(f"Matching source image not found: {source}")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Could not decode source image: {source}")
        height, width = image.shape[:2]
        declared_width = int(image_node.get("width", "0"))
        declared_height = int(image_node.get("height", "0"))
        if (width, height) != (declared_width, declared_height):
            raise ValueError(
                f"Image dimensions differ for {name}: actual {width}x{height}, "
                f"CVAT says {declared_width}x{declared_height}"
            )

        rows: list[str] = []
        for box in image_node.findall("box"):
            class_name = box.get("label")
            if class_name not in class_to_id:
                raise ValueError(f"Unknown class {class_name!r} in image {name}")
            x1, y1, x2, y2 = (
                float(box.get(key, "nan")) for key in ("xtl", "ytl", "xbr", "ybr")
            )
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid box coordinates in {name}: {(x1, y1, x2, y2)}")
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            rows.append(
                f"{class_to_id[class_name]} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )
        shutil.copy2(source, image_out / name)
        (label_out / f"{Path(name).stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        box_count += len(rows)

    class_list = "\n".join(f"{index}: {name}" for name, index in class_to_id.items())
    (args.output / "README.md").write_text(
        "# CVAT YOLO pilot export\n\n"
        f"Images: {len(image_nodes)}\n\nBounding boxes: {box_count}\n\n"
        "Class IDs (preserved from CVAT label order):\n\n"
        f"```text\n{class_list}\n```\n\n"
        "This is a small annotation pilot, not a train/validation/test split. "
        "Review labels and add more scenes before training.\n",
        encoding="utf-8",
    )
    print(f"Images copied: {len(image_nodes)}")
    print(f"YOLO boxes written: {box_count}")
    print(f"Classes: {class_to_id}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
