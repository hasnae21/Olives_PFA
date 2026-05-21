"""
Prepare a maturity-classification dataset from a YOLO split dataset.

What it does:
- scans dataset/split/{train,valid,test}/images and labels
- keeps only samples whose label file contains exactly one annotation line
- extracts the class id from the YOLO label line
- writes one simplified label file per image containing only that class id
- writes a flat labels.txt file for compatibility with older training code

Output structure:
    output_root/
        images/
        labels/
        labels.txt
        labels_index.csv

The label file format is a single integer per image, for example:
    0
    2
    1
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def read_single_yolo_class(label_path: Path) -> Optional[int]:
    """Return the class id if the file contains exactly one valid annotation line."""
    if not label_path.exists():
        return None

    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        return None

    parts = lines[0].split()
    if not parts:
        return None

    try:
        return int(float(parts[0]))
    except ValueError:
        return None


def copy_selected_samples(split_root: Path, output_root: Path, copy_mode: str = "copy") -> Tuple[int, int]:
    """Copy images and create simplified label files.

    Returns:
        tuple: (number_of_kept_samples, number_of_skipped_samples)
    """
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    flat_labels = []

    for split_name in ("train", "valid", "test"):
        images_dir = split_root / split_name / "images"
        labels_dir = split_root / split_name / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            continue

        image_files = sorted([p for p in images_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS])

        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            class_id = read_single_yolo_class(label_path)

            if class_id is None:
                skipped += 1
                continue

            destination_image = output_images / image_path.name
            destination_label = output_labels / f"{image_path.stem}.txt"

            if copy_mode == "copy":
                shutil.copy2(image_path, destination_image)
            elif copy_mode == "move":
                shutil.move(str(image_path), str(destination_image))
            else:
                raise ValueError(f"Unsupported copy_mode: {copy_mode}")

            destination_label.write_text(f"{class_id}\n", encoding="utf-8")
            flat_labels.append((image_path.name, class_id))
            kept += 1

    flat_labels_path = output_root / "labels.txt"
    with flat_labels_path.open("w", encoding="utf-8") as f:
        for _, class_id in flat_labels:
            f.write(f"{class_id}\n")

    index_path = output_root / "labels_index.csv"
    with index_path.open("w", encoding="utf-8") as f:
        f.write("image,label\n")
        for image_name, class_id in flat_labels:
            f.write(f"{image_name},{class_id}\n")

    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a maturity dataset from a YOLO split dataset by keeping single-object samples only."
    )
    parser.add_argument(
        "--split-root",
        type=str,
        default=r"dataset/split",
        help="Path to the YOLO split folder containing train/valid/test.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"dataset/prepared/maturity_from_split",
        help="Output folder for the filtered maturity dataset.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "move"),
        default="copy",
        help="Copy images to the output folder or move them.",
    )
    args = parser.parse_args()

    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kept, skipped = copy_selected_samples(split_root, output_root, copy_mode=args.copy_mode)

    print("Done.")
    print(f"Kept samples: {kept}")
    print(f"Skipped samples: {skipped}")
    print(f"Images folder: {output_root / 'images'}")
    print(f"Per-image labels: {output_root / 'labels'}")
    print(f"Flat labels file: {output_root / 'labels.txt'}")
    print(f"CSV index: {output_root / 'labels_index.csv'}")


if __name__ == "__main__":
    main()