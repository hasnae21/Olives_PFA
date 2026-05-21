"""
Crop olive regions from UNet masks and build a CNN-ready dataset.

Input layout:
    images_dir/
        image1.jpg
        image2.jpg
    masks_dir/
        image1_mask.png
        image2_mask.png
    labels_dir/
        image1.txt
        image2.txt

Output layout:
    output_root/
        images/
        labels/
        crops_index.csv

Each cropped image gets the same class label as its source image.
This is useful when each source image contains a single olive.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def find_image_files(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS])


def read_class_label(label_path: Path) -> Optional[int]:
    if not label_path.exists():
        return None

    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None

    first = lines[0].split()
    if not first:
        return None

    try:
        return int(float(first[0]))
    except ValueError:
        return None


def find_mask_path(masks_dir: Path, image_stem: str) -> Optional[Path]:
    candidates = [
        masks_dir / f"{image_stem}_mask.png",
        masks_dir / f"{image_stem}.png",
        masks_dir / f"{image_stem}_mask.jpg",
        masks_dir / f"{image_stem}.jpg",
        masks_dir / f"{image_stem}_mask.jpeg",
        masks_dir / f"{image_stem}.jpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_contours(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    if mask is None:
        return []

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = np.asarray(mask)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def clip_box(x: int, y: int, w: int, h: int, width: int, height: int, pad: int) -> Tuple[int, int, int, int]:
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return x1, y1, x2, y2


def masked_crop(image: np.ndarray, mask: np.ndarray, contour: np.ndarray, padding: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = cv2.boundingRect(contour)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_box(x, y, w, h, width, height, padding)

    crop_img = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()
    if crop_mask.ndim == 3:
        crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

    if crop_mask.max() <= 1:
        crop_mask = (crop_mask * 255).astype(np.uint8)
    else:
        crop_mask = crop_mask.astype(np.uint8)

    crop_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)
    return crop_img, (x1, y1, x2, y2)


def build_crops(
    images_dir: Path,
    masks_dir: Path,
    labels_dir: Path,
    output_root: Path,
    min_area: int = 100,
    padding: int = 8,
) -> tuple[int, int]:
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "crops_index.csv"
    kept = 0
    skipped = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_image", "crop_image", "label", "x1", "y1", "x2", "y2"])

        for image_path in find_image_files(images_dir):
            mask_path = find_mask_path(masks_dir, image_path.stem)
            label_path = labels_dir / f"{image_path.stem}.txt"

            if mask_path is None or not label_path.exists():
                skipped += 1
                continue

            class_id = read_class_label(label_path)
            if class_id is None:
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                skipped += 1
                continue

            contours = extract_contours(mask, min_area=min_area)
            if not contours:
                skipped += 1
                continue

            for idx, contour in enumerate(contours, start=1):
                crop_img, (x1, y1, x2, y2) = masked_crop(image, mask, contour, padding=padding)
                if crop_img.size == 0:
                    continue

                crop_name = f"{image_path.stem}_crop{idx:02d}.jpg"
                crop_label = f"{image_path.stem}_crop{idx:02d}.txt"

                cv2.imwrite(str(output_images / crop_name), crop_img)
                (output_labels / crop_label).write_text(f"{class_id}\n", encoding="utf-8")
                writer.writerow([image_path.name, crop_name, class_id, x1, y1, x2, y2])
                kept += 1

    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop olive regions from UNet masks for CNN training.")
    parser.add_argument("--images", type=str, default=r"dataset/prepared/maturity/images", help="Source images folder.")
    parser.add_argument("--masks", type=str, default=r"runs/unet_masks", help="Folder with UNet masks.")
    parser.add_argument("--labels-dir", type=str, default=r"dataset/prepared/maturity/labels", help="Per-image label files.")
    parser.add_argument("--output-root", type=str, default=r"dataset/prepared/maturity_crops", help="Output folder for crops.")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum contour area to keep.")
    parser.add_argument("--padding", type=int, default=8, help="Padding added around each crop.")
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    labels_dir = Path(args.labels_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kept, skipped = build_crops(
        images_dir=images_dir,
        masks_dir=masks_dir,
        labels_dir=labels_dir,
        output_root=output_root,
        min_area=args.min_area,
        padding=args.padding,
    )

    print("Done.")
    print(f"Crops saved: {kept}")
    print(f"Images skipped: {skipped}")
    print(f"Crops images folder: {output_root / 'images'}")
    print(f"Crops labels folder: {output_root / 'labels'}")
    print(f"Index file: {output_root / 'crops_index.csv'}")


if __name__ == "__main__":
    main()