"""
Minimal UNet workflow for olive segmentation.

Keeps the project small:
- one script
- one weights file
- optional prediction masks only

Train:
    python scripts/unet_minimal.py train --images dataset/prepared/segmentation/images --masks dataset/prepared/segmentation/masks --out models/unet.pth

Infer:
    python scripts/unet_minimal.py infer --weights models/unet.pth --image path/to/image.jpg --out output_mask.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# CONFIGURATION - Modify these variables to change behavior
# ============================================================================

# Mode: Set this if you want to run train or infer directly without command line args
# Options: "train", "infer", or None (use command line args)
AUTO_MODE = "infer"

# ============ TRAIN Configuration ============
TRAIN_IMAGES_DIR = "dataset/prepared/segmentation/images"
TRAIN_MASKS_DIR = "dataset/prepared/segmentation/masks"
TRAIN_OUTPUT_MODEL = "models/unet.pth"
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 8
TRAIN_IMAGE_SIZE = 256
TRAIN_LEARNING_RATE = 1e-3

# ============ INFER Configuration ============
INFER_WEIGHTS_PATH = "models/unet.pth"
INFER_INPUT_IMAGE = "test_preview.jpeg"  # Change this to your image path
INFER_OUTPUT_MASK = "mask.png"
INFER_THRESHOLD = 0.5
INFER_MIN_CONTOUR_AREA = 100

# ============================================================================


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)
        x = self.pool1(s1)
        s2 = self.down2(x)
        x = self.pool2(s2)
        x = self.bridge(x)
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.conv2(x)
        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(self.out(x))


class SegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, size: int = 256):
        self.images = sorted([p for p in Path(images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.masks_dir = Path(masks_dir)
        self.size = size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size)).astype(np.float32) / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size)).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten()
    target = target.flatten()
    inter = (pred * target).sum()
    return 1 - ((2 * inter + smooth) / (pred.sum() + target.sum() + smooth))


def extract_contours(mask: np.ndarray, min_area: int = 100):
    """Extract filtered contours from a binary mask."""
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def draw_contours(image: np.ndarray, contours):
    """Draw contours on a copy of the image."""
    out = image.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def train(images: str = None, masks: str = None, out: str = None, size: int = None, epochs: int = None, batch: int = None):
    # Use config if not provided
    images = images or TRAIN_IMAGES_DIR
    masks = masks or TRAIN_MASKS_DIR
    out = out or TRAIN_OUTPUT_MODEL
    size = size or TRAIN_IMAGE_SIZE
    epochs = epochs or TRAIN_EPOCHS
    batch = batch or TRAIN_BATCH_SIZE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SegDataset(images, masks, size=size)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model = TinyUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LEARNING_RATE)
    bce = nn.BCELoss()

    best = float("inf")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    for e in range(epochs):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            p = model(x)
            loss = 0.5 * bce(p, y) + 0.5 * dice_loss(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / max(1, len(dl))
        print(f"epoch {e+1}/{epochs} loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict(), "size": size}, out)
            print(f"saved: {out}")


def infer(weights: str = None, image_path: str = None, out: str = None):
    # Use config if not provided
    weights = weights or INFER_WEIGHTS_PATH
    image_path = image_path or INFER_INPUT_IMAGE
    out = out or INFER_OUTPUT_MASK
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(weights, map_location=device)
    size = int(ckpt.get("size", TRAIN_IMAGE_SIZE))
    model = TinyUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()

    mask = (pred > INFER_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    cv2.imwrite(out, mask)
    print(f"saved: {out}")

    contours = extract_contours(mask, min_area=INFER_MIN_CONTOUR_AREA)
    contour_img = draw_contours(img, contours)
    contour_out = str(Path(out).with_name(Path(out).stem + "_contours.png"))
    cv2.imwrite(contour_out, contour_img)
    print(f"saved: {contour_out}")

    print(f"contours detected: {len(contours)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--images", default=TRAIN_IMAGES_DIR, help=f"Images directory (default: {TRAIN_IMAGES_DIR})")
    t.add_argument("--masks", default=TRAIN_MASKS_DIR, help=f"Masks directory (default: {TRAIN_MASKS_DIR})")
    t.add_argument("--out", default=TRAIN_OUTPUT_MODEL, help=f"Output model path (default: {TRAIN_OUTPUT_MODEL})")
    t.add_argument("--size", type=int, default=TRAIN_IMAGE_SIZE, help=f"Image size (default: {TRAIN_IMAGE_SIZE})")
    t.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help=f"Number of epochs (default: {TRAIN_EPOCHS})")
    t.add_argument("--batch", type=int, default=TRAIN_BATCH_SIZE, help=f"Batch size (default: {TRAIN_BATCH_SIZE})")

    i = sub.add_parser("infer")
    i.add_argument("--weights", default=INFER_WEIGHTS_PATH, help=f"Model weights path (default: {INFER_WEIGHTS_PATH})")
    i.add_argument("--image", default=INFER_INPUT_IMAGE, help=f"Input image path (default: {INFER_INPUT_IMAGE})")
    i.add_argument("--out", default=INFER_OUTPUT_MASK, help=f"Output mask path (default: {INFER_OUTPUT_MASK})")

    args = p.parse_args()
    # If a subcommand was provided, honor CLI regardless of AUTO_MODE
    if args.cmd is not None:
        if args.cmd == "train":
            train(args.images, args.masks, args.out, size=args.size, epochs=args.epochs, batch=args.batch)
        else:
            infer(args.weights, args.image, args.out)
        return

    # No subcommand: fall back to AUTO_MODE if set
    if AUTO_MODE:
        if AUTO_MODE == "train":
            train()
        elif AUTO_MODE == "infer":
            infer()
        else:
            p.print_help()
        return

    # Nothing to do
    p.print_help()
    return


if __name__ == "__main__":
    main()