from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("/home/wayay/Documents/Olives_Pfa/Olives_PFA/runs/detect/train16/results.csv")
OUTPUT_PATH = Path("/home/wayay/Documents/Olives_Pfa/Olives_PFA/runs/detect/train16/training_results_1920x1080.png")

TOP_METRICS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
]

BOTTOM_METRICS = [
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    fig, axes = plt.subplots(2, 5, figsize=(1920 / 100, 1080 / 100), dpi=100)

    for idx, metric in enumerate(TOP_METRICS):
        ax = axes[0, idx]
        ax.plot(df["epoch"], df[metric], marker="o", linewidth=2, markersize=4)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")

    for idx, metric in enumerate(BOTTOM_METRICS):
        ax = axes[1, idx]
        ax.plot(df["epoch"], df[metric], marker="o", linewidth=2, markersize=4)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches="tight")
    print(f"Image saved to: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
