from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

BASE_DIR = Path("runs") / "detect"
TRAIN_RANGE = range(7, 25)
OUTPUT_ALL = Path("train_comparison_train7_to_train24.csv")
OUTPUT_640 = Path("train_comparison_640.csv")
OUTPUT_1280 = Path("train_comparison_1280.csv")

# Small manual correction requested by the user.
IMG_SIZE_OVERRIDE = {
    "train8": 1280,
}

METRICS = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]


def read_imgsz(train_dir: Path) -> int | None:
    args_path = train_dir / "args.yaml"
    if not args_path.exists():
        return None

    with args_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)

    if train_dir.name in IMG_SIZE_OVERRIDE:
        return int(IMG_SIZE_OVERRIDE[train_dir.name])

    value = data.get("imgsz")
    if value is None:
        return None
    return int(value)


def summarize_results(train_dir: Path) -> dict[str, Any] | None:
    csv_path = train_dir / "results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required = ["epoch", "metrics/mAP50-95(B)"]
    if any(column not in df.columns for column in required):
        return None

    df = df.copy()
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=["metrics/mAP50-95(B)"])
    if df.empty:
        return None

    best_idx = df["metrics/mAP50-95(B)"].astype(float).idxmax()
    best_row = df.loc[best_idx]
    args_path = train_dir / "args.yaml"
    model_name = train_dir.name
    if args_path.exists():
        with args_path.open("r", encoding="utf-8") as handle:
            args_data: dict[str, Any] = yaml.safe_load(handle)
        model_value = args_data.get("model")
        if isinstance(model_value, str) and model_value:
            model_name = Path(model_value).stem

    summary: dict[str, Any] = {
        "model": model_name,
        "run": train_dir.name,
        "imgsz": read_imgsz(train_dir),
        "imgsz_group": read_imgsz(train_dir),
        "best_epoch": int(best_row["epoch"]),
    }

    for metric in METRICS[1:]:
        if metric in df.columns:
            summary[metric] = float(best_row[metric])
        else:
            summary[metric] = pd.NA

    summary["source_csv"] = str(csv_path.as_posix())
    return summary


def main() -> None:
    rows: list[dict[str, Any]] = []

    for train_number in TRAIN_RANGE:
        train_dir = BASE_DIR / f"train{train_number}"
        summary = summarize_results(train_dir)
        if summary is None:
            print(f"Skipped {train_dir}: missing or invalid results.csv")
            continue

        rows.append(summary)

    if not rows:
        raise SystemExit("No training summaries were generated.")

    df = pd.DataFrame(rows)
    df["train_number"] = df["run"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(by=["imgsz_group", "train_number", "model"]).drop(columns=["train_number"])
    df.to_csv(OUTPUT_ALL, index=False)

    df_640 = df[df["imgsz_group"] == 640]
    df_1280 = df[df["imgsz_group"] == 1280]
    df_640.to_csv(OUTPUT_640, index=False)
    df_1280.to_csv(OUTPUT_1280, index=False)

    print(f"Saved: {OUTPUT_ALL}")
    print(f"Saved: {OUTPUT_640}")
    print(f"Saved: {OUTPUT_1280}")
    print("Included models:", ", ".join(df["model"].tolist()))


if __name__ == "__main__":
    main()
