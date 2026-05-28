import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Metrics layout (same as plot_results.py)
top_metrics = [
    'train/box_loss',
    'train/cls_loss',
    'train/dfl_loss',
    'metrics/precision(B)',
    'metrics/recall(B)'
]

bottom_metrics = [
    'val/box_loss',
    'val/cls_loss',
    'val/dfl_loss',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)'
]

base = Path('runs') / 'detect'
if not base.exists():
    print(f"Path not found: {base.resolve()}")
    raise SystemExit(1)

found = 0
for train_dir in sorted(base.iterdir()):
    if not train_dir.is_dir():
        continue
    csv_path = train_dir / 'results.csv'
    if not csv_path.exists():
        continue

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        continue

    # sanitize
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ensure epoch exists
    if 'epoch' not in df.columns:
        print(f"No 'epoch' column in {csv_path}, skipping")
        continue

    # create figure
    fig, axes = plt.subplots(2, 5, figsize=(1920/100, 1080/100), dpi=100)

    # plot top
    for idx, metric in enumerate(top_metrics):
        ax = axes[0, idx]
        if metric in df.columns:
            ax.plot(df['epoch'], df[metric], marker='o', linewidth=2, markersize=4)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch')

    # plot bottom
    for idx, metric in enumerate(bottom_metrics):
        ax = axes[1, idx]
        if metric in df.columns:
            ax.plot(df['epoch'], df[metric], marker='o', linewidth=2, markersize=4)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch')

    plt.tight_layout()

    output_path = train_dir / 'training_results_1920x1080.png'
    try:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        found += 1
    except Exception as e:
        print(f"Failed to save plot for {train_dir}: {e}")
    finally:
        plt.close(fig)

if found == 0:
    print("No results.csv files found under runs/detect/*")
else:
    print(f"Generated {found} plot(s).")
