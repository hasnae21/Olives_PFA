import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
csv_path = '/home/wayay/Documents/Olives_Pfa/Olives_PFA/runs/detect/train5/results.csv'
df = pd.read_csv(csv_path)

# Create figure with 2 rows and 5 columns
fig, axes = plt.subplots(2, 5, figsize=(1920/100, 1080/100), dpi=100)

# Metrics to plot - top row
top_metrics = [
    'train/box_loss',
    'train/cls_loss', 
    'train/dfl_loss',
    'metrics/precision(B)',
    'metrics/recall(B)'
]

# Metrics to plot - bottom row
bottom_metrics = [
    'val/box_loss',
    'val/cls_loss',
    'val/dfl_loss',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)'
]

# Plot top row
for idx, metric in enumerate(top_metrics):
    ax = axes[0, idx]
    ax.plot(df['epoch'], df[metric], marker='o', linewidth=2, markersize=4)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch')

# Plot bottom row
for idx, metric in enumerate(bottom_metrics):
    ax = axes[1, idx]
    ax.plot(df['epoch'], df[metric], marker='o', linewidth=2, markersize=4)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch')

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = '/home/wayay/Documents/Olives_Pfa/Olives_PFA/training_results_1920x1080.png'
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"Image saved to: {output_path}")

# Also display the image
plt.show()
