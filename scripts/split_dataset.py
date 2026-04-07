import os
import shutil
from pathlib import Path
import random

# Settings
augmented_image_folder = r'C:\Users\louay\Downloads\olives_2\treated\image'
augmented_label_folder = r'c:\Users\louay\Downloads\olives_2\treated\label'
output_base_folder = r'c:\Users\louay\Downloads\olives_2\test_split'

# Train/Val/Test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Seed for reproducibility
random.seed(42)

# Create output folder structure
output_folders = {
    'train': {
        'images': os.path.join(output_base_folder, 'train', 'images'),
        'labels': os.path.join(output_base_folder, 'train', 'labels')
    },
    'val': {
        'images': os.path.join(output_base_folder, 'valid', 'images'),
        'labels': os.path.join(output_base_folder, 'valid', 'labels')
    },
    'test': {
        'images': os.path.join(output_base_folder, 'test', 'images'),
        'labels': os.path.join(output_base_folder, 'test', 'labels')
    }
}

# Create all directories
for split in output_folders.values():
    for path in split.values():
        os.makedirs(path, exist_ok=True)

# Get all image files
image_files = sorted([f for f in os.listdir(augmented_image_folder) 
                     if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

print(f"Found {len(image_files)} augmented images")
print(f"Split ratio: Train {TRAIN_RATIO*100}% | Val {VAL_RATIO*100}% | Test {TEST_RATIO*100}%")
print("-" * 60)

# Shuffle the files
random.shuffle(image_files)

# Calculate split points
total = len(image_files)
train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)
test_count = total - train_count - val_count

# Split files
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Dictionary to track file counts
splits = {
    'train': train_files,
    'val': val_files,
    'test': test_files
}

# Copy files
total_copied = 0

for split_name, files in splits.items():
    for idx, image_filename in enumerate(files, 1):
        try:
            # Get corresponding label file
            label_filename = Path(image_filename).stem + '.txt'
            
            # Source paths
            src_image = os.path.join(augmented_image_folder, image_filename)
            src_label = os.path.join(augmented_label_folder, label_filename)
            
            # Destination paths
            dst_image = os.path.join(output_folders[split_name]['images'], image_filename)
            dst_label = os.path.join(output_folders[split_name]['labels'], label_filename)
            
            # Copy image
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            
            # Copy label
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            
            total_copied += 1
            
            # Progress update
            if idx % 500 == 0 or idx == len(files):
                print(f"✓ {split_name.upper()}: {idx}/{len(files)}")
        
        except Exception as e:
            print(f"✗ Error copying {image_filename}: {str(e)}")

print("-" * 60)
print(f"✓ Split Complete!")
print(f"\nDataset distribution:")
print(f"  Train: {len(train_files)} images ({len(train_files)/total*100:.1f}%)")
print(f"  Val:   {len(val_files)} images ({len(val_files)/total*100:.1f}%)")
print(f"  Test:  {len(test_files)} images ({len(test_files)/total*100:.1f}%)")
print(f"  TOTAL: {total} images")
print(f"\nOutput structure:")
print(f"  {output_base_folder}/")
print(f"    ├── train/")
print(f"    │   ├── images/ ({len(train_files)} images)")
print(f"    │   └── labels/ ({len(train_files)} labels)")
print(f"    ├── valid/")
print(f"    │   ├── images/ ({len(val_files)} images)")
print(f"    │   └── labels/ ({len(val_files)} labels)")
print(f"    └── test/")
print(f"        ├── images/ ({len(test_files)} images)")
print(f"        └── labels/ ({len(test_files)} labels)")
