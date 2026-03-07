from PIL import Image, ImageFilter, ImageEnhance
import os
from pathlib import Path
import random
from itertools import combinations

# Settings
source_image_folder = r'c:\Users\louay\Downloads\olives_2\data\image'
source_label_folder = r'c:\Users\louay\Downloads\olives_2\data\label'
output_image_folder = r'c:\Users\louay\Downloads\olives_2\data\image_augmented'
output_label_folder = r'c:\Users\louay\Downloads\olives_2\data\label_augmented'

# Seed for reproducibility
random.seed(42)

# Create output folders
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

def transform_yolo_coords(x_center, y_center, width, height, transform_type):
    """Transform YOLO coordinates based on image transformation"""
    
    if transform_type == "rotate_90":
        # 90° rotation: (x,y) -> (y, 1-x)
        new_x = y_center
        new_y = 1 - x_center
        # Width and height swap
        new_w = height
        new_h = width
        
    elif transform_type == "rotate_180":
        # 180° rotation: (x,y) -> (1-x, 1-y)
        new_x = 1 - x_center
        new_y = 1 - y_center
        new_w = width
        new_h = height
        
    elif transform_type == "rotate_270":
        # 270° rotation: (x,y) -> (1-y, x)
        new_x = 1 - y_center
        new_y = x_center
        new_w = height
        new_h = width
        
    elif transform_type == "flip_vertical":
        # Vertical flip: (x,y) -> (x, 1-y)
        new_x = x_center
        new_y = 1 - y_center
        new_w = width
        new_h = height
        
    else:
        # No transformation (blur, brightness)
        new_x = x_center
        new_y = y_center
        new_w = width
        new_h = height
    
    # Ensure values are in [0, 1]
    new_x = max(0, min(1, new_x))
    new_y = max(0, min(1, new_y))
    new_w = max(0, min(1, new_w))
    new_h = max(0, min(1, new_h))
    
    return new_x, new_y, new_w, new_h

def read_yolo_labels(label_path):
    """Read YOLO format label file"""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((class_id, x_center, y_center, width, height))
    
    return labels

def write_yolo_labels(label_path, labels):
    """Write YOLO format label file"""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for class_id, x_center, y_center, width, height in labels:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_image(img, labels, augmentation_list):
    """Apply multiple augmentations to image and transform labels accordingly"""
    
    current_img = img.copy()
    current_labels = labels.copy()
    
    # Apply each augmentation in the list sequentially
    for aug_type in augmentation_list:
        if aug_type == "rotate_90":
            current_img = current_img.transpose(Image.ROTATE_270)  # PIL transpose inverse
        elif aug_type == "rotate_180":
            current_img = current_img.transpose(Image.ROTATE_180)
        elif aug_type == "rotate_270":
            current_img = current_img.transpose(Image.ROTATE_90)  # PIL transpose inverse
        elif aug_type == "flip_vertical":
            current_img = current_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif aug_type == "blur":
            current_img = current_img.filter(ImageFilter.GaussianBlur(radius=2))
        elif aug_type == "brightness_increase":
            enhancer = ImageEnhance.Brightness(current_img)
            current_img = enhancer.enhance(1.2)  # 20% brighter
        elif aug_type == "brightness_decrease":
            enhancer = ImageEnhance.Brightness(current_img)
            current_img = enhancer.enhance(0.8)  # 20% darker
        
        # Transform labels for geometric transformations
        if aug_type in ["rotate_90", "rotate_180", "rotate_270", "flip_vertical"]:
            transformed_labels = []
            for class_id, x_center, y_center, width, height in current_labels:
                new_x, new_y, new_w, new_h = transform_yolo_coords(
                    x_center, y_center, width, height, aug_type
                )
                transformed_labels.append((class_id, new_x, new_y, new_w, new_h))
            current_labels = transformed_labels
    
    return current_img, current_labels

# Get all image files
image_files = sorted([f for f in os.listdir(source_image_folder) 
                     if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

print(f"Found {len(image_files)} images to augment")
print("-" * 60)

# Define available augmentations
all_augmentations = [
    "rotate_90",
    "rotate_180", 
    "rotate_270",
    "flip_vertical",
    "blur",
    "brightness_increase",
    "brightness_decrease"
]

# Generate combinations
combinations_1 = [[aug] for aug in all_augmentations]  # 7 combinations of 1
combinations_2 = list(combinations(all_augmentations, 2))  # C(7,2) = 21 combinations of 2
combinations_3 = list(combinations(all_augmentations, 3))  # C(7,3) = 35 combinations of 3

print(f"Augmentation combinations:")
print(f"  - 1 transformation: {len(combinations_1)}")
print(f"  - 2 transformations: {len(combinations_2)}")
print(f"  - 3 transformations: {len(combinations_3)}")
print("-" * 60)

total_processed = 0
total_images_created = 0

for idx, filename in enumerate(image_files, 1):
    try:
        # Load image
        img_path = os.path.join(source_image_folder, filename)
        img = Image.open(img_path).convert('RGB')
        
        # Load labels
        label_filename = Path(filename).stem + '.txt'
        label_path = os.path.join(source_label_folder, label_filename)
        labels = read_yolo_labels(label_path)
        
        # Copy original
        output_img_path = os.path.join(output_image_folder, filename)
        output_label_path = os.path.join(output_label_folder, label_filename)
        img.save(output_img_path, quality=95)
        write_yolo_labels(output_label_path, labels)
        total_images_created += 1
        
        # Get random combinations
        # 3 images with 1 transformation
        selected_1 = random.sample(combinations_1, min(3, len(combinations_1)))
        # 3 images with 2 transformations
        selected_2 = random.sample(combinations_2, min(3, len(combinations_2)))
        # 2 images with 3 transformations
        selected_3 = random.sample(combinations_3, min(2, len(combinations_3)))
        
        all_selected = selected_1 + selected_2 + selected_3
        
        # Apply augmentations
        for aug_idx, aug_list in enumerate(all_selected, 1):
            augmented_img, transformed_labels = augment_image(img, labels, aug_list)
            
            # Create filename with augmentation info
            name_without_ext = Path(filename).stem
            ext = Path(filename).suffix
            aug_names = "_".join(aug_list)
            output_filename = f"{name_without_ext}_aug{len(aug_list)}_{aug_idx}_{aug_names}{ext}"
            
            # Save augmented image
            output_img_path = os.path.join(output_image_folder, output_filename)
            augmented_img.save(output_img_path, quality=95)
            
            # Save transformed labels
            output_label_path = os.path.join(output_label_folder, 
                                            f"{name_without_ext}_aug{len(aug_list)}_{aug_idx}_{aug_names}.txt")
            write_yolo_labels(output_label_path, transformed_labels)
            
            total_images_created += 1
        
        total_processed += 1
        if idx % 10 == 0 or idx == len(image_files):
            print(f"✓ Processed {idx}/{len(image_files)} | Total images created: {total_images_created}")
        
    except Exception as e:
        print(f"✗ Error processing {filename}: {str(e)}")

print("-" * 60)
print(f"✓ Complete!")
print(f"Original images: {total_processed}")
print(f"Augmented variations per image:")
print(f"  - 3 images with 1 transformation")
print(f"  - 3 images with 2 transformations")
print(f"  - 2 images with 3 transformations")
print(f"Total augmented images: {total_images_created - total_processed}")
print(f"TOTAL files created: {total_images_created}")
print(f"\nOutput folders:")
print(f"  Images: {output_image_folder}")
print(f"  Labels: {output_label_folder}")
