from PIL import Image
import os
from pathlib import Path

# Settings
CROP_AMOUNT = 750
source_folder = r'c:\Users\louay\Downloads\olives_2\second'

# Get all JPEG files
image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpeg')])
total_images = len(image_files)

print(f"Found {total_images} images to process")
print(f"Removing {CROP_AMOUNT} pixels from the bottom of each image")
print("-" * 50)

processed = 0
failed = 0

for idx, filename in enumerate(image_files, 1):
    try:
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path)
        
        width, height = img.size
        new_height = height - CROP_AMOUNT
        
        # Crop the image
        cropped_img = img.crop((0, 0, width, new_height))
        
        # Save back to the same location
        cropped_img.save(img_path, quality=95)
        
        processed += 1
        if idx % 50 == 0 or idx == total_images:
            print(f"✓ Processed {idx}/{total_images}")
        
    except Exception as e:
        failed += 1
        print(f"✗ Error processing {filename}: {str(e)}")

print("-" * 50)
print(f"Complete! Processed: {processed}/{total_images}")
if failed > 0:
    print(f"Failed: {failed}")
