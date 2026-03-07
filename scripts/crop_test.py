from PIL import Image
import os

# Load the test image
img_path = r'c:\Users\louay\Downloads\olives_2\second\IMG_0118.jpeg'
img = Image.open(img_path)

width, height = img.size
print(f"Original image size: {width} x {height}")
print()

# Interactive crop test
while True:
    try:
        pixels_to_remove = int(input("How many pixels to remove from the bottom? "))
        
        if pixels_to_remove < 0 or pixels_to_remove >= height:
            print(f"Please enter a value between 0 and {height-1}")
            continue
        
        # Crop the image (left, upper, right, lower)
        new_height = height - pixels_to_remove
        cropped_img = img.crop((0, 0, width, new_height))
        
        print(f"New image size: {cropped_img.size}")
        
        # Save preview
        preview_path = r'c:\Users\louay\Downloads\olives_2\test_preview.jpeg'
        cropped_img.save(preview_path, quality=95)
        print(f"✓ Preview saved to: {preview_path}")
        print()
        
        apply = input("Are you happy with this crop? (yes/no): ").strip().lower()
        if apply in ['yes', 'y']:
            print(f"\nCropping {pixels_to_remove} pixels from bottom will be applied to all images.")
            print(f"Pixels to remove: {pixels_to_remove}")
            break
        
    except ValueError:
        print("Please enter a valid number")
