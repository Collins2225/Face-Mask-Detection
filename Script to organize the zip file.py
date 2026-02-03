# organize_downloaded_dataset.py
# This script organizes the manually downloaded Kaggle dataset

import os
import zipfile
import shutil

print("Organizing Downloaded Dataset...")
print("=" * 50)

# Check if the zip file exists
if not os.path.exists('archive.zip'):
    print("✗ Error: archive.zip not found!")
    print("\nPlease:")
    print("1. Download from: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset")
    print("2. Place 'archive.zip' in your project folder")
    print("3. Run this script again")
    exit()

print("✓ Found archive.zip")

# Extract the zip file
print("\nExtracting archive.zip...")
try:
    # Open the zip file in read mode
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        # Extract all contents to a temporary folder
        zip_ref.extractall('temp_dataset')
    print("✓ Extraction complete")
except Exception as e:
    print(f"✗ Extraction failed: {e}")
    exit()

# Now let's find and organize the images
print("\nOrganizing images into dataset folders...")

# Counter variables to track our progress
with_mask_count = 0
without_mask_count = 0

# Walk through all extracted files
# os.walk() goes through every folder and subfolder
for root, dirs, files in os.walk('temp_dataset'):
    # root = current folder path we're looking at
    # dirs = list of subfolders in current folder
    # files = list of files in current folder

    for file in files:
        # Check if file is an image (by extension)
        # file.lower() converts filename to lowercase
        # .endswith() checks if filename ends with these extensions
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

            # Get the full path of the current file
            # os.path.join() combines folder path and filename properly
            source_path = os.path.join(root, file)

            # Determine destination based on folder name
            # We check if 'mask' is in the folder path to categorize
            if 'withmask' in root.lower() or ('mask' in root.lower() and 'without' not in root.lower()):
                # This is a WITH mask image
                destination = os.path.join('dataset/with_mask', file)
                # shutil.copy2() copies file with metadata
                shutil.copy2(source_path, destination)
                with_mask_count += 1

            elif 'withoutmask' in root.lower() or 'without' in root.lower():
                # This is a WITHOUT mask image
                destination = os.path.join('dataset/without_mask', file)
                shutil.copy2(source_path, destination)
                without_mask_count += 1

print(f"\n✓ Organized {with_mask_count} images with masks")
print(f"✓ Organized {without_mask_count} images without masks")

# Clean up temporary files
print("\nCleaning up temporary files...")
try:
    # shutil.rmtree() deletes a folder and all its contents
    shutil.rmtree('temp_dataset')
    print("✓ Temporary files removed")
except Exception as e:
    print(f"⚠ Warning: Could not remove temp files: {e}")

print("\n" + "=" * 50)
print("Dataset Setup Complete!")
print(f"\nTotal images: {with_mask_count + without_mask_count}")
print(f"  - With mask: {with_mask_count}")
print(f"  - Without mask: {without_mask_count}")
print("=" * 50)

print("\nYou can now run: python data_exploration.py")