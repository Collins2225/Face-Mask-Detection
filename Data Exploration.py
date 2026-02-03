# data_exploration.py
# This script explores our dataset and shows sample images

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("Exploring Face Mask Dataset...")
print("=" * 50)

# Define paths to our dataset folders
# These variables store the location of our image folders
with_mask_path = 'dataset/with_mask'
without_mask_path = 'dataset/without_mask'

# Count images in each folder
# os.listdir() returns a list of all files in a directory
# len() counts how many items are in the list
with_mask_images = os.listdir(with_mask_path)
without_mask_images = os.listdir(without_mask_path)

# Calculate totals
num_with_mask = len(with_mask_images)
num_without_mask = len(without_mask_images)
total_images = num_with_mask + num_without_mask

# Display statistics
print("\nDataset Statistics:")
print(f"  Images WITH mask: {num_with_mask}")
print(f"  Images WITHOUT mask: {num_without_mask}")
print(f"  Total images: {total_images}")
print(f"  Balance ratio: {num_with_mask}/{num_without_mask}")

# Check if dataset is balanced (roughly equal images in both classes)
if num_with_mask > 0 and num_without_mask > 0:
    # Calculate the ratio to see how balanced our dataset is
    ratio = num_with_mask / num_without_mask
    if 0.8 <= ratio <= 1.2:
        print("  ✓ Dataset is well balanced")
    else:
        print("  ⚠ Dataset is imbalanced (may affect training)")
else:
    print("  ✗ ERROR: One or both folders are empty!")
    print("\nPlease ensure you have images in both folders:")
    print(f"  - {with_mask_path}")
    print(f"  - {without_mask_path}")
    exit()

print("\n" + "=" * 50)

# Visualize sample images
print("\nCreating visualization of sample images...")

# Create a figure with subplots
# plt.figure() creates a new figure window
# figsize=(15, 8) sets the width and height in inches
plt.figure(figsize=(15, 8))

# We'll display 10 images: 5 with masks, 5 without masks
num_samples = 5

# Display images WITH masks
for i in range(num_samples):
    # plt.subplot(rows, cols, position) creates a grid of subplots
    # 2 rows, 5 columns, position (i+1)
    plt.subplot(2, num_samples, i + 1)

    # Get the image filename
    # with_mask_images[i] gets the i-th image filename from our list
    img_name = with_mask_images[i]

    # Load the image using OpenCV
    # os.path.join() combines folder path and filename
    # cv2.imread() reads the image file
    img_path = os.path.join(with_mask_path, img_name)
    img = cv2.imread(img_path)

    # OpenCV loads images in BGR (Blue-Green-Red) format
    # But matplotlib displays in RGB (Red-Green-Blue) format
    # cv2.cvtColor() converts color format
    # cv2.COLOR_BGR2RGB tells it to convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    # plt.imshow() shows the image
    plt.imshow(img_rgb)

    # Add title to the subplot
    plt.title('With Mask', fontsize=10)

    # Remove axis ticks for cleaner look
    # plt.axis('off') hides the x and y axis numbers
    plt.axis('off')

# Display images WITHOUT masks
for i in range(num_samples):
    # Position in second row: num_samples + i + 1
    plt.subplot(2, num_samples, num_samples + i + 1)

    img_name = without_mask_images[i]
    img_path = os.path.join(without_mask_path, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title('Without Mask', fontsize=10)
    plt.axis('off')

# Adjust spacing between subplots
# plt.tight_layout() automatically adjusts subplot spacing
plt.tight_layout()

# Save the visualization
# plt.savefig() saves the figure as an image file
plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: results/sample_images.png")

# Show the plot
# plt.show() displays the figure window
plt.show()

print("\n" + "=" * 50)
print("Data exploration complete!")
print("\nNext step: Run data_preprocessing.py to prepare data for training")