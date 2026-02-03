# test_installations.py
# This script checks if all required libraries are installed correctly

print("Testing library installations...\n")
print("=" * 50)

# Test 1: TensorFlow
try:
    import tensorflow as tf
    print("✓ TensorFlow installed successfully")
    print(f"  Version: {tf.__version__}")
except ImportError as e:
    print("✗ TensorFlow NOT installed")
    print(f"  Error: {e}")

# Test 2: OpenCV
try:
    import cv2
    print("✓ OpenCV installed successfully")
    print(f"  Version: {cv2.__version__}")
except ImportError as e:
    print("✗ OpenCV NOT installed")
    print(f"  Error: {e}")

# Test 3: NumPy
try:
    import numpy as np
    print("✓ NumPy installed successfully")
    print(f"  Version: {np.__version__}")
except ImportError as e:
    print("✗ NumPy NOT installed")
    print(f"  Error: {e}")

# Test 4: Matplotlib
try:
    import matplotlib
    print("✓ Matplotlib installed successfully")
    print(f"  Version: {matplotlib.__version__}")
except ImportError as e:
    print("✗ Matplotlib NOT installed")
    print(f"  Error: {e}")

# Test 5: Scikit-learn
try:
    import sklearn
    print("✓ Scikit-learn installed successfully")
    print(f"  Version: {sklearn.__version__}")
except ImportError as e:
    print("✗ Scikit-learn NOT installed")
    print(f"  Error: {e}")

# Test 6: Pillow (PIL)
try:
    from PIL import Image
    import PIL
    print("✓ Pillow installed successfully")
    print(f"  Version: {PIL.__version__}")
except ImportError as e:
    print("✗ Pillow NOT installed")
    print(f"  Error: {e}")

print("=" * 50)
print("\nInstallation test complete!")