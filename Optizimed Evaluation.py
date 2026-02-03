# evaluate_model_optimized.py
# Evaluation script for optimized model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import cv2
import os

print("Face Mask Detection - OPTIMIZED MODEL EVALUATION")
print("=" * 60)

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

print("\n[STEP 1] Loading optimized model...")

model = load_model('models/face_mask_detector_optimized.keras')
print("  Model loaded successfully")

print("\n[STEP 2] Loading test data...")

IMG_WIDTH = 224
IMG_HEIGHT = 224

data_dir = 'dataset'
with_mask_dir = os.path.join(data_dir, 'with_mask')
without_mask_dir = os.path.join(data_dir, 'without_mask')

images = []
labels = []

# Load images
for img_file in os.listdir(with_mask_dir):
    img_path = os.path.join(with_mask_dir, img_file)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        images.append(img)
        labels.append(1)

for img_file in os.listdir(without_mask_dir):
    img_path = os.path.join(without_mask_dir, img_file)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        images.append(img)
        labels.append(0)

images = np.array(images, dtype='float32')
labels = np.array(labels)

# Normalize (same as training)
images = images / 127.5 - 1.0

# Split (same random_state as training)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"  Loaded {len(X_test)} test images")

# =============================================================================
# MAKE PREDICTIONS
# =============================================================================

print("\n[STEP 3] Making predictions...")

predictions_prob = model.predict(X_test, verbose=1)
predictions = (predictions_prob > 0.5).astype(int).flatten()

print("  Predictions complete")

# =============================================================================
# CALCULATE METRICS
# =============================================================================

print("\n[STEP 4] Calculating metrics...")
print("=" * 60)

accuracy = accuracy_score(y_test, predictions)
print(f"\nOVERALL ACCURACY: {accuracy * 100:.2f}%")

print("\nDETAILED CLASSIFICATION REPORT:")
print("-" * 60)

report = classification_report(
    y_test,
    predictions,
    target_names=['Without Mask', 'With Mask'],
    digits=4
)
print(report)

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print("\n[STEP 5] Creating confusion matrix...")

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Without Mask', 'With Mask'],
    yticklabels=['Without Mask', 'With Mask'],
    cbar_kws={'label': 'Count'}
)

plt.title('Confusion Matrix - Optimized Model', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix_optimized.png', dpi=150, bbox_inches='tight')
print("  Confusion matrix saved")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives: {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives: {tp}")

# =============================================================================
# ROC CURVE AND AUC
# =============================================================================

print("\n[STEP 6] Creating ROC curve...")

fpr, tpr, thresholds = roc_curve(y_test, predictions_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Optimized Model', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve_optimized.png', dpi=150, bbox_inches='tight')
print(f"  ROC curve saved (AUC: {roc_auc:.4f})")
plt.show()

# =============================================================================
# SAVE METRICS
# =============================================================================

print("\n[STEP 7] Saving metrics...")

with open('results/evaluation_metrics_optimized.txt', 'w') as f:
    f.write("OPTIMIZED FACE MASK DETECTION MODEL - EVALUATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
    f.write("CONFUSION MATRIX:\n")
    f.write(f"True Negatives: {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives: {tp}\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write("-" * 60 + "\n")
    f.write(report)

print("  Metrics saved")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Improvement: {(accuracy - 0.9818) * 100:+.2f}%")
print("=" * 60)