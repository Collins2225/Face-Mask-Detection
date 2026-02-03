# train_model_optimized.py
# Optimized training with transfer learning and advanced techniques

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import cv2

print("Face Mask Detection - OPTIMIZED TRAINING")
print("=" * 60)

# =============================================================================
# STEP 1: CONFIGURATION
# =============================================================================

print("\n[STEP 1] Setting up configuration...")

# Image dimensions - using larger size for better accuracy
IMG_WIDTH = 224  # MobileNetV2 expects 224x224
IMG_HEIGHT = 224
IMG_CHANNELS = 3

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning

print(f"  Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

# =============================================================================
# STEP 2: LOAD AND PREPROCESS DATA WITH BETTER TECHNIQUES
# =============================================================================

print("\n[STEP 2] Loading and preprocessing data...")

data_dir = 'dataset'
with_mask_dir = os.path.join(data_dir, 'with_mask')
without_mask_dir = os.path.join(data_dir, 'without_mask')

images = []
labels = []


def load_and_preprocess_image(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Load and preprocess image with better quality
    """
    # Read image
    img = cv2.imread(img_path)

    if img is None:
        return None

    # Resize with high-quality interpolation
    # cv2.INTER_AREA is best for shrinking images
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply slight Gaussian blur to reduce noise
    # This helps model focus on important features, not noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


print("  Loading images WITH masks...")
with_mask_files = os.listdir(with_mask_dir)
for img_file in with_mask_files:
    img_path = os.path.join(with_mask_dir, img_file)
    img = load_and_preprocess_image(img_path)

    if img is not None:
        images.append(img)
        labels.append(1)  # 1 = with mask

print(f"    Loaded {sum(labels)} images with masks")

print("  Loading images WITHOUT masks...")
without_mask_files = os.listdir(without_mask_dir)
for img_file in without_mask_files:
    img_path = os.path.join(without_mask_dir, img_file)
    img = load_and_preprocess_image(img_path)

    if img is not None:
        images.append(img)
        labels.append(0)  # 0 = without mask

print(f"    Loaded {len(labels) - sum(labels)} images without masks")

# Convert to numpy arrays
images = np.array(images, dtype='float32')
labels = np.array(labels)

print(f"\n  Total images: {len(images)}")

# Normalize pixel values
# MobileNetV2 expects pixels in range [-1, 1]
images = images / 127.5 - 1.0

print("  Pixel values normalized to [-1, 1]")

# =============================================================================
# STEP 3: SPLIT DATA WITH STRATIFICATION
# =============================================================================

print("\n[STEP 3] Splitting data...")

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # Ensures balanced distribution
)

print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")
print(f"  Train - With mask: {sum(y_train)}, Without: {len(y_train) - sum(y_train)}")
print(f"  Test - With mask: {sum(y_test)}, Without: {len(y_test) - sum(y_test)}")

# =============================================================================
# STEP 4: ADVANCED DATA AUGMENTATION
# =============================================================================

print("\n[STEP 4] Setting up advanced data augmentation...")

# More aggressive augmentation for better generalization
train_datagen = ImageDataGenerator(
    rotation_range=30,  # Rotate up to 30 degrees
    width_shift_range=0.2,  # Shift horizontally
    height_shift_range=0.2,  # Shift vertically
    shear_range=0.2,  # Shearing transformation
    zoom_range=0.3,  # Zoom in/out
    horizontal_flip=True,  # Mirror horizontally
    brightness_range=[0.7, 1.3],  # Vary brightness
    fill_mode='nearest'  # Fill empty pixels
)

# No augmentation for test data
test_datagen = ImageDataGenerator()

print("  Training augmentation configured:")
print("    - Rotation: ±30 degrees")
print("    - Shifts: ±20%")
print("    - Zoom: ±30%")
print("    - Brightness: 70%-130%")

# =============================================================================
# STEP 5: BUILD OPTIMIZED MODEL WITH TRANSFER LEARNING
# =============================================================================

print("\n[STEP 5] Building optimized model with transfer learning...")

# Load MobileNetV2 pre-trained on ImageNet
# include_top=False: Remove the final classification layer
# weights='imagenet': Use pre-trained weights from ImageNet
# input_shape: Our input image dimensions
base_model = MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
    include_top=False,
    weights='imagenet'
)

print("  MobileNetV2 base model loaded")
print(f"  Total layers in base model: {len(base_model.layers)}")

# Freeze the base model layers initially
# This preserves the pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

print("  Base model layers frozen (will fine-tune later)")

# Build our custom top layers
# x is the output from base_model
x = base_model.output

# GlobalAveragePooling2D reduces each feature map to single value
# Better than Flatten for transfer learning
x = GlobalAveragePooling2D()(x)

# BatchNormalization normalizes activations
# Helps training stability and speed
x = BatchNormalization()(x)

# Dense layer with 256 neurons
x = Dense(256, activation='relu')(x)

# Dropout to prevent overfitting
x = Dropout(0.5)(x)

# Another BatchNormalization
x = BatchNormalization()(x)

# Another Dense layer
x = Dense(128, activation='relu')(x)

# Dropout again
x = Dropout(0.4)(x)

# Output layer
# 1 neuron with sigmoid for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

print("\n  Custom layers added:")
print("    - GlobalAveragePooling2D")
print("    - BatchNormalization")
print("    - Dense(256) + Dropout(0.5)")
print("    - BatchNormalization")
print("    - Dense(128) + Dropout(0.4)")
print("    - Dense(1, sigmoid)")

# =============================================================================
# STEP 6: COMPILE MODEL WITH OPTIMIZED SETTINGS
# =============================================================================

print("\n[STEP 6] Compiling model...")

# Adam optimizer with lower learning rate for fine-tuning
optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("  Model compiled")

# Display model summary
print("\n" + "=" * 60)
print("MODEL SUMMARY:")
print("=" * 60)
model.summary()
print("=" * 60)

# =============================================================================
# STEP 7: SETUP ADVANCED CALLBACKS
# =============================================================================

print("\n[STEP 7] Setting up callbacks...")

# ModelCheckpoint - save best model
checkpoint = ModelCheckpoint(
    'models/optimized_best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# EarlyStopping - stop if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # More patience for better results
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau - reduce learning rate when plateau
# This helps escape local minima
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Multiply LR by 0.5
    patience=5,  # Wait 5 epochs before reducing
    min_lr=1e-7,  # Minimum learning rate
    verbose=1
)

print("  Callbacks configured:")
print("    - ModelCheckpoint (save best)")
print("    - EarlyStopping (patience=10)")
print("    - ReduceLROnPlateau (factor=0.5, patience=5)")

# =============================================================================
# STEP 8: CALCULATE CLASS WEIGHTS
# =============================================================================

print("\n[STEP 8] Calculating class weights...")

# Calculate class weights to handle imbalanced data
# Gives more weight to minority class
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print(f"  Class weights: {class_weights_dict}")
print("  This helps handle class imbalance")

# =============================================================================
# STEP 9: TRAIN MODEL (PHASE 1 - FROZEN BASE)
# =============================================================================

print("\n[STEP 9] Starting training - PHASE 1 (frozen base)...")
print("=" * 60)

# Train with frozen base model first
history_phase1 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=15,  # Train for 15 epochs first
    validation_data=test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
    validation_steps=len(X_test) // BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights_dict,
    verbose=1
)

print("\n" + "=" * 60)
print("PHASE 1 TRAINING COMPLETE")
print("=" * 60)

# =============================================================================
# STEP 10: FINE-TUNING (PHASE 2 - UNFREEZE LAYERS)
# =============================================================================

print("\n[STEP 10] Starting PHASE 2 - Fine-tuning...")

# Unfreeze the last 50 layers of base model for fine-tuning
# This allows the model to adapt pre-trained features to our specific task
print("  Unfreezing last 50 layers of base model...")

for layer in base_model.layers[-50:]:
    layer.trainable = True

print(f"  Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

# Re-compile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # 10x lower LR
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("  Model recompiled with lower learning rate")

# Continue training (fine-tuning)
print("\n  Starting fine-tuning...")
print("=" * 60)

history_phase2 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,  # Train for remaining epochs
    validation_data=test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
    validation_steps=len(X_test) // BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weights_dict,
    verbose=1
)

print("\n" + "=" * 60)
print("PHASE 2 FINE-TUNING COMPLETE")
print("=" * 60)

# =============================================================================
# STEP 11: SAVE FINAL MODEL
# =============================================================================

print("\n[STEP 11] Saving final model...")

model.save('models/face_mask_detector_optimized.keras')
print("  Model saved to: models/face_mask_detector_optimized.keras")

# =============================================================================
# STEP 12: VISUALIZE TRAINING HISTORY
# =============================================================================

print("\n[STEP 12] Creating training visualizations...")

# Combine both phases
history_combined = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot accuracy
epochs_range = range(1, len(history_combined['accuracy']) + 1)
ax1.plot(epochs_range, history_combined['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs_range, history_combined['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)

# Mark phase transition
phase1_end = len(history_phase1.history['accuracy'])
ax1.axvline(x=phase1_end, color='green', linestyle='--', linewidth=2, label='Fine-tuning starts')

ax1.set_title('Model Accuracy (Optimized Training)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(epochs_range, history_combined['loss'], 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs_range, history_combined['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax2.axvline(x=phase1_end, color='green', linestyle='--', linewidth=2, label='Fine-tuning starts')

ax2.set_title('Model Loss (Optimized Training)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history_optimized.png', dpi=150, bbox_inches='tight')
print("  Training history saved to: results/training_history_optimized.png")
plt.show()

# Print final metrics
final_train_acc = history_combined['accuracy'][-1]
final_val_acc = history_combined['val_accuracy'][-1]

print("\n" + "=" * 60)
print("FINAL TRAINING RESULTS:")
print("=" * 60)
print(f"  Final Training Accuracy: {final_train_acc * 100:.2f}%")
print(f"  Final Validation Accuracy: {final_val_acc * 100:.2f}%")
print("=" * 60)

print("\nNext step: Run evaluate_model_optimized.py to test performance")
print("=" * 60)