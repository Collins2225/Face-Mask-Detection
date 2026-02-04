# real_time_detection_lowlight.py
# Optimized for low-light conditions with automatic brightness adjustment

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import urllib.request
import os

print("Face Mask Detection - LOW LIGHT OPTIMIZED")
print("=" * 60)

# Load model
print("\n[STEP 1] Loading model...")
model = load_model('models/face_mask_detector_optimized.keras')
print("  Model loaded")

# Setup face detector
print("\n[STEP 2] Setting up face detector...")

prototxt_path = "deploy.prototxt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
    print("  Downloading face detector files...")
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    urllib.request.urlretrieve(prototxt_url, prototxt_path)
    urllib.request.urlretrieve(caffemodel_url, caffemodel_path)

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
print("  Face detector ready")

# Initialize webcam
print("\n[STEP 3] Initializing webcam with optimized settings...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("  ERROR: Could not open webcam")
    exit()

# Optimize camera settings for low light
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# AUTO EXPOSURE AND BRIGHTNESS
# These settings help in low light
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Increase exposure time (negative = longer)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Increase brightness (0.0 to 1.0)
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)  # Adjust contrast
cap.set(cv2.CAP_PROP_GAIN, 10)  # Increase gain for low light

print("  Webcam optimized for low-light conditions")


# Smoothing classes
class SmoothPredictor:
    def __init__(self, window_size=15):
        self.predictions = deque(maxlen=window_size)

    def add_prediction(self, prediction):
        self.predictions.append(prediction)
        if len(self.predictions) > 0:
            return np.mean(self.predictions)
        return prediction

    def reset(self):
        self.predictions.clear()


class SmoothBoundingBox:
    def __init__(self, window_size=7):
        self.x_coords = deque(maxlen=window_size)
        self.y_coords = deque(maxlen=window_size)
        self.w_coords = deque(maxlen=window_size)
        self.h_coords = deque(maxlen=window_size)

    def add_box(self, x, y, w, h):
        self.x_coords.append(x)
        self.y_coords.append(y)
        self.w_coords.append(w)
        self.h_coords.append(h)

        return (
            int(np.mean(self.x_coords)),
            int(np.mean(self.y_coords)),
            int(np.mean(self.w_coords)),
            int(np.mean(self.h_coords))
        )

    def reset(self):
        self.x_coords.clear()
        self.y_coords.clear()
        self.w_coords.clear()
        self.h_coords.clear()


def enhance_brightness(frame):
    """
    Automatically enhance brightness for low-light conditions
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    # Convert to LAB color space
    # L = Lightness, A = Green-Red, B = Blue-Yellow
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    # clipLimit: Threshold for contrast limiting (higher = more contrast)
    # tileGridSize: Size of grid for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge channels back
    enhanced_lab = cv2.merge([l_enhanced, a, b])

    # Convert back to BGR
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_frame


def detect_faces_dnn(frame, confidence_threshold=0.5):
    """
    Detect faces with LOWER threshold for low light
    """
    (h, w) = frame.shape[:2]

    # Enhance frame brightness before detection
    enhanced = enhance_brightness(frame)

    blob = cv2.dnn.blobFromImage(
        cv2.resize(enhanced, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # LOWER threshold for better detection in low light
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            x = max(0, startX)
            y = max(0, startY)
            width = min(w - x, endX - startX)
            height = min(h - y, endY - startY)

            faces.append((x, y, width, height))

    return faces


def preprocess_face(face_img):
    """Preprocess with brightness enhancement"""
    if face_img.size == 0:
        return None

    # Enhance brightness BEFORE preprocessing
    face_img = enhance_brightness(face_img)

    # Resize to 224x224
    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Apply Gaussian blur
    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)

    # Normalize to [-1, 1]
    face_img = face_img.astype('float32')
    face_img = face_img / 127.5 - 1.0

    # Add batch dimension
    face_img = np.expand_dims(face_img, axis=0)

    return face_img


def draw_detection_box(frame, x, y, w, h, label, confidence, color):
    """Draw detection box"""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Corner accents
    corner_length = 20
    corner_thickness = 4

    cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thickness)

    cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, corner_thickness)

    cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, corner_thickness)

    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)

    # Label
    label_text = f"{label}: {confidence:.1f}%"
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    padding = 5
    cv2.rectangle(
        frame,
        (x, y - text_h - padding * 2 - baseline),
        (x + text_w + padding * 2, y),
        color,
        -1
    )

    cv2.putText(
        frame,
        label_text,
        (x + padding, y - padding - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# Main loop
print("\n[STEP 4] Starting low-light optimized detection...")
print("=" * 60)
print("OPTIMIZATIONS:")
print("  - Automatic brightness enhancement (CLAHE)")
print("  - Increased camera exposure")
print("  - Lower detection threshold (50%)")
print("  - Contrast adjustment")
print("\nCONTROLS:")
print("  - Press 'q' to quit")
print("  - Press 's' to screenshot")
print("  - Press 'b' to toggle brightness boost")
print("=" * 60)

predictor_smoother = SmoothPredictor(window_size=15)
box_smoother = SmoothBoundingBox(window_size=7)

frame_count = 0
screenshot_count = 0
brightness_boost = True  # Toggle for brightness enhancement

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Apply brightness enhancement if enabled
    if brightness_boost:
        frame = enhance_brightness(frame)

    height, width, _ = frame.shape

    # Detect faces (using enhanced frame and lower threshold)
    faces = detect_faces_dnn(frame, confidence_threshold=0.5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Smooth bounding box
            x_smooth, y_smooth, w_smooth, h_smooth = box_smoother.add_box(x, y, w, h)

            # Extract and preprocess face
            face_img = frame[y_smooth:y_smooth + h_smooth, x_smooth:x_smooth + w_smooth]
            preprocessed_face = preprocess_face(face_img)

            if preprocessed_face is not None:
                # Predict
                prediction = model.predict(preprocessed_face, verbose=0)[0][0]

                # Smooth prediction
                smoothed_prediction = predictor_smoother.add_prediction(prediction)

                # Determine label
                if smoothed_prediction > 0.5:
                    label = "MASK"
                    confidence = smoothed_prediction * 100
                    color = (0, 255, 0)
                else:
                    label = "NO MASK"
                    confidence = (1 - smoothed_prediction) * 100
                    color = (0, 0, 255)

                # Draw
                draw_detection_box(
                    frame,
                    x_smooth,
                    y_smooth,
                    w_smooth,
                    h_smooth,
                    label,
                    confidence,
                    color
                )
    else:
        predictor_smoother.reset()
        box_smoother.reset()

    # Info overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    info_lines = [
        f"Frame: {frame_count} | LOW-LIGHT MODE",
        f"Faces: {len(faces)}",
        f"Brightness Boost: {'ON' if brightness_boost else 'OFF'}",
        f"'q' quit | 's' screenshot | 'b' toggle boost"
    ]

    for i, line in enumerate(info_lines):
        cv2.putText(
            frame,
            line,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imshow('Low-Light Optimized Face Mask Detection', frame)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        screenshot_count += 1
        filename = f'results/screenshot_lowlight_{screenshot_count}.png'
        cv2.imwrite(filename, frame)
        print(f"  Screenshot saved: {filename}")
    elif key == ord('b'):
        brightness_boost = not brightness_boost
        print(f"  Brightness boost: {'ON' if brightness_boost else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("DETECTION COMPLETE!")
print("=" * 60)