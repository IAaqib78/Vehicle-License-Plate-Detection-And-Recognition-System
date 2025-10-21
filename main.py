# main.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from utils import get_car, write_csv, read_license_plate
import os

# ==============================
# CONFIGURATION
# ==============================
# Models (absolute paths)
COCO_MODEL_PATH =  "path_to_the_yolo model"  # Vehicle detector
LP_MODEL_PATH = "Path_to_The_trained_model"  # License plate detector

# Video path (absolute path)
VIDEO_PATH = "sample_4.mp4"

# Vehicle class IDs in COCO (cars, buses, trucks, motorcycles)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Output CSV
OUTPUT_CSV = os.path.join(os.path.dirname(VIDEO_PATH), 'output.csv')


# ==============================
# INITIALIZATION
# ==============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

# Load YOLO models
coco_model = YOLO(COCO_MODEL_PATH)
license_plate_detector = YOLO(LP_MODEL_PATH)

# Initialize SORT tracker
mot_tracker = Sort()

# Video capture
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

# Results dictionary
results = {}
frame_nmr = -1

# ==============================
# DETECTION LOOP
# ==============================
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # --- Vehicle detection ---
    detections = coco_model(frame, verbose=False)[0]          # YOLO returns predictions
    vehicle_dets = []
    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in VEHICLE_CLASSES and score > 0.3:
            vehicle_dets.append([x1, y1, x2, y2, score])

    # --- Track vehicles ---
    tracked_vehicles = mot_tracker.update(np.array(vehicle_dets))

    # --- License plate detection ---
    lp_detections = license_plate_detector(frame, verbose=False)[0]
    for lp in lp_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = map(float, lp)
        car_x1, car_y1, car_x2, car_y2, car_id = get_car(lp, tracked_vehicles)
        if car_id == -1:
            continue

        # Crop license plate and check validity
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        lp_crop = frame[y1:y2, x1:x2]
        if lp_crop is None or lp_crop.size == 0:
            continue

        # OCR text extraction
        lp_text, lp_score = read_license_plate(lp_crop)
        if lp_text is None:
            continue

        # Save results
        results[frame_nmr][int(car_id)] = {
            'car': {'bbox': [car_x1, car_y1, car_x2, car_y2]},
            'license_plate': {
                'bbox': [x1, y1, x2, y2],
                'bbox_score': round(float(score), 3),
                'text': lp_text,
                'text_score': round(float(lp_score), 3)
            }
        }

        # Optional: print live detection
        print(f"Frame {frame_nmr}, Car {car_id}, Plate: {lp_text}, Score: {lp_score:.2f}")

# ==============================
# SAVE RESULTS
# ==============================
write_csv(results, OUTPUT_CSV)
print(f"Detection complete. Results saved to {OUTPUT_CSV}")

cap.release()
