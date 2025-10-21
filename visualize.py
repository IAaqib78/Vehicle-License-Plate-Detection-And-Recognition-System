import cv2
import numpy as np
import pandas as pd
import os


# ======= CONFIG =======
CSV_PATH = r"test_interpolated.csv"
VIDEO_PATH = r"sample_4.mp4"
OUTPUT_PATH = r"Project4/Aaqib_test.mp4"
# ======================


def parse_bbox(s):
    """Safely parse bbox strings like 'x1 x2 x3 x4' or '[x1, x2, x3, x4]'."""
    s = str(s).replace('[', '').replace(']', '').replace(',', ' ')
    nums = [float(x) for x in s.split() if x.strip() != '']
    if len(nums) != 4:
        return [0, 0, 0, 0]
    return nums


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10):
    """Draw fancy green corner borders."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


# Load data
results = pd.read_csv(CSV_PATH)
print(f"Loaded results: {len(results)} rows")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Precompute license plate crops
license_plate = {}
for car_id in np.unique(results['car_id']):
    df_car = results[results['car_id'] == car_id]
    if df_car['license_number_score'].max() == 0:
        continue
    best_row = df_car.loc[df_car['license_number_score'].idxmax()]
    frame_idx = int(best_row['frame_nmr'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue

    x1, y1, x2, y2 = map(int, parse_bbox(best_row['license_plate_bbox']))
    lp_crop = frame[y1:y2, x1:x2]
    if lp_crop.size == 0:
        continue

    # Resize LP crop for overlay
    h, w = lp_crop.shape[:2]
    scale_h = 80  # height of overlay crop
    new_w = int(w * (scale_h / h))
    lp_crop_resized = cv2.resize(lp_crop, (new_w, scale_h))

    license_plate[car_id] = {
        'crop': lp_crop_resized,
        'text': str(best_row['license_number'])
    }

print(f"Prepared {len(license_plate)} license plate crops.")

# Reset video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = -1

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    df_frame = results[results['frame_nmr'] == frame_idx]
    for _, row in df_frame.iterrows():
        car_id = int(row['car_id'])
        car_x1, car_y1, car_x2, car_y2 = map(int, parse_bbox(row['car_bbox']))
        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, parse_bbox(row['license_plate_bbox']))

        # Draw borders
        draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 6)
        cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 4)

        # Add overlay only if we have a valid crop
        if car_id in license_plate:
            lp_crop = license_plate[car_id]['crop']
            text = license_plate[car_id]['text']

            h, w = lp_crop.shape[:2]
            overlay_y = max(10, car_y1 - h - 40)  # shift up, clamp above frame
            overlay_x = max(10, int((car_x1 + car_x2 - w) / 2))
            overlay_x2 = min(width - 10, overlay_x + w)

            # Draw white background for plate number
            bg_y1 = overlay_y - 50
            bg_y2 = overlay_y - 10
            cv2.rectangle(frame, (overlay_x, bg_y1), (overlay_x2, bg_y2), (255, 255, 255), -1)

            # Paste LP crop
            frame[overlay_y:overlay_y + h, overlay_x:overlay_x2] = lp_crop[:, :overlay_x2 - overlay_x]

            # Write text on white box
            font_scale = 1.2
            thickness = 3
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = overlay_x + int((w - tw) / 2)
            text_y = bg_y2 - 15
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print(f"Annotated video saved to: {os.path.abspath(OUTPUT_PATH)}")
