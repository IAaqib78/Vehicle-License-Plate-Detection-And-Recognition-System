"""
ui_streamlit.py
Cinematic Streamlit frontend for your VLPR system.


Notes:
- This app uses The existing utils.py and sort/sort.py.
- Change COCO_MODEL_PATH and LP_MODEL_PATH below if needed.
"""

import os
import time
import tempfile
import threading
from pathlib import Path
from typing import Tuple, List, Dict, Any

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Import your helpers (must be in project)
from utils import read_license_plate, write_csv
from sort.sort import Sort  # uses your existing SORT implementation

# ---------------------------
# Configuration (edit if needed)
# ---------------------------
COCO_MODEL_PATH = r"Aaqib/yolov8n.pt"
LP_MODEL_PATH = r" Aaqib/best.pt"
DEFAULT_VIDEO = r" sample_4.mp4"
OUTPUT_DIR = Path.cwd() / "VLPRs_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# UI color palette
NAVY = "#0A1F44"
CYAN = "#00FFFF"
CRIMSON = "#DC143C"
SILVER = "#C0C0C0"

# ---------------------------
# Helper drawing utilities
# ---------------------------
def hex_to_bgr(hex_code: str):
    h = hex_code.lstrip('#')
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (b, g, r)

COL_NAVY = hex_to_bgr(NAVY)
COL_CYAN = hex_to_bgr(CYAN)
COL_CRIMSON = hex_to_bgr(CRIMSON)
COL_SILVER = hex_to_bgr(SILVER)

def draw_vehicle_border(img: np.ndarray, bbox: Tuple[int,int,int,int], thickness: int = 3):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), COL_NAVY, thickness)

def draw_plate_box(img: np.ndarray, bbox: Tuple[int,int,int,int], thickness: int = 2):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), COL_CRIMSON, thickness)

def paste_plate_crop_with_text(img: np.ndarray, lp_crop: np.ndarray, car_bbox: Tuple[int,int,int,int], text: str):
    """
    Paste lp_crop above the car bbox (if space), otherwise below; draw white background and text
    """
    h, w = lp_crop.shape[:2]
    car_x1, car_y1, car_x2, car_y2 = car_bbox
    center_x = int((car_x1 + car_x2) / 2)
    overlay_x1 = max(0, center_x - w//2)
    overlay_x2 = overlay_x1 + w
    # clamp horizontally
    if overlay_x2 > img.shape[1]:
        overlay_x2 = img.shape[1] - 1
        overlay_x1 = overlay_x2 - w
        if overlay_x1 < 0:
            overlay_x1 = 0; overlay_x2 = min(w, img.shape[1]-1)
            lp_crop = cv2.resize(lp_crop, (overlay_x2-overlay_x1, max(10, int(h * (overlay_x2-overlay_x1)/w))))
            h, w = lp_crop.shape[:2]
    # preferred position above car
    top_y = car_y1 - h - 12
    if top_y < 6:
        # not enough space above — place below car
        top_y = car_y2 + 12
        # if below goes out of frame, clamp to bottom area
        if top_y + h + 40 > img.shape[0]:
            top_y = max(6, img.shape[0] - h - 60)
    bottom_y = top_y + h

    # white background for text above or below crop
    bg_top = top_y - 36
    bg_bottom = top_y - 6
    if bg_top < 0:
        bg_top = top_y + h + 6
        bg_bottom = bg_top + 36
    # ensure bg within frame
    bg_top = max(0, bg_top)
    bg_bottom = min(img.shape[0]-1, bg_bottom)
    # paste crop
    try:
        img[top_y:bottom_y, overlay_x1:overlay_x1+w] = lp_crop
    except Exception:
        # try resizing to fit
        w2 = min(w, img.shape[1]-overlay_x1-1)
        if w2 <= 0: return
        lp_small = cv2.resize(lp_crop, (w2, h))
        img[top_y:top_y+lp_small.shape[0], overlay_x1:overlay_x1+w2] = lp_small
    # draw white background rectangle for text
    cv2.rectangle(img, (overlay_x1, bg_top), (overlay_x1 + w, bg_bottom), (255,255,255), -1)
    # put bold black text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    txt_x = overlay_x1 + max(2, int((w - tw) / 2))
    txt_y = bg_bottom - 8
    cv2.putText(img, text, (txt_x, txt_y), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

# ---------------------------
# Model loading (single load)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models(coco_path: str, lp_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load YOLO wrappers
    coco = YOLO(coco_path)
    lp = YOLO(lp_path)
    return coco, lp, device

# ---------------------------
# Processing function (frame-by-frame)
# ---------------------------
def process_frame(frame: np.ndarray, coco_model, lp_model, tracker: Sort, device: str,
                  conf_thresh: float = 0.3) -> Tuple[np.ndarray, List[Dict]]:
    """
    Runs detection on a single frame, updates tracker, performs OCR for any plate detection,
    annotates the frame and returns annotated frame and list of plates info for CSV.
    """
    # ensure frame is BGR numpy array
    out_frame = frame.copy()
    # YOLO inference (batch size 1)
    try:
        res1 = coco_model(frame, verbose=False)[0]
    except Exception as e:
        # fallback to predict API
        res1 = coco_model.predict(source=frame)[0]
    # parse detections for vehicles
    vehicle_boxes = []
    for b in res1.boxes.data.tolist():
        x1,y1,x2,y2,score,cls = b
        if score < conf_thresh: continue
        # keep vehicle classes from COCO (car, motorcycle, bus, truck)
        if int(cls) in [2,3,5,7]:
            vehicle_boxes.append([float(x1),float(y1),float(x2),float(y2),float(score)])

    # track vehicles
    trks = tracker.update(np.array(vehicle_boxes) if len(vehicle_boxes)>0 else np.empty((0,5)))
    detections_for_annot = []
    for tr in trks:
        # tr: x1,y1,x2,y2,track_id
        x1,y1,x2,y2,tid = tr[:5]
        detections_for_annot.append({'bbox':[int(x1),int(y1),int(x2),int(y2)], 'track_id':int(tid)})

    # plates detection (on a separate LP model)
    try:
        res2 = lp_model(frame, verbose=False)[0]
    except Exception:
        res2 = lp_model.predict(source=frame)[0]

    plate_infos = []
    for b in res2.boxes.data.tolist():
        x1,y1,x2,y2,score,cls = b
        if score < conf_thresh: continue
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        # crop plate region
        # clamp coordinates
        H,W = frame.shape[:2]
        x1c = max(0, min(W-1, x1)); x2c = max(0, min(W-1, x2))
        y1c = max(0, min(H-1, y1)); y2c = max(0, min(H-1, y2))
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = frame[y1c:y2c, x1c:x2c].copy()
        text, conf = read_license_plate(crop)
        plate_infos.append({'bbox':[x1c,y1c,x2c,y2c], 'text': text or "", 'conf': float(conf or 0.0), 'crop': crop})

    # associate plates to nearest tracked vehicle by IoU
    def iou(a,b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
        ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
        iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
        inter = iw*ih
        areaA = max(1,(ax2-ax1)*(ay2-ay1)); areaB = max(1,(bx2-bx1)*(by2-by1))
        return inter / (areaA + areaB - inter)
    for p in plate_infos:
        best_tid = None; best_iou = 0.0
        for d in detections_for_annot:
            i = iou(p['bbox'], d['bbox'])
            if i > best_iou:
                best_iou = i; best_tid = d['track_id']
        p['track_id'] = best_tid

    # annotate frame
    for d in detections_for_annot:
        bbox = d['bbox']
        draw_vehicle_border(out_frame, tuple(bbox), thickness=2)
    for p in plate_infos:
        bbox = p['bbox']
        draw_plate_box(out_frame, tuple(bbox), thickness=2)
        # paste crop and text if plate associated to a track
        if p.get('track_id') is not None:
            # find car bbox for that track
            car_bbox = None
            for d in detections_for_annot:
                if d['track_id'] == p['track_id']:
                    car_bbox = d['bbox']; break
            if car_bbox is not None:
                # resize crop width to 120 px preserving aspect
                ch, cw = p['crop'].shape[:2]
                target_h = 64
                new_w = max(20, int(cw * (target_h/ch)))
                lp_small = cv2.resize(p['crop'], (new_w, target_h))
                paste_plate_crop_with_text(out_frame, lp_small, tuple(car_bbox), p['text'])

    # prepare rows for CSV
    csv_rows = []
    for p in plate_infos:
        csv_rows.append({
            'frame': int(time.time()*1000) % (10**9),  # ephemeral frame id if needed
            'track_id': int(p['track_id']) if p.get('track_id') is not None else -1,
            'plate_text': p['text'],
            'plate_conf': float(p['conf']),
            'x1': int(p['bbox'][0]), 'y1': int(p['bbox'][1]), 'x2': int(p['bbox'][2]), 'y2': int(p['bbox'][3])
        })

    return out_frame, csv_rows

# ---------------------------
# Streamlit UI layout
# ---------------------------
st.set_page_config(page_title="VLPRs — Cinematic", layout="wide")
st.markdown(f"""
    <style>
    body {{ background-color: {NAVY}; color: {SILVER}; font-family: 'Roboto', sans-serif; }}
    .stButton>button {{ background-color: {CYAN}; color: #000; font-weight: 700; }}
    .stDownloadButton>button {{ background-color: {CRIMSON}; color: #fff; font-weight: 700; }}
    .stProgress > div > div > div {{ background-color: {CRIMSON}; }}
    .css-1v3fvcr {{ background-color: {NAVY}; }}
    </style>
""", unsafe_allow_html=True)

# header
st.markdown(f"# <span style='color:{CYAN}'>VLPR</span><span style='color:{SILVER}'> — Cinematic</span>", unsafe_allow_html=True)
st.markdown("### Real-time Vehicle License Plate Recognition — upload video, image, or open webcam")

col_left, col_right = st.columns([2,1])

with col_left:
    # Upload controls
    source = st.radio("Input source", ["Upload video", "Upload image", "Webcam"], index=0)
    uploaded_file = None
    if source == "Upload video":
        uploaded_file = st.file_uploader("Upload video (mp4/avi/mkv)", type=['mp4','avi','mkv'], accept_multiple_files=False)
    elif source == "Upload image":
        uploaded_file = st.file_uploader("Upload image (jpg,png,jpeg)", type=['jpg','jpeg','png'], accept_multiple_files=False)
    elif source == "Webcam":
        st.info("Webcam will stream from the machine hosting Streamlit. Click Start to begin live detection.")

    st.write("Model weights (you can change paths):")
    coco_path = st.text_input("Vehicle model path", COCO_MODEL_PATH)
    lp_path = st.text_input("License plate model path", LP_MODEL_PATH)

    # detection controls
    start = st.button("Start Detection")
    stop = st.button("Stop Detection")
    st.write("Confidence threshold:")
    conf_th = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)

with col_right:
    st.markdown("### Outputs")
    output_video_slot = st.empty()
    output_image_slot = st.empty()
    csv_slot = st.empty()
    progress_bar = st.progress(0)
    status = st.empty()

# Session state for background loop
if 'running' not in st.session_state:
    st.session_state['running'] = False
if 'thread' not in st.session_state:
    st.session_state['thread'] = None
if 'results_rows' not in st.session_state:
    st.session_state['results_rows'] = []

# load models (deferred)
try:
    coco_model, lp_model, device = load_models(coco_path, lp_path)
except Exception as e:
    st.error(f"Model load failed: {e}")
    coco_model = lp_model = None
    device = 'cpu'

# worker function for video/image/webcam processing
def detection_worker(input_source: Any):
    st.session_state['results_rows'] = []
    tracker = Sort()
    # output video writer setup for uploaded video
    tmp_out = OUTPUT_DIR / f"annotated_{int(time.time())}.mp4"
    vw = None
    cap = None
    frame_count = 0
    total_frames = None

    # initialize capture
    if isinstance(input_source, str) and os.path.exists(input_source):
        cap = cv2.VideoCapture(input_source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    elif isinstance(input_source, int):
        cap = cv2.VideoCapture(input_source)
        total_frames = None
    elif hasattr(input_source, "read"):
        # uploaded file-like object saved to disk path
        cap = cv2.VideoCapture(input_source.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    else:
        cap = None

    if cap is None or not cap.isOpened():
        status.error("Failed to open input source.")
        st.session_state['running'] = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(tmp_out), fourcc, fps, (width, height))

    last_frame_time = time.time()
    while st.session_state['running']:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # run inference & annotate
        annotated, rows = process_frame(frame, coco_model, lp_model, tracker, device, conf_thresh=conf_th)
        # write to video
        vw.write(annotated)
        # collect CSV rows
        st.session_state['results_rows'].extend(rows)
        # display annotated frame (downscale for speed)
        disp = cv2.cvtColor(cv2.resize(annotated, (min(960, width), int((min(960, width)/width)*height))), cv2.COLOR_BGR2RGB)
        output_image_slot.image(disp, use_column_width=True)
        # update progress
        if total_frames:
            progress = min(1.0, frame_count / total_frames)
            progress_bar.progress(int(progress * 100))
        else:
            # animate progress for webcam
            progress_bar.progress(int((time.time() % 1) * 100))
        # break if stopped by UI
        time.sleep(0.01)  # small sleep to allow UI refresh

    # cleanup
    vw.release()
    cap.release()

    # write CSV of aggregated results
    csv_path = OUTPUT_DIR / f"results_{int(time.time())}.csv"
    write_csv(_group_results_by_frame(st.session_state['results_rows']), str(csv_path))

    status.success(f"Done. Annotated video saved to {tmp_out}")
    output_video_slot.video(str(tmp_out))
    csv_slot.download_button("Download CSV", data=open(str(csv_path),'rb'), file_name=os.path.basename(csv_path))
    st.session_state['running'] = False

def _group_results_by_frame(rows: List[Dict]) -> Dict:
    """
    Convert list-of-rows to nested dict {frame: {track: {...}}} compatible with your write_csv
    Each row is: {'frame','track_id','plate_text','plate_conf','x1','y1','x2','y2'}
    """
    grouped = {}
    for r in rows:
        frame = r.get('frame', -1)
        tid = r.get('track_id', -1)
        if frame not in grouped:
            grouped[frame] = {}
        grouped[frame][tid] = {
            'car': {'bbox':[r.get('x1',0), r.get('y1',0), r.get('x2',0), r.get('y2',0)]},
            'license_plate': {
                'bbox':[r.get('x1',0), r.get('y1',0), r.get('x2',0), r.get('y2',0)],
                'bbox_score': r.get('plate_conf', 0),
                'text': r.get('plate_text',''),
                'text_score': r.get('plate_conf',0)
            }
        }
    return grouped

# UI button handlers
if start and not st.session_state['running']:
    st.session_state['running'] = True
    # prepare input source
    if source == "Upload video" and uploaded_file is not None:
        # save uploaded video to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tmp.write(uploaded_file.read()); tmp.flush()
        input_src = tmp.name
    elif source == "Upload image" and uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tmp.write(uploaded_file.read()); tmp.flush()
        # image: process single frame
        input_src = tmp.name
    elif source == "Webcam":
        input_src = 0
    else:
        st.error("Please provide an input or choose webcam.")
        st.session_state['running'] = False
        input_src = None

    if input_src is not None:
        # run worker in a thread so Streamlit UI remains responsive
        thread = threading.Thread(target=detection_worker, args=(input_src,), daemon=True)
        st.session_state['thread'] = thread
        thread.start()

if stop and st.session_state['running']:
    st.session_state['running'] = False
    status.info("Stopping...")

# Footer
st.markdown("---")
st.markdown("VLPRs — Cinematic UI · Electric cyan accents · Powered by YOLOv8 + SORT + EasyOCR/pytesseract")
