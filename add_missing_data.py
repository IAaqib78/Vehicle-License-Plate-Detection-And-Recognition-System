# add_missing_data.py (robust, PyCharm-ready)
import os
import sys
import math
import csv
from typing import List, Any
import pandas as pd
import numpy as np

# -------------------------
# CONFIG - change only if needed
# -------------------------
INPUT_PATH = r"path_to_the_Generated CSV file"
OUTPUT_PATH = r"test_interpolated.csv"
# -------------------------

def debug_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def parse_bbox_field(val: Any) -> List[float]:
    """
    Accepts bbox strings like:
      "[x1 x2 x3 x4]"
      "[x1, x2, x3, x4]"
      "x1 x2 x3 x4"
      "x1,x2,x3,x4"
    Returns list of 4 floats [x1,y1,x2,y2], or [nan]*4 on failure.
    """
    if val is None:
        return [math.nan]*4
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return [math.nan]*4
    # remove brackets
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1].strip()
    # replace commas with spaces, collapse multiple spaces
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p.strip() != ""]
    if len(parts) < 4:
        # maybe file used comma separation without spaces - try split by comma
        parts = s.split(",")
        parts = [p.strip() for p in parts if p.strip() != ""]
    try:
        nums = [float(parts[i]) for i in range(4)]
        return nums
    except Exception:
        return [math.nan]*4

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def load_input(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    debug_print(f"Loaded input CSV: {path}  (rows={len(df)})")
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    expected = {'frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox'}
    present = set(df.columns)
    if not expected.issubset(present):
        debug_print("Warning: input CSV missing some expected columns.")
        debug_print("Present columns:", present)
    # force frame_nmr and car_id to numeric where possible
    df['frame_nmr_raw'] = df.get('frame_nmr', df.get('frame_number', df.columns[0]))
    df['frame_nmr'] = df['frame_nmr_raw'].apply(safe_int)
    if df['frame_nmr'].isnull().any():
        debug_print("Warning: some frame_nmr values could not be parsed to int. They will be dropped.")
    df = df.dropna(subset=['frame_nmr']).copy()
    df['frame_nmr'] = df['frame_nmr'].astype(int)
    df['car_id_raw'] = df.get('car_id', "")
    df['car_id'] = df['car_id_raw'].apply(lambda x: safe_int(x) if str(x).strip() != "" else None)
    df = df.dropna(subset=['car_id']).copy()
    df['car_id'] = df['car_id'].astype(int)
    # parse bbox columns
    df['car_bbox_parsed'] = df.get('car_bbox', "").apply(parse_bbox_field)
    df['lp_bbox_parsed'] = df.get('license_plate_bbox', "").apply(parse_bbox_field)
    return df

def interpolate_for_one_track(frames: np.ndarray, boxes: np.ndarray, full_frames: np.ndarray) -> np.ndarray:
    """
    frames: 1D array of existing frame indices (n,)
    boxes: (n,4) array
    full_frames: 1D array of full_frames to produce (m,)
    returns: (m,4) array interpolated (float)
    """
    # find valid rows (non-NaN)
    valid = ~np.isnan(boxes).any(axis=1)
    if valid.sum() == 0:
        # nothing valid, return NaNs
        return np.full((len(full_frames), boxes.shape[1]), np.nan)
    if valid.sum() == 1:
        # single observation — replicate it across full_frames
        single = boxes[valid][0]
        return np.tile(single, (len(full_frames), 1))
    # use numpy.interp per column (safe, simple)
    out = np.zeros((len(full_frames), boxes.shape[1]), dtype=float)
    for col in range(boxes.shape[1]):
        xp = frames[valid]
        fp = boxes[valid, col]
        out[:, col] = np.interp(full_frames, xp, fp)
    return out

def main():
    debug_print("Starting interpolation script...")
    df = load_input(INPUT_PATH)

    # Group by car_id
    output_rows = []
    total_imputed = 0
    total_written = 0

    car_ids = sorted(df['car_id'].unique())
    debug_print(f"Found {len(car_ids)} unique car_id(s): {car_ids}")

    for cid in car_ids:
        g = df[df['car_id'] == cid].sort_values('frame_nmr')
        frames = g['frame_nmr'].to_numpy(dtype=int)
        if len(frames) == 0:
            continue
        first, last = frames[0], frames[-1]
        full_frames = np.arange(first, last + 1, dtype=int)
        # prepare boxes arrays
        car_boxes = np.vstack(g['car_bbox_parsed'].to_numpy())
        lp_boxes = np.vstack(g['lp_bbox_parsed'].to_numpy())
        # interpolate
        car_interp = interpolate_for_one_track(frames, car_boxes, full_frames)
        lp_interp = interpolate_for_one_track(frames, lp_boxes, full_frames)

        # for each frame in full_frames create a row
        for idx, f in enumerate(full_frames):
            row = {}
            row['frame_nmr'] = int(f)
            row['car_id'] = int(cid)
            # write interpolated bboxes as space-separated strings
            cb = car_interp[idx]
            lb = lp_interp[idx]
            row['car_bbox'] = " ".join([f"{float(x):.3f}" for x in cb])
            row['license_plate_bbox'] = " ".join([f"{float(x):.3f}" for x in lb])

            # check if this frame existed in original group
            existed = (g['frame_nmr'] == f).any()
            if not existed:
                total_imputed += 1
                row['license_plate_bbox_score'] = 0
                row['license_number'] = 0
                row['license_number_score'] = 0
            else:
                # copy available meta fields if present, else default
                orig_row = g[g['frame_nmr'] == f].iloc[0].to_dict()
                row['license_plate_bbox_score'] = orig_row.get('license_plate_bbox_score', 0)
                row['license_number'] = orig_row.get('license_number', orig_row.get('license_number', 0))
                row['license_number_score'] = orig_row.get('license_number_score', 0)
            output_rows.append(row)
            total_written += 1

    # Always write output file (even if no imputation)
    if len(output_rows) == 0:
        debug_print("No valid rows found after parsing input. Writing an empty CSV with headers.")
        header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
    else:
        # write DataFrame
        out_df = pd.DataFrame(output_rows)
        # ensure column order
        cols = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
        for c in cols:
            if c not in out_df.columns:
                out_df[c] = 0
        out_df = out_df[cols]
        out_df.to_csv(OUTPUT_PATH, index=False)

    debug_print(f"Interpolation finished. Wrote {total_written} rows (imputed {total_imputed} frames).")
    debug_print(f"Output file: {os.path.abspath(OUTPUT_PATH)}")
    return 0

if __name__ == "__main__":
    main()
