import os
import json
import numpy as np
import nibabel as nib
import cv2
import torch
import csv
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import binary_erosion, distance_transform_edt
from segment_anything import sam_model_registry, SamPredictor

# ─────────────── Settings ───────────────
TEST_MODE = True

if TEST_MODE:
    TASK_NAME           = "Task02_Heart"
    BBOX_ENLARGE_PIXELS = 0
    NUM_TEST_SCANS      = 1
    LIMIT               = None
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., Task01_BrainTumour)")
    parser.add_argument("--bbox", type=int, default=0, help="Bounding box enlargement in pixels (optional)")
    parser.add_argument("--limit", type=int, help="Maximum iterations (optional)")
    args = parser.parse_args()

    TASK_NAME           = args.task
    BBOX_ENLARGE_PIXELS = args.bbox
    LIMIT               = args.limit

DEFAULT_LABELS = {
    "Task01_BrainTumour"  : [1,2,3],
    "Task02_Heart"        : [1],
    "Task03_Liver"        : [1],
    "Task04_Hippocampus"  : [1,2],
    "Task05_Prostate"     : [1,2],
    "Task06_Lung"         : [1],
    "Task07_Pancreas"     : [1,2],
    "Task08_HepaticVessel": [1,2],
    "Task09_Spleen"       : [1],
    "Task10_Colon"        : [1],
}
LABELS_OF_INTEREST = DEFAULT_LABELS[TASK_NAME]

MODALITY_MAP = {
    "Task01_BrainTumour"  : "FLAIR",
    "Task02_Heart"        : "MRI",
    "Task03_Liver"        : "CT",
    "Task04_Hippocampus"  : "MRI",
    "Task05_Prostate"     : "T2",
    "Task06_Lung"         : "CT",
    "Task07_Pancreas"     : "CT",
    "Task08_HepaticVessel": "CT",
    "Task09_Spleen"       : "CT",
    "Task10_Colon"        : "CT"
}
MODALITY_NAME = MODALITY_MAP[TASK_NAME]

# ─────────────── Paths ───────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LOG_PATH    = os.path.join(BASE_DIR, "benchmark_times.txt")
DATA_PATH   = os.path.join(BASE_DIR, "msd", TASK_NAME)
CSV_NAME    = f"{TASK_NAME}_Labels{LABELS_OF_INTEREST}_Bbox{BBOX_ENLARGE_PIXELS}_Results.csv"
CSV_PATH    = os.path.join(BASE_DIR, "results", CSV_NAME)
IM_DIR      = os.path.join(DATA_PATH, "imagesTr")
LB_DIR      = os.path.join(DATA_PATH, "labelsTr")
SAM_CKPT    = os.path.join(BASE_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
MEDSAM_CKPT = os.path.join(BASE_DIR, "checkpoints", "medsam_vit_b.pth")

# ─────────── Starting message ───────────
if TEST_MODE:
    print("\nTEST MODE")
bbox_note = f" | BBox Enlargement: {BBOX_ENLARGE_PIXELS}px" if BBOX_ENLARGE_PIXELS > 0 else ""
print(f"\n✨✨✨ Running benchmark: {TASK_NAME} | Modality: {MODALITY_NAME} | Labels: {LABELS_OF_INTEREST}{bbox_note} ✨✨✨")

# ─────────────── Helpers ───────────────
MODEL_TYPE = "vit_b"

def load_predictor(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model = sam_model_registry[MODEL_TYPE](checkpoint=None)
    model.load_state_dict(state_dict)
    model.to(device)
    return SamPredictor(model)

def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    return 2 * inter / (p.sum() + g.sum() + 1e-8)

def get_surface(mask):
    return mask ^ binary_erosion(mask)

def normalized_surface_dice(pred, gt, tolerance=1.0):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not np.any(pred) or not np.any(gt):
        return 0.0

    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    pred_to_gt = dt_gt[pred_surface]
    gt_to_pred = dt_pred[gt_surface]

    pred_within = np.sum(pred_to_gt <= tolerance)
    gt_within = np.sum(gt_to_pred <= tolerance)
    total = pred_surface.sum() + gt_surface.sum()

    return (pred_within + gt_within) / total

def append_rows(rows):
    mode = "a" if os.path.exists(CSV_PATH) else "w"
    with open(CSV_PATH, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "slice_idx", "modality", 
            "dice_sam", "nsd_sam", 
            "dice_medsam", "nsd_medsam",
            "sam_pixels", "medsam_pixels",
            "foreground_pixels", "total_pixels",
            "labels_present", "slice_time_sec"
        ])
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)

# ───────── Read dataset.json ─────────
with open(os.path.join(DATA_PATH, "dataset.json"), 'r') as f:
    meta = json.load(f)
modality_to_idx = { name:int(idx) for idx,name in meta.get("modality",{}).items() }

# ────────── Load predictors ──────────
if not torch.cuda.is_available():
    _old = torch.load
    torch.load = lambda f, *a, **k: _old(f, map_location="cpu")
    
sam    = load_predictor(SAM_CKPT)
medsam = load_predictor(MEDSAM_CKPT)

# ──────────── Load files ────────────
ALL_FILES = sorted(f for f in os.listdir(IM_DIR) if f.endswith(".nii.gz"))

if TEST_MODE:
    ALL_FILES = ALL_FILES[:NUM_TEST_SCANS]
elif LIMIT:
    ALL_FILES = ALL_FILES[:LIMIT]

# ───────── Load seen records ─────────
seen = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["image_id"], int(row["slice_idx"]), row["modality"])
            seen.add(key)

results_buffer = []
start_time = datetime.now()
start = time.time()

# ─────────── Progress bar ───────────
slice_counts = [nib.load(os.path.join(LB_DIR, f)).shape[2] for f in ALL_FILES]
total_slices = sum(slice_counts)

with tqdm(total=total_slices, desc="Slices") as pbar:

    # ─────────── Load 3D scans ───────────
    for fname in ALL_FILES:
        image_id = fname.replace(".nii.gz", "")
        try:
            label3d = nib.load(os.path.join(LB_DIR, fname)).get_fdata()
            scan3d = nib.load(os.path.join(IM_DIR, fname)).get_fdata()

            # ─────────── Iterate through slices & Update progress bar ───────────
            for sl in range(label3d.shape[2]):
                pbar.update(1)
                slice_start = time.perf_counter()

                # ─────────── Skip if not of interest ───────────
                label = label3d[:, :, sl].astype(np.uint8)
                if not np.isin(label, LABELS_OF_INTEREST).any():
                    continue

                # ───────── Ground-truth mask & Bounding bbox ─────────
                gt = np.isin(label, LABELS_OF_INTEREST).astype(np.uint8)
                ys, xs = np.where(gt)
                if xs.size == 0:
                    continue
                box = np.array([
                    max(0, xs.min() - BBOX_ENLARGE_PIXELS),
                    max(0, ys.min() - BBOX_ENLARGE_PIXELS),
                    min(gt.shape[1] - 1, xs.max() + BBOX_ENLARGE_PIXELS),
                    min(gt.shape[0] - 1, ys.max() + BBOX_ENLARGE_PIXELS)
                ])

                # ─────────── Skip if seen ───────────
                key = (image_id, sl, MODALITY_NAME)
                if key in seen:
                    continue

                # ───────── Extract 2D slice ─────────
                if scan3d.ndim == 4:
                    idx = modality_to_idx[MODALITY_NAME]
                    img = scan3d[:, :, sl, idx]
                else:
                    img = scan3d[:, :, sl]
                
                # ──────── Normalize & convert to RGB ────────
                norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                u8 = (norm * 255).astype(np.uint8)
                rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)

                # ───────── Run SAM & MedSAM ─────────
                sam.set_image(rgb)
                sam_mask, _, _ = sam.predict(box=box[None, :], multimask_output=False)
                sam_mask = sam_mask[0].astype(np.uint8)

                medsam.set_image(rgb)
                med_mask, _, _ = medsam.predict(box=box[None, :], multimask_output=False)
                med_mask = med_mask[0].astype(np.uint8)

                # ───────── Compute Dice & NSD scores ─────────
                dice_sam = round(dice(sam_mask, gt), 4)
                dice_medsam = round(dice(med_mask, gt), 4)
                nsd_sam = round(normalized_surface_dice(sam_mask, gt, tolerance=1.5), 4)
                nsd_medsam = round(normalized_surface_dice(med_mask, gt, tolerance=1.5), 4)

                # ───────── Compute duration ─────────
                slice_duration = round(time.perf_counter() - slice_start, 4)

                # ────────── Add results to buffer ──────────
                row = {
                    "image_id": image_id,
                    "slice_idx": sl,
                    "modality": MODALITY_NAME,
                    "dice_sam": dice_sam,
                    "nsd_sam": nsd_sam,
                    "dice_medsam": dice_medsam,
                    "nsd_medsam": nsd_medsam,
                    "sam_pixels": int(sam_mask.sum()),
                    "medsam_pixels": int(med_mask.sum()),
                    "foreground_pixels": int(gt.sum()), 
                    "total_pixels": int(gt.size),
                    "labels_present": str(sorted(np.unique(label[gt.astype(bool)]).tolist())),
                    "slice_time_sec": slice_duration
                }
                results_buffer.append(row)
                seen.add(key)

            # ─────────── Save results ───────────
            if results_buffer:
                append_rows(results_buffer)
                results_buffer = []

        # ─────────── Error message ───────────
        except Exception as e:
            print(f"\nError on {image_id}: {e}")

    # ─────────── Save results ───────────
    if results_buffer:
        append_rows(results_buffer)

# ─────────── Log duration ───────────
end_time = datetime.now()
end = time.time()
total_duration_sec = round(end - start, 2)

log_message = (
    f"Task: {TASK_NAME} "
    f"| Modality: {MODALITY_NAME} "
    f"| Labels: {LABELS_OF_INTEREST} "
    f"{f'| BBox Enlargement: {BBOX_ENLARGE_PIXELS}px ' if BBOX_ENLARGE_PIXELS > 0 else ''}"
    f"{f'| Limit: {LIMIT} ' if LIMIT is not None else ''}"
    f"| Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
    f"| End: {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
    f"| Duration: {total_duration_sec} seconds\n"
)

with open(LOG_PATH, "a") as f:
    f.write(log_message)

# ─────────── Success message ───────────
print(f"Results saved to {CSV_PATH}\n")
