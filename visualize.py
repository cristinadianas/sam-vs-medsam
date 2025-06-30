import os
import json
import numpy as np
import nibabel as nib
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import binary_erosion, distance_transform_edt
from segment_anything import sam_model_registry, SamPredictor

# ─────────────── Settings ───────────────
TASK_NAME           = "Task02_Heart"
IMAGE_INDEX         = 3
SLICE_INDEX         = 50
BBOX_ENLARGE_PIXELS = 0

DEFAULT_LABELS = {
    "Task01_BrainTumour"  : [1,2,3],
    "Task02_Heart"        : [1],
    "Task03_Liver"        : [2],
    "Task04_Hippocampus"  : [1,2],
    "Task05_Prostate"     : [1,2],
    "Task06_Lung"         : [1],
    "Task07_Pancreas"     : [2],
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
DATA_PATH   = os.path.join(BASE_DIR, "msd", TASK_NAME)
IM_DIR      = os.path.join(DATA_PATH, "imagesTr")
LB_DIR      = os.path.join(DATA_PATH, "labelsTr")
SAM_CKPT    = os.path.join(BASE_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
MEDSAM_CKPT = os.path.join(BASE_DIR, "checkpoints", "medsam_vit_b.pth")

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

# ───────── Read dataset.json ─────────
with open(os.path.join(DATA_PATH, "dataset.json"), 'r') as f:
    meta = json.load(f)
modality_to_idx = { name:int(idx) for idx,name in meta.get("modality",{}).items() }

# ───────── Load predictors ─────────
sam    = load_predictor(SAM_CKPT)
medsam = load_predictor(MEDSAM_CKPT)

# ───────── Load the chosen volume ─────────
target_suffix_1 = f"_{IMAGE_INDEX}.nii.gz"
target_suffix_2 = f"_{IMAGE_INDEX:02d}.nii.gz"
target_suffix_3 = f"_{IMAGE_INDEX:03d}.nii.gz"
matching_files = [f for f in os.listdir(IM_DIR) if (f.endswith(target_suffix_3) or f.endswith(target_suffix_2) or f.endswith(target_suffix_1))]
if not matching_files:
    raise FileNotFoundError(f"No file in '{IM_DIR}' ends with '{IMAGE_INDEX}'")
file_name = matching_files[0]

scan_vol    = nib.load(os.path.join(IM_DIR, file_name)).get_fdata()
label_vol   = nib.load(os.path.join(LB_DIR, file_name)).get_fdata()

# ───────── Extract 2D slice ─────────
if scan_vol.ndim == 4:
    ch = modality_to_idx[MODALITY_NAME]
    img = scan_vol[:, :, SLICE_INDEX, ch]
else:
    img = scan_vol[:, :, SLICE_INDEX]
label = label_vol[:, :, SLICE_INDEX].astype(np.uint8)

# ───────── Normalize & convert to RGB ─────────
norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
u8   = (norm * 255).astype(np.uint8)
rgb  = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)

# ───────── Ground-truth mask & bbox ─────────
gt = np.isin(label, LABELS_OF_INTEREST).astype(np.uint8)
ys, xs = np.where(gt)
if xs.size == 0:
    raise ValueError(f"No labels {LABELS_OF_INTEREST} in slice {SLICE_INDEX}")
box = np.array([
    max(0, xs.min() - BBOX_ENLARGE_PIXELS),
    max(0, ys.min() - BBOX_ENLARGE_PIXELS),
    min(gt.shape[1] - 1, xs.max() + BBOX_ENLARGE_PIXELS),
    min(gt.shape[0] - 1, ys.max() + BBOX_ENLARGE_PIXELS)
])

# ───────── Run SAM & MedSAM ─────────
sam.set_image(rgb)
sam_mask, _, _ = sam.predict(box=box[None,:], multimask_output=False)
sam_mask = sam_mask[0].astype(np.uint8)

medsam.set_image(rgb)
med_mask, _, _ = medsam.predict(box=box[None,:], multimask_output=False)
med_mask = med_mask[0].astype(np.uint8)

# ───────── Compute Dice & NSD scores ─────────
dice_sam = dice(sam_mask, gt)
dice_med = dice(med_mask, gt)
nsd_sam = normalized_surface_dice(sam_mask, gt, tolerance=1.0)
nsd_med = normalized_surface_dice(med_mask, gt, tolerance=1.0)

# ───────── Plotting ─────────
sam_rgba = (0,1,0,0.4)
med_rgba = (0,1,1,0.4)
base_colors = [(1,0,0,0.4), (0,1,0,0.4), (0,0,1,0.4), (1,1,0,0.4), (1,0,1,0.4), (0,1,1,0.4)]
gt_palette = {lab: base_colors[i % len(base_colors)] for i, lab in enumerate(LABELS_OF_INTEREST)}
gt_single_rgba = (1,0,0,0.4)

fig, axes = plt.subplots(2, 3, figsize=(18,12))
axes = axes.flatten()

def make_bbox_patch(box, color='yellow', lw=2):
    return patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             linewidth=lw, edgecolor=color, facecolor='none')

# 1) Original + bbox
ax = axes[0]
ax.imshow(rgb)
ax.add_patch(make_bbox_patch(box))
ax.set_title(f"{TASK_NAME} | {MODALITY_NAME} Scan | Labels {LABELS_OF_INTEREST}\n{file_name} Slice {SLICE_INDEX}")
ax.axis('off')

# 2) SAM + bbox + Dice/NSD
ax = axes[1]
ax.imshow(rgb)
sam_overlay = np.zeros((*sam_mask.shape, 4))
sam_overlay[..., :3] = sam_rgba[:3]
sam_overlay[..., 3] = sam_mask * sam_rgba[3]
ax.imshow(sam_overlay)
ax.add_patch(make_bbox_patch(box))
ax.set_title(f"SAM Prediction\nDSC = {dice_sam:.3f} | NSD = {nsd_sam:.3f}")
ax.axis('off')

# 3) MedSAM + bbox + Dice/NSD
ax = axes[2]
ax.imshow(rgb)
med_overlay = np.zeros((*med_mask.shape, 4))
med_overlay[..., :3] = med_rgba[:3]
med_overlay[..., 3] = med_mask * med_rgba[3]
ax.imshow(med_overlay)
ax.add_patch(make_bbox_patch(box))
ax.set_title(f"MedSAM Prediction\nDSC = {dice_med:.3f} | NSD = {nsd_med:.3f}")
ax.axis('off')

# 4) Ground Truth
ax = axes[3]
ax.imshow(rgb, cmap='gray')
combined_mask = np.isin(label, LABELS_OF_INTEREST).astype(np.uint8)
colored = np.zeros((*label.shape, 4))
colored[..., :3] = (1, 0, 0)
colored[..., 3] = combined_mask * 0.4
ax.imshow(colored)
ax.set_title("Ground Truth")
ax.axis('off')

# 5) SAM ∩ GT
ax = axes[4]
ax.imshow(rgb)
ax.imshow(sam_overlay)
gt_overlay = np.zeros((*gt.shape, 4))
gt_overlay[..., :3] = gt_single_rgba[:3]
gt_overlay[..., 3] = gt * gt_single_rgba[3]
ax.imshow(gt_overlay)
ax.set_title("SAM ∩ Ground Truth")
ax.axis('off')

# 6) MedSAM ∩ GT
ax = axes[5]
ax.imshow(rgb)
ax.imshow(med_overlay)
ax.imshow(gt_overlay)
ax.set_title("MedSAM ∩ Ground Truth")
ax.axis('off')

plt.tight_layout()
plt.show()
