import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
from segment_anything import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("SAM vs MedSAM Image Segmentation")

# ─── Config & Helpers ───────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAM_CKPT    = os.path.join(BASE_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
MEDSAM_CKPT = os.path.join(BASE_DIR, "checkpoints", "medsam_vit_b.pth")
MODEL_TYPE  = "vit_b"

@st.cache_resource
def load_predictor(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)
    model = sam_model_registry[MODEL_TYPE](checkpoint=None)
    model.load_state_dict(state_dict)
    model.to(device)
    return SamPredictor(model)

sam_predictor    = load_predictor(SAM_CKPT)
medsam_predictor = load_predictor(MEDSAM_CKPT)

def run_segmentation(predictor, pil_img, clicks):
    img_np = np.array(pil_img)
    predictor.set_image(img_np)
    (x1r, y1r), (x2r, y2r) = clicks
    x0, x1 = sorted([x1r, x2r])
    y0, y1 = sorted([y1r, y2r])
    box = np.array([[x0, y0, x1, y1]])
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    return masks[0].astype(np.uint8)

def overlay_mask(pil_img, mask):
    h, w = mask.shape
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
    overlay_arr[..., 2] = 255            # Blue
    overlay_arr[..., 3] = mask * 100     # Alpha
    overlay = Image.fromarray(overlay_arr)
    base = pil_img.convert("RGBA")
    return Image.alpha_composite(base, overlay)

# ─── Callback functions ────────────────────────────────────────────────────────
def show_box_callback():
    st.session_state.show_box = True

def run_callback():
    predictor = sam_predictor if st.session_state.model == "SAM" else medsam_predictor
    mask = run_segmentation(predictor, st.session_state.image, st.session_state.clicks)
    st.session_state.result = overlay_mask(st.session_state.image, mask)
    st.session_state.ran = True

def refresh_callback():
    for key in ["image", "clicks", "model", "ran", "result", "show_box"]:
        st.session_state[key] = None if key == "image" else [] if key == "clicks" else False if key in ("ran","show_box") else None

# ─── Session state initialization ──────────────────────────────────────────────
if "image" not in st.session_state:  st.session_state.image    = None
if "clicks" not in st.session_state: st.session_state.clicks   = []
if "model" not in st.session_state:  st.session_state.model    = None
if "ran" not in st.session_state:    st.session_state.ran      = False
if "result" not in st.session_state: st.session_state.result   = None
if "show_box" not in st.session_state: st.session_state.show_box = False

# ─── Upload step ───────────────────────────────────────────────────────────────
if st.session_state.image is None:
    uploaded = st.file_uploader(
        "📤 Attach an image to run SAM or MedSAM on it",
        type=["jpg", "png", "jpeg"],
    )
    if uploaded:
        st.session_state.image = Image.open(uploaded).convert("RGB")

# ─── Main interface ────────────────────────────────────────────────────────────
if st.session_state.image:
    img = st.session_state.image
    left_col, right_col = st.columns([2, 1])

    placeholder = left_col.empty()
    with placeholder:
        if len(st.session_state.clicks) < 2 and not st.session_state.ran:
            coords = streamlit_image_coordinates(img, key="img_clicks")
            if coords:
                st.session_state.clicks.append((coords["x"], coords["y"]))

        elif len(st.session_state.clicks) == 2 and not st.session_state.show_box and not st.session_state.ran:
            st.image(img, use_container_width=True)

        elif len(st.session_state.clicks) == 2 and st.session_state.show_box and not st.session_state.ran:
            (x1r, y1r), (x2r, y2r) = st.session_state.clicks
            x0, x1 = sorted([x1r, x2r]); y0, y1 = sorted([y1r, y2r])
            disp = img.copy()
            draw = ImageDraw.Draw(disp, "RGBA")
            draw.rectangle([(x0, y0), (x1, y1)],
                           outline=(255,255,0,255), width=3,
                           fill=(255,255,0,63))
            st.image(disp, use_container_width=True)

        elif st.session_state.ran:
            st.image(st.session_state.result, use_container_width=True)

    with right_col:
        if len(st.session_state.clicks) == 0:
            st.markdown("**Select the first corner of the box**")

        elif len(st.session_state.clicks) == 1 and not st.session_state.ran:
            st.markdown("**Select the opposite corner of the box**")

        elif len(st.session_state.clicks) == 2 and not st.session_state.show_box and not st.session_state.ran:
            st.button("Show box", on_click=show_box_callback)

        elif len(st.session_state.clicks) == 2 and st.session_state.show_box and not st.session_state.ran:
            st.markdown("**Do you want to run SAM or MedSAM?**")
            st.radio("", ["SAM", "MedSAM"], key="model")
            st.button("▶️ RUN", on_click=run_callback)

        elif st.session_state.ran:
            st.button("Refresh", on_click=refresh_callback)
