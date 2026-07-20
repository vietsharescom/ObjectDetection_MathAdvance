"""Streamlit demo — Person/Head detection, compares all 3 tracks side by side.

Usage:
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_ROOT = _DEMO_DIR.parent
_SRC = _ROOT / "src"
for p in [_SRC, _SRC / "track_a_yolo26", _SRC / "track_b_rfdetr", _SRC / "track_c_scratch"]:
    sys.path.insert(0, str(p))

import numpy as np
import streamlit as st
from PIL import Image

from common.dataset_paths import CLASS_NAMES  # noqa: E402

st.set_page_config(page_title="Person/Head Detection", layout="wide")
st.title("Person + Head Detection — 3 Tracks Comparison")
st.caption("Track A: YOLO26 (pretrained + fine-tuned) · Track B: RF-DETR (transformer, pretrained + fine-tuned) · Track C: CNN trained from scratch")

TRACK_A_WEIGHTS = _SRC / "track_a_yolo26" / "runs" / "yolo26_finetune" / "weights" / "best.pt"
TRACK_B_WEIGHTS = _SRC / "track_b_rfdetr" / "runs" / "rfdetr_finetune" / "checkpoint_best_total.pth"
TRACK_C_WEIGHTS = _SRC / "track_c_scratch" / "runs" / "scratch_detector.pt"


@st.cache_resource
def load_track_a():
    from ultralytics import YOLO
    return YOLO(str(TRACK_A_WEIGHTS))


@st.cache_resource
def load_track_b():
    from rfdetr import RFDETRNano
    return RFDETRNano.from_checkpoint(str(TRACK_B_WEIGHTS))


@st.cache_resource
def load_track_c():
    import torch
    from model import ScratchDetector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScratchDetector().to(device)
    model.load_state_dict(torch.load(TRACK_C_WEIGHTS, map_location=device))
    model.eval()
    return model, device


def run_track_a(image: Image.Image, conf: float):
    model = load_track_a()
    result = model.predict(np.array(image), conf=conf, verbose=False)[0]
    return Image.fromarray(result.plot()[:, :, ::-1]), len(result.boxes)


def run_track_b(image: Image.Image, conf: float):
    import supervision as sv
    model = load_track_b()
    detections = model.predict(image, threshold=conf)
    labels = [f"{CLASS_NAMES[c]} {s:.2f}" for c, s in zip(detections.class_id, detections.confidence)]
    annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
    return annotated, len(detections)


def run_track_c(image: Image.Image, conf: float):
    import cv2
    import torch
    from utils import decode_predictions, nms
    from predict import draw_boxes

    model, device = load_track_c()
    img_rgb = np.array(image)
    resized = cv2.resize(img_rgb, (128, 128))
    img_t = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        pred = model(img_t)[0]
    boxes = nms(decode_predictions(pred, grid_size=8, conf_thresh=conf), iou_thresh=0.4)
    annotated = draw_boxes(img_rgb, boxes)
    return Image.fromarray(annotated), len(boxes)


TRACKS = {
    "Track A -- YOLO26 (pretrained + fine-tuned)": (run_track_a, TRACK_A_WEIGHTS),
    "Track B -- RF-DETR (transformer, pretrained + fine-tuned)": (run_track_b, TRACK_B_WEIGHTS),
    "Track C -- CNN from scratch": (run_track_c, TRACK_C_WEIGHTS),
}

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.3, 0.05)
    selected = st.multiselect("Tracks to run", list(TRACKS.keys()), default=list(TRACKS.keys()))

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", width=400)

    cols = st.columns(len(selected)) if selected else []
    for col, name in zip(cols, selected):
        run_fn, weights_path = TRACKS[name]
        with col:
            st.subheader(name)
            if not Path(weights_path).exists():
                st.warning(f"Weights not found: `{weights_path}`\nTrain this track first.")
                continue
            with st.spinner("Running inference..."):
                try:
                    annotated, n_boxes = run_fn(image, conf)
                    st.image(annotated, caption=f"{n_boxes} detections")
                except Exception as exc:  # surface the real error in the UI for debugging
                    st.error(f"Inference failed: {exc}")
else:
    st.info("Upload an image to run detection.")
