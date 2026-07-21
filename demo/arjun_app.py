import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "Arjun_Dataset_EDA_Scratch_Baseline.ipynb"

TRAINING_HISTORY = pd.DataFrame({
    "Epoch": list(range(1, 26)),
    "Training loss": [
        1.2970, 1.0126, 0.9111, 0.8492, 0.7972,
        0.7628, 0.7291, 0.7058, 0.6885, 0.6665,
        0.6504, 0.6341, 0.6239, 0.6092, 0.5997,
        0.5904, 0.5817, 0.5757, 0.5282, 0.5080,
        0.5006, 0.4924, 0.4844, 0.4768, 0.4532,
    ],
    "Validation loss": [
        1.0692, 0.8962, 0.8553, 0.7883, 0.7552,
        0.7471, 0.7183, 0.7239, 0.7090, 0.6919,
        0.6985, 0.6933, 0.6783, 0.6810, 0.6720,
        0.6806, 0.6790, 0.6866, 0.6584, 0.6588,
        0.6575, 0.6597, 0.6636, 0.6636, 0.6509,
    ],
})

CONFIDENCE_RESULTS = pd.DataFrame({
    "Confidence": np.arange(0.20, 0.96, 0.05).round(2),
    "Precision": [
        0.2348, 0.2546, 0.2710, 0.2895, 0.3066, 0.3236,
        0.3418, 0.3618, 0.3833, 0.4063, 0.4299, 0.4538,
        0.4789, 0.5051, 0.5396, 0.5786,
    ],
    "Recall": [
        0.4914, 0.4888, 0.4859, 0.4834, 0.4818, 0.4795,
        0.4746, 0.4710, 0.4667, 0.4608, 0.4532, 0.4432,
        0.4287, 0.4090, 0.3832, 0.3234,
    ],
    "F1 score": [
        0.3178, 0.3349, 0.3479, 0.3621, 0.3747, 0.3864,
        0.3974, 0.4093, 0.4209, 0.4319, 0.4412, 0.4484,
        0.4524, 0.4520, 0.4481, 0.4149,
    ],
})

DATASET_RESULTS = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Images": [12231, 899, 846],
    "Head boxes": [54310, 4145, 3812],
    "Person boxes": [60969, 4791, 4141],
    "Empty labels": [264, 18, 20],
})

GRID_RESULTS = pd.DataFrame({
    "Grid": ["8 x 8", "10 x 10", "16 x 16"],
    "Train boxes kept": [81405, 89979, 103062],
    "Train boxes skipped": [33874, 25300, 12217],
    "Skipped percent": [29.38, 21.95, 10.60],
})


@st.cache_data
def load_notebook_image(cell_number):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for output in notebook["cells"][cell_number].get("outputs", []):
        image_data = output.get("data", {}).get("image/png")
        if image_data:
            if isinstance(image_data, list):
                image_data = "".join(image_data)
            return Image.open(BytesIO(base64.b64decode(image_data))).copy()
    return None


st.set_page_config(page_title="Arjun | Object Detection", layout="wide")
st.title("Head and Person Detection")
st.caption("Dataset analysis and scratch CNN model")
st.write("Arjun Bishnoi")

overview_tab, dataset_tab, training_tab, evaluation_tab = st.tabs([
    "Overview",
    "Dataset",
    "Training",
    "Evaluation",
])

with overview_tab:
    st.subheader("Project goal")
    st.write(
        "To detect heads and people in images using an object detector built and "
        "trained from scratch."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total images", "13,976")
    col2.metric("Detection classes", "2")
    col3.metric("Trainable parameters", "981,767")
    col4.metric("Test images", "846")

    st.subheader("Project steps")
    steps = pd.DataFrame({
        "Step": ["1", "2", "3", "4"],
        "Work completed": [
            "Checked images, labels, empty files, and invalid label lines",
            "Compared grid sizes and selected a 16 x 16 detection grid",
            "Built and trained a CNN with batch normalization and dropout",
            "Selected a confidence level and evaluated the model on test data",
        ],
    })
    st.dataframe(steps, hide_index=True, use_container_width=True)

    st.subheader("Model information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Trainable parameters", "981,767")
    col2.metric("Model size", "3.75 MB")
    col3.metric("Training time", "72.92 min")

    model_table = pd.DataFrame({
        "Setting": [
            "Image size", "Grid size", "Batch size", "Maximum epochs",
            "Learning rate", "Optimizer", "Data augmentation",
        ],
        "Value": [
            "256 x 256", "16 x 16", "32", "25", "0.001", "Adam",
            "Random horizontal flip",
        ],
    })
    st.dataframe(model_table, hide_index=True, use_container_width=True)

with dataset_tab:
    st.subheader("Dataset summary")
    st.dataframe(DATASET_RESULTS, hide_index=True, use_container_width=True)
    st.bar_chart(
        DATASET_RESULTS.set_index("Split"),
        y=["Head boxes", "Person boxes"],
    )

    if NOTEBOOK_PATH.exists():
        distribution_image = load_notebook_image(10)
        sample_image = load_notebook_image(14)
        if distribution_image is not None:
            st.subheader("Class distribution from the notebook")
            st.image(distribution_image, use_container_width=True)
        if sample_image is not None:
            st.subheader("Annotated dataset samples")
            st.image(sample_image, use_container_width=True)

    st.subheader("Grid-size comparison")
    st.write(
        "A larger grid keeps more objects when several object centres appear close "
        "together. The notebook selected the 16 x 16 grid."
    )
    st.dataframe(GRID_RESULTS, hide_index=True, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Empty label files", "302")
    col2.metric("Invalid label lines", "139")
    col3.metric("Train boxes kept", "89.40%")

with training_tab:
    st.subheader("Training summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Epochs completed", "25")
    col2.metric("Best epoch", "25")
    col3.metric("Best validation loss", "0.6509")
    col4.metric("Peak memory", "1.64 GB")

    st.line_chart(
        TRAINING_HISTORY.set_index("Epoch"),
        y=["Training loss", "Validation loss"],
    )
    st.write(
        "Training loss continued to improve. The gap between training and "
        "validation loss shows some overfitting."
    )

    if NOTEBOOK_PATH.exists():
        training_image = load_notebook_image(22)
        if training_image is not None:
            st.image(training_image, use_container_width=True)

    st.subheader("Training system")
    system_table = pd.DataFrame({
        "Item": ["Training device", "GPU memory", "System RAM", "Mixed precision"],
        "Value": ["NVIDIA RTX 4060 Laptop GPU", "8 GB", "31.42 GB", "Enabled"],
    })
    st.dataframe(system_table, hide_index=True, use_container_width=True)

with evaluation_tab:
    st.subheader("Final test results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", "0.5302")
    col2.metric("Recall", "0.4879")
    col3.metric("F1 score", "0.5082")
    col4.metric("Mean IoU", "0.6799")

    st.write("The model was tested on 846 images using a confidence of 0.80.")

    col1, col2, col3 = st.columns(3)
    col1.metric("True positives", "3,880")
    col2.metric("False positives", "3,438")
    col3.metric("False negatives", "4,073")

    st.subheader("Confidence comparison")
    st.line_chart(
        CONFIDENCE_RESULTS.set_index("Confidence"),
        y=["Precision", "Recall", "F1 score"],
    )
    st.dataframe(CONFIDENCE_RESULTS, hide_index=True, use_container_width=True)

    if NOTEBOOK_PATH.exists():
        prediction_image = load_notebook_image(28)
        if prediction_image is not None:
            st.subheader("Prediction examples from the notebook")
            st.image(prediction_image, use_container_width=True)
            st.caption("Solid boxes are labels. Red dashed boxes are model predictions.")

    st.subheader("Results")
    st.write(
        "The model is a useful scratch baseline. It detects many objects correctly, "
        "but it can miss people or add extra boxes in crowded images."
    )
