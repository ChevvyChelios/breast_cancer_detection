# app_nomask.py
import io
import os
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, models

import streamlit as st
import pandas as pd


# ----------------------
# PATHS
# ----------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_PATH = PROJECT_ROOT / "models" / "model1.pt"
LABEL_MAP_PATH = PROJECT_ROOT / "metadata" / "label_mapping.json"
IMG_SIZE = (224, 224)


# ----------------------
# Load label mapping
# ----------------------
def load_label_map(path):
    if not path.exists():
        st.error(f"Label mapping not found at {path}")
        return None
    with open(path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    return idx_to_label


# ----------------------
# Load the model
# ----------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect model architecture (resnet18 vs resnet50)
    try:
        size_bytes = model_path.stat().st_size
    except:
        size_bytes = 0

    if size_bytes > 50 * 1024 * 1024:
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# ----------------------
# Preprocessing (must match training)
# ----------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])


# ----------------------
# Prediction
# ----------------------
def predict_image(model, device, pil_img):
    transform = get_transform()
    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    return pred_idx, conf, probs


# ----------------------
# STREAMLIT UI
# ----------------------
st.set_page_config(page_title="Breast Ultrasound Classifier (No Mask)", layout="centered")
st.title("Breast Ultrasound Classification (Main-Image Model)")

st.write("Upload a **main ultrasound image** (no mask required).")


# Load label map
idx_to_label = load_label_map(LABEL_MAP_PATH)
if idx_to_label is None:
    st.stop()

NUM_CLASSES = len(idx_to_label)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
model_file = st.file_uploader("Upload custom model (optional)", type=["pt", "pth"])

if st.button("Predict"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
        st.stop()

    # pick which model file to load
    if model_file is not None:
        temp_model_path = PROJECT_ROOT / "models" / "uploaded_nomask_model.pt"
        with open(temp_model_path, "wb") as f:
            f.write(model_file.read())
        model_path_to_load = temp_model_path
    else:
        model_path_to_load = MODEL_PATH

    if not model_path_to_load.exists():
        st.error(f"Model not found at: {model_path_to_load}")
        st.stop()

    model, device = load_model(model_path_to_load, NUM_CLASSES)

    # Read image
    try:
        img_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Error reading image: {e}")
        st.stop()

    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # Predict
    pred_idx, conf, probs = predict_image(model, device, pil_img)
    pred_label = idx_to_label.get(pred_idx, str(pred_idx))

    st.subheader("Prediction")
    st.write(f"**Predicted Class:** `{pred_label}`")
    st.write(f"**Confidence:** {conf:.3f}")

    # Show class probabilities
    rows = []
    for i in range(NUM_CLASSES):
        label = idx_to_label.get(i, str(i))
        rows.append({"Class": label, "Probability": float(probs[i])})

    df_probs = pd.DataFrame(rows).sort_values("Probability", ascending=False)
    st.table(df_probs)
