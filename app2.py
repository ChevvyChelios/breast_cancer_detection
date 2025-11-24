import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import os

from src.model import create_model
from src.utils import get_mask_bbox


# ------------ SETTINGS ------------
MODEL_PATH = "models/model2.pt"
LABELS = ["normal", "benign", "malignant"]

transform_roi = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------ LOAD MODEL ------------
@st.cache_resource
def load_model():
    model = create_model(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


# ------------ ROI FUNCTION ------------
def crop_roi(main_img, mask_img):
    if mask_img is None:
        return main_img

    bbox = get_mask_bbox(mask_img)
    if bbox is None:
        return main_img

    return main_img.crop(bbox)


# ------------ PREDICTION ------------
def predict(main_img, mask_img):
    roi = crop_roi(main_img, mask_img)
    img_tensor = transform_roi(roi).unsqueeze(0).to(device)

    model = load_model()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    pred_conf = float(probs[pred_idx])

    return pred_label, pred_conf, roi


# ------------ STREAMLIT UI ------------
st.set_page_config(page_title="Breast Ultrasound Classifier",
                   page_icon="🩺",
                   layout="wide")

st.title("🩺 Breast Ultrasound Classification using ROI Mask Model")
st.write("Upload the **main ultrasound image** and optionally upload the **tumor mask** to detect ROI.")


# ------------ FILE UPLOADS ------------
uploaded_main = st.file_uploader("Upload Main Ultrasound Image", type=["png", "jpg", "jpeg"])
uploaded_mask = st.file_uploader("Upload Tumor Mask Image (Optional)", type=["png"])

col1, col2 = st.columns(2)

if uploaded_main:
    main_img = Image.open(uploaded_main).convert("RGB")

    with col1:
        st.subheader("Main Image")
        st.image(main_img, use_column_width=True)

    mask_img = None
    if uploaded_mask:
        mask_img = Image.open(uploaded_mask).convert("L")
        with col2:
            st.subheader("Mask Image")
            st.image(mask_img, use_column_width=True)

    # Predict button
    if st.button("Run Prediction"):
        pred_label, pred_conf, roi = predict(main_img, mask_img)

        st.success(f"**Prediction: {pred_label.upper()}** (confidence: {pred_conf:.4f})")

        st.subheader("ROI Used for Prediction")
        st.image(roi, use_column_width=False, width=300)
