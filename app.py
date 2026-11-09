"""Streamlit interface for the trained groundnut leaf classifier."""
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.model_utils import prepare_image, load_class_names

MODEL_PATH = Path("models/groundnut_efficientnetv2b0.keras")
CLASSES_PATH = Path("models/class_names.json")

st.set_page_config(page_title="Groundnut Leaf Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Groundnut Leaf Disease Detection")
st.write(
    "Upload a groundnut leaf image to classify it as healthy or one of four disease categories. "
    "This is a research demonstration, not a substitute for an agricultural expert."
)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.warning("A trained model was not found. Add the dataset and run `python train.py` first.")
    st.stop()

@st.cache_resource
def load_artifacts():
    return tf.keras.models.load_model(MODEL_PATH), load_class_names(CLASSES_PATH)

model, class_names = load_artifacts()
uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf image", use_container_width=True)
    if st.button("Predict disease", type="primary"):
        with st.spinner("Analyzing image..."):
            scores = model.predict(prepare_image(image), verbose=0)[0]
        best = int(np.argmax(scores))
        st.subheader(f"Prediction: {class_names[best]}")
        st.metric("Model softmax score", f"{scores[best] * 100:.2f}%")
        st.caption("Softmax scores are model outputs and are not calibrated diagnostic probabilities.")
        st.write("Class scores")
        score_rows = sorted(zip(class_names, scores), key=lambda x: x[1], reverse=True)
        for label, score in score_rows:
            st.write(f"**{label}** — {score * 100:.2f}%")
            st.progress(float(score))

st.divider()
st.caption("Use predictions as an initial screening aid only. Confirm disease identification with a qualified agricultural specialist.")
