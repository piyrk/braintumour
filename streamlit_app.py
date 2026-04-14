from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from src.gan_brain_tumour_challenge.config import CONFIG
from src.gan_brain_tumour_challenge.models import (
    build_classifier,
    build_detection_model,
    build_discriminator,
    build_generator,
    build_unet,
)


st.set_page_config(page_title="Brain Tumour GAN Challenge", layout="wide")

st.title("Global Brain Tumour GAN Challenge")
st.caption("Working scaffold for detection, segmentation, classification, and GAN augmentation.")

artifacts_root = Path("artifacts")
gan_artifacts = artifacts_root / "gan"
detection_artifacts = artifacts_root / "detection"
segmentation_artifacts = artifacts_root / "segmentation"
classifier_artifacts = artifacts_root / "classifier"


def list_files(folder: Path, patterns: tuple[str, ...]):
    if not folder.exists():
        return []
    files = []
    for pattern in patterns:
        files.extend(folder.rglob(pattern))
    return sorted(files)


def show_artifact_group(title: str, folder: Path):
    st.markdown(f"### {title}")
    if not folder.exists():
        st.write(f"No folder found at {folder}")
        return

    model_files = list_files(folder, ("*.keras", "*.h5"))
    log_files = list_files(folder, ("*.csv", "*.json"))
    image_files = list_files(folder, ("*.png", "*.jpg", "*.jpeg"))

    st.write(f"Path: {folder}")
    st.write(f"Models: {len(model_files)} | Logs: {len(log_files)} | Images: {len(image_files)}")

    if model_files:
        st.write("Latest model:")
        st.code(str(model_files[-1]), language="text")
    if log_files:
        st.write("Latest log:")
        st.code(str(log_files[-1]), language="text")
    if image_files:
        st.write("Latest image:")
        st.image(str(image_files[-1]), caption=image_files[-1].name)

tab_overview, tab_models, tab_gallery, tab_artifacts = st.tabs(["Overview", "Models", "Gallery", "Artifacts"])

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Detection image size", f"{CONFIG.detection_image_size[0]} x {CONFIG.detection_image_size[1]}")
    col2.metric("Segmentation size", f"{CONFIG.segmentation_image_size[0]} x {CONFIG.segmentation_image_size[1]}")
    col3.metric("GAN latent dim", CONFIG.latent_dim)

    st.write("This app will later show trained results, sample outputs, and evaluation plots.")
    st.info("FS is treated as F1-score in this scaffold until the faculty defines it differently.")
    st.write("You can use the app before training to inspect architecture and later return here to view saved outputs.")

with tab_models:
    model_choice = st.selectbox(
        "Pick a model",
        ["Detection", "Segmentation", "Classifier", "GAN Generator", "GAN Discriminator"],
    )

    if model_choice == "Detection":
        model = build_detection_model((*CONFIG.detection_image_size, CONFIG.grayscale_channels))
    elif model_choice == "Segmentation":
        model = build_unet((*CONFIG.segmentation_image_size, CONFIG.grayscale_channels))
    elif model_choice == "Classifier":
        model = build_classifier(CONFIG.num_classes, (*CONFIG.detection_image_size, CONFIG.grayscale_channels))
    elif model_choice == "GAN Generator":
        model = build_generator(CONFIG.latent_dim, (*CONFIG.gan_image_size, CONFIG.grayscale_channels))
    else:
        model = build_discriminator((*CONFIG.gan_image_size, CONFIG.grayscale_channels))

    buffer = []
    model.summary(print_fn=buffer.append)
    st.code("\n".join(buffer), language="text")

with tab_gallery:
    st.write("Example synthetic samples will appear here after GAN training.")
    cols = st.columns(4)
    rng = np.random.default_rng(7)
    for index, col in enumerate(cols, start=1):
        sample = rng.normal(size=CONFIG.gan_image_size).astype("float32")
        col.image(sample, clamp=True, caption=f"Placeholder {index}")

with tab_artifacts:
    st.subheader("Saved outputs")

    history_path = gan_artifacts / "history.json"
    loss_curve_path = gan_artifacts / "loss_curve.png"
    fid_curve_path = gan_artifacts / "fid_curve.png"
    sample_dir = gan_artifacts / "samples"
    checkpoint_dir = gan_artifacts / "checkpoints"

    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as stream:
            history = json.load(stream)
        st.json(history)
    else:
        st.write("No GAN history found yet.")

    col1, col2 = st.columns(2)
    with col1:
        if loss_curve_path.exists():
            st.image(str(loss_curve_path), caption="GAN loss curve")
        else:
            st.write("Loss curve will appear after GAN training.")

    with col2:
        if fid_curve_path.exists():
            st.image(str(fid_curve_path), caption="FID vs epoch")
        else:
            st.write("FID curve will appear after GAN training.")

    if sample_dir.exists():
        sample_images = sorted(sample_dir.glob("*.png"))
        if sample_images:
            latest = sample_images[-1]
            st.image(str(latest), caption=f"Latest GAN sample grid: {latest.name}")
        else:
            st.write("No sample grids found yet.")
    else:
        st.write("Sample grids will be saved under artifacts/gan/samples.")

    st.write(f"Checkpoint folder: {checkpoint_dir}")

    st.divider()
    show_artifact_group("Detection artifacts", detection_artifacts)
    show_artifact_group("Segmentation artifacts", segmentation_artifacts)
    show_artifact_group("Classifier artifacts", classifier_artifacts)

