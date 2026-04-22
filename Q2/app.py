

import os, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CityScapes Segmentation",
    page_icon="🏙️",
    layout="wide",
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 96, 128
NUM_CLASSES  = 23
DEVICE       = "cpu"

# app.py lives INSIDE Question2/ — all sibling files are right here
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "unet_cityscapes.pth")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
CURVES_PATH  = os.path.join(BASE_DIR, "training_curves.png")

PALETTE = np.array([
    [128,  64, 128],[244,  35, 232],[ 70,  70,  70],[102, 102, 156],
    [190, 153, 153],[153, 153, 153],[250, 170,  30],[220, 220,   0],
    [107, 142,  35],[152, 251, 152],[ 70, 130, 180],[220,  20,  60],
    [255,   0,   0],[  0,   0, 142],[  0,   0,  70],[  0,  60, 100],
    [  0,  80, 100],[  0,   0, 230],[119,  11,  32],[  0, 255, 255],
    [255, 165,   0],[128,   0, 128],[255, 255,   0],
], dtype=np.uint8)

CLASS_NAMES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle",
    "bicycle","water","fire hydrant","stop sign","parking meter",
]

# ─── MODEL ────────────────────────────────────────────────────────────────────
def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = double_conv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = double_conv(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = double_conv(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = double_conv(128, 64)
        self.out  = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def mask_to_rgb(mask_2d):
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask_2d == c] = PALETTE[c]
    return rgb

def preprocess_image(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor

def preprocess_mask(pil_img):
    mask = np.array(pil_img.convert("RGB"))
    mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    mask = np.max(mask, axis=-1)
    return np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)

def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
    return logits.argmax(1).squeeze(0).cpu().numpy()

# ─── SIDEBAR NAV ──────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏙️ CityScapes\n**Image Segmentation**")
page = st.sidebar.radio("Navigate", ["📊 Training Metrics", "🔍 Model Inference"])
st.sidebar.markdown("---")
st.sidebar.info("UNet trained on CityScapes | 23 classes | 15+ epochs")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Training Metrics
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Training Metrics":
    st.title("📊 Training Metrics — CityScapes UNet")

    if not os.path.exists(METRICS_PATH):
        st.error(f"metrics.json not found at `{METRICS_PATH}`. Run `train.py` first.")
        st.stop()

    with open(METRICS_PATH) as f:
        m = json.load(f)

    test_iou  = m.get("test_miou",  0)
    test_dice = m.get("test_mdice", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏁 Epochs",     len(m["loss"]))
    col2.metric("📉 Final Loss", f"{m['loss'][-1]:.4f}")
    col3.metric("📐 Test mIoU",  f"{test_iou:.4f}",  delta="✅ ≥0.48" if test_iou  >= .48 else "⚠️ <0.48")
    col4.metric("🎯 Test mDice", f"{test_dice:.4f}", delta="✅ ≥0.48" if test_dice >= .48 else "⚠️ <0.48")

    st.markdown("---")

    if os.path.exists(CURVES_PATH):
        st.subheader("Training Curves")
        st.image(CURVES_PATH, use_container_width=True)
    else:
        st.subheader("Training Curves (live)")
        epochs = list(range(1, len(m["loss"]) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#0e1117")
        [ax.set_facecolor("#1a1d27") for ax in axes]

        axes[0].plot(epochs, m["loss"],  "#e63946", linewidth=2)
        axes[0].set_title("Training Loss", color="white")
        axes[0].tick_params(colors="white"); axes[0].grid(alpha=.2)

        axes[1].plot(epochs, m["miou"],  "#2a9d8f", linewidth=2, marker="o", markersize=4)
        axes[1].axhline(test_iou, linestyle="--", color="#a8dadc", label=f"Test={test_iou:.4f}")
        axes[1].set_title("mIoU", color="white")
        axes[1].legend(facecolor="#1a1d27", labelcolor="white")
        axes[1].tick_params(colors="white"); axes[1].grid(alpha=.2)

        axes[2].plot(epochs, m["mdice"], "#457b9d", linewidth=2, marker="o", markersize=4)
        axes[2].axhline(test_dice, linestyle="--", color="#a8dadc", label=f"Test={test_dice:.4f}")
        axes[2].set_title("mDice", color="white")
        axes[2].legend(facecolor="#1a1d27", labelcolor="white")
        axes[2].tick_params(colors="white"); axes[2].grid(alpha=.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander("📋 Epoch-wise numbers"):
        import pandas as pd
        df = pd.DataFrame({
            "Epoch": list(range(1, len(m["loss"]) + 1)),
            "Loss":  [f"{v:.4f}" for v in m["loss"]],
            "mIoU":  [f"{v:.4f}" for v in m["miou"]],
            "mDice": [f"{v:.4f}" for v in m["mdice"]],
        })
        st.dataframe(df, use_container_width=True)

    st.success(f"**Test mIoU = {test_iou:.4f}   |   Test mDice = {test_dice:.4f}**")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Inference
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🔍 Segmentation Inference")
    st.markdown(
        "Upload **4 RGB images** from the test set and (optionally) their **ground-truth masks**. "
        "The model will predict the segmentation and display a side-by-side comparison."
    )

    model = load_model()

    col_img, col_mask = st.columns(2)
    with col_img:
        uploaded_imgs  = st.file_uploader("Upload 4 test **images**",
                                          type=["png","jpg","jpeg"],
                                          accept_multiple_files=True,
                                          key="imgs")
    with col_mask:
        uploaded_masks = st.file_uploader("Upload 4 corresponding **GT masks** (optional)",
                                          type=["png","jpg","jpeg"],
                                          accept_multiple_files=True,
                                          key="masks")

    if uploaded_imgs:
        n = min(len(uploaded_imgs), 4)
        st.markdown(f"### Results for {n} image(s)")

        for i in range(n):
            pil_img  = Image.open(uploaded_imgs[i])
            tensor   = preprocess_image(pil_img)
            pred     = predict(model, tensor)
            pred_rgb = mask_to_rgb(pred)

            orig_disp = np.array(pil_img.convert("RGB"))
            orig_disp = cv2.resize(orig_disp, (IMG_W * 4, IMG_H * 4), interpolation=cv2.INTER_LINEAR)
            pred_disp = cv2.resize(pred_rgb,  (IMG_W * 4, IMG_H * 4), interpolation=cv2.INTER_NEAREST)

            st.markdown(f"---\n#### Image {i+1}: `{uploaded_imgs[i].name}`")

            has_gt = i < len(uploaded_masks)
            cols   = st.columns(3 if has_gt else 2)

            cols[0].image(orig_disp,  caption="Input Image",    use_container_width=True, clamp=True)
            cols[-1].image(pred_disp, caption="Predicted Mask", use_container_width=True, clamp=True)

            if has_gt:
                pil_mask = Image.open(uploaded_masks[i])
                gt_arr   = preprocess_mask(pil_mask)
                gt_disp  = mask_to_rgb(gt_arr)
                gt_disp  = cv2.resize(gt_disp, (IMG_W * 4, IMG_H * 4), interpolation=cv2.INTER_NEAREST)
                cols[1].image(gt_disp, caption="Ground-Truth Mask", use_container_width=True, clamp=True)

        with st.expander("🎨 Class colour legend"):
            fig_leg, ax = plt.subplots(figsize=(10, 5))
            ax.axis("off")
            patches = [mpatches.Patch(color=PALETTE[c] / 255., label=CLASS_NAMES[c])
                       for c in range(NUM_CLASSES)]
            ax.legend(handles=patches, ncol=4, loc="center", fontsize=9,
                      framealpha=0, labelcolor="black")
            st.pyplot(fig_leg)
            plt.close()
    else:
        st.info("👆 Upload images above to see predictions.")

st.sidebar.markdown("---")
st.sidebar.caption("CityScapes UNet Segmentation · 23 classes")
