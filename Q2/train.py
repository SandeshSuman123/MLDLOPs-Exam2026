import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# CONFIG 
SEED         = 42
IMG_H, IMG_W = 96, 128
NUM_CLASSES  = 23
EPOCHS       = 15
BATCH_SIZE   = 8
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── DATASET ──────────────────────────────────────────────────────────────────
class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)                     # collapse RGB → single channel
        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)

        img  = torch.from_numpy(img).permute(2, 0, 1)   # C,H,W
        mask = torch.from_numpy(mask).long()
        return img, mask


# ─── U-NET ────────────────────────────────────────────────────────────────────
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

        self.up4   = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4  = double_conv(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3  = double_conv(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2  = double_conv(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1  = double_conv(128, 64)

        self.out   = nn.Conv2d(64, num_classes, 1)

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


# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_miou(preds, masks, num_classes=NUM_CLASSES):
    ious = []
    preds = preds.view(-1); masks = masks.view(-1)
    for c in range(num_classes):
        inter = ((preds == c) & (masks == c)).sum().item()
        union = ((preds == c) | (masks == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

def compute_mdice(preds, masks, num_classes=NUM_CLASSES):
    dices = []
    preds = preds.view(-1); masks = masks.view(-1)
    for c in range(num_classes):
        tp = ((preds == c) & (masks == c)).sum().item()
        fp = ((preds == c) & (masks != c)).sum().item()
        fn = ((preds != c) & (masks == c)).sum().item()
        denom = 2 * tp + fp + fn
        if denom > 0:
            dices.append(2 * tp / denom)
    return np.mean(dices) if dices else 0.0


# ─── TRAIN / EVAL ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, total_iou, total_dice, n = 0, 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, masks)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(1)
            total_loss += loss.item()
            total_iou  += compute_miou(preds, masks)
            total_dice += compute_mdice(preds, masks)
            n += 1
    return total_loss / n, total_iou / n, total_dice / n


def main():
    # Paths
    img_dir = "../data/extracted/CameraRGB"
    mask_dir = "../data/extracted/CameraMask"
    imgs  = sorted([os.path.join(img_dir,  f) for f in os.listdir(img_dir)  if f.endswith(('.png','.jpg'))])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png','.jpg'))])
    assert len(imgs) == len(masks), f"Mismatch: {len(imgs)} imgs vs {len(masks)} masks"

    tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(imgs, masks, test_size=0.2, random_state=SEED)

    tr_loader = DataLoader(CityscapesDataset(tr_imgs, tr_masks), BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    te_loader = DataLoader(CityscapesDataset(te_imgs, te_masks), BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"loss": [], "miou": [], "mdice": []}

    print(f"Training on {DEVICE} | {len(tr_imgs)} train / {len(te_imgs)} test samples")
    for epoch in range(1, EPOCHS + 1):
        loss, iou, dice = run_epoch(model, tr_loader, criterion, optimizer)
        scheduler.step()
        history["loss"].append(loss)
        history["miou"].append(iou)
        history["mdice"].append(dice)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  mIoU={iou:.4f}  mDice={dice:.4f}")

    # ── Test set metrics ──────────────────────────────────────────────────────
    _, test_iou, test_dice = run_epoch(model, te_loader, criterion)
    print(f"\nTest  mIoU={test_iou:.4f}  mDice={test_dice:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("Question2", exist_ok=True)
    torch.save(model.state_dict(), "Question2/unet_cityscapes.pth")

    metrics = {**history, "test_miou": test_iou, "test_mdice": test_dice,
               "test_image_paths": te_imgs, "test_mask_paths": te_masks}
    with open("Question2/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    epochs = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("UNet Training — Cityscapes", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["loss"],  color="#e63946", linewidth=2)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=.3)

    axes[1].plot(epochs, history["miou"],  color="#2a9d8f", linewidth=2, marker="o", markersize=4)
    axes[1].axhline(test_iou, linestyle="--", color="#264653", label=f"Test={test_iou:.4f}")
    axes[1].set_title("mIoU"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=.3)

    axes[2].plot(epochs, history["mdice"], color="#457b9d", linewidth=2, marker="o", markersize=4)
    axes[2].axhline(test_dice, linestyle="--", color="#1d3557", label=f"Test={test_dice:.4f}")
    axes[2].set_title("mDice"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("Question2/training_curves.png", dpi=150)
    plt.close()
    print("Saved Question2/training_curves.png")
    print("Saved Question2/unet_cityscapes.pth")
    print("Saved Question2/metrics.json")


if __name__ == "__main__":
    main()
