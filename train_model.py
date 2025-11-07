import os
import cv2
import json
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets


# ======================
# CONFIG
# ======================
DATASET_DIR = "dataset"              # train / valid / test nằm ở đây
DATASET_SEG = "dataset_segmented"    # output segmentation

MODEL_DIR = "model"
MODEL_PTH = os.path.join(MODEL_DIR, "model.pth")
MODEL_H5 = os.path.join(MODEL_DIR, "model.h5")
CLASS_MAP_PATH = os.path.join(MODEL_DIR, "class_map.json")

IMG_SIZE = 128
EPOCHS = 15
BATCH_SIZE = 16
LR = 0.0005


# ======================
# UTILS
# ======================
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATASET_SEG, exist_ok=True)


# ======================
# SEGMENTATION
# ======================

def segment_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    crop = img[y:y+h, x:x+w]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


def build_segmented_dataset():
    ensure_dirs()

    print("🔧 Running segmentation on dataset...")

    for split in ["train", "valid", "test"]:
        raw_split_dir = os.path.join(DATASET_DIR, split)
        seg_split_dir = os.path.join(DATASET_SEG, split)

        if not os.path.exists(raw_split_dir):
            continue

        os.makedirs(seg_split_dir, exist_ok=True)

        for cls in os.listdir(raw_split_dir):
            raw_cls_dir = os.path.join(raw_split_dir, cls)
            seg_cls_dir = os.path.join(seg_split_dir, cls)

            if not os.path.isdir(raw_cls_dir):
                continue

            os.makedirs(seg_cls_dir, exist_ok=True)

            for file in os.listdir(raw_cls_dir):
                img_path = os.path.join(raw_cls_dir, file)

                try:
                    crop = segment_image(img_path)
                    if crop is not None:
                        cv2.imwrite(os.path.join(seg_cls_dir, file), crop)
                except:
                    continue

    print("✅ Segmentation done!")


# ======================
# CNN MODEL (PyTorch)
# ======================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ======================
# TRAINING
# ======================

def train_model():
    build_segmented_dataset()

    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # Load train/valid
    train_ds = datasets.ImageFolder(os.path.join(DATASET_SEG, "train"), transform=tfms)
    valid_ds = datasets.ImageFolder(os.path.join(DATASET_SEG, "valid"), transform=tfms)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_ds.class_to_idx)

    # Save class map
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(train_ds.class_to_idx, f, ensure_ascii=False, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 Training CNN from scratch...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss/len(train_loader):.4f}")

    print("✅ Training complete!")

    torch.save(model.state_dict(), MODEL_PTH)
    print("✅ Saved PyTorch model:", MODEL_PTH)

    save_h5(model)
    print("✅ Saved H5 model:", MODEL_H5)


# ======================
# EXPORT H5 (Không cần TensorFlow)
# ======================

def save_h5(model):
    """Lưu trọng số PyTorch vào file HDF5 (.h5)."""
    with h5py.File(MODEL_H5, "w") as f:
        for name, param in model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())


if __name__ == "__main__":
    train_model()
