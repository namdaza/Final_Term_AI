import os
import json
import yaml
import time
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from tqdm import tqdm

CONFIG_PATH = "config.yaml"
MODEL_DIR = "model"
LOGS_DIR = "logs"
METADATA_PATH = "metadata.json"

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_transforms(cfg: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    size = cfg["dataset"]["input_size"]
    aug = cfg.get("augmentations", {})
    train_tfms = [transforms.Resize((size, size))]
    if aug.get("rotation", 0):
        train_tfms.append(transforms.RandomRotation(aug["rotation"]))
    if aug.get("flip", True):
        train_tfms.append(transforms.RandomHorizontalFlip())
    if aug.get("brightness", True):
        train_tfms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    train_tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    val_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(train_tfms), val_tfms

def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_dir = cfg["dataset"]["train_dir"]
    val_dir = cfg["dataset"]["val_dir"]
    batch_size = cfg["dataset"]["batch_size"]
    num_workers = cfg["dataset"]["num_workers"]

    train_tfms, val_tfms = build_transforms(cfg)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.class_to_idx

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, metric: float):
        if self.best is None:
            self.best = metric
            return
        improve = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improve:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True

def plot_curves(train_losses: List[float], val_losses: List[float], val_accs: List[float]):
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "training_loss.png"))
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "training_accuracy.png"))
    plt.close()

    # For compatibility with the original requirement name
    # we also save a merged placeholder
    plt.figure(figsize=(8, 5))
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Training Overview (Val Acc)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.close()

def main():
    print("🔧 Loading config...")
    cfg = load_config(CONFIG_PATH)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    print("📦 Creating dataloaders...")
    train_loader, val_loader, class_to_idx = create_dataloaders(cfg)

    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w", encoding="utf-8") as f:
        # Save both directions
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()}
        }, f, ensure_ascii=False, indent=2)

    print("🧠 Building model...")
    num_classes = cfg["dataset"]["num_classes"]
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    best_acc = 0.0
    early = EarlyStopping(patience=cfg["train"]["early_stopping_patience"], mode="max")

    train_losses, val_losses, val_accs = [], [], []

    print("🏁 Training start...")
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_running_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"📊 Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f} | Val Acc {val_acc:.2f}%")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "cake_recognizer.pth"))
            print("💾 Saving best model...")

        # Early stopping
        early.step(val_acc)
        if early.should_stop:
            print("⏹ Early stopping triggered.")
            break

    print("🖼 Plotting curves...")
    plot_curves(train_losses, val_losses, val_accs)

    print("✅ Training done. Best Val Acc: {:.2f}%".format(best_acc))

if __name__ == "__main__":
    main()
