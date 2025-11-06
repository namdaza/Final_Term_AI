import os
import io
import json
from typing import Dict, Any, List

import torch
from torchvision import transforms, models
from PIL import Image

MODEL_DIR = "model"
METADATA_PATH = "metadata.json"
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "cake_recognizer.pth")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_meta = None
_idx_to_class = None

def _build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def _load_assets():
    global _model, _meta, _idx_to_class

    if _meta is None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)

    if _idx_to_class is None:
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            _idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

    if _model is None:
        num_classes = len(_meta)
        model = _build_model(num_classes)
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}. Please train the model first.")
        state = torch.load(WEIGHTS_PATH, map_location=_device)
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _model = model

def _transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def normalize_class_name(name: str) -> str:
    """Convert class name to standardized snake_case format."""
    return name.lower().replace(" ", "_").strip()

def predict_cake(img_path: str) -> Dict[str, Any]:
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    return predict_cake_bytes(img_bytes, source_name=img_path)

def predict_cake_bytes(img_bytes: bytes, source_name: str = "buffer") -> Dict[str, Any]:
    _load_assets()
    tfm = _transform()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = tfm(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        outputs = _model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    raw_class = _idx_to_class[int(pred_idx.item())]
    cls = normalize_class_name(raw_class)  # converts to snake_case
    meta = _meta.get(cls, {"display_name": cls, "creative_label": "", "price": 0})
    confidence = float(conf.item())

    # Top-3 (excluding top-1)
    topk = torch.topk(probs, k=min(3, probs.numel()))
    suggestions: List[Dict[str, Any]] = []
    for i in range(topk.indices.size(0)):
        idx = int(topk.indices[i].item())
        if idx == int(pred_idx.item()):
            continue
        raw_cls_i = _idx_to_class[idx]
        cls_i = normalize_class_name(raw_cls_i)
        m = _meta.get(cls_i, {"display_name": cls_i, "creative_label": "", "price": 0})
        suggestions.append({
            "class": cls_i,
            "display_name": m.get("display_name", cls_i),
            "creative_label": m.get("creative_label", ""),
            "price": m.get("price", 0),
            "confidence": float(topk.values[i].item())
        })

    # only include suggestions when confidence is low
    top_3_suggestions = suggestions[:2] if confidence < 0.7 else []

    result = {
        "predicted_class": cls,
        "display_name": meta.get("display_name", cls),
        "creative_label": meta.get("creative_label", ""),
        "price": meta.get("price", 0),
        "confidence": confidence,
        "top_3_suggestions": top_3_suggestions,
        "source": source_name
    }
    return result
