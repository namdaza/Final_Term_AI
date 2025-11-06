import os
import json
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from infer import predict_cake_bytes

META_PATH = "metadata.json"

app = FastAPI(title="Smart Bakery AI Backend", version="1.0.0")

# Allow all origins for quick prototyping (tune for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_metadata() -> Dict[str, Any]:
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("metadata.json not found.")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/cakes")
def get_cakes():
    """Return all cakes metadata (for UI lists)."""
    meta = load_metadata()
    items = []
    for k, v in meta.items():
        items.append({
            "class": k,
            "display_name": v.get("display_name", k),
            "creative_label": v.get("creative_label", ""),
            "price": v.get("price", 0)
        })
    return {"cakes": items, "count": len(items)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image and return prediction."""
    try:
        img_bytes = await file.read()
        result = predict_cake_bytes(img_bytes, source_name=file.filename)
        # Wrap to match required output schema (array + total)
        predicted_items = [{
            "class": result["predicted_class"],  # normalized class name
            "display_name": result["display_name"],
            "creative_label": result["creative_label"],
            "confidence": result["confidence"],
            "price": result["price"]  # price from normalized metadata lookup
        }]
        return {
            "predicted_items": predicted_items,
            "top_3_suggestions": result.get("top_3_suggestions", []),
            "total": sum(item["price"] for item in predicted_items)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/confirm_order")
async def confirm_order(payload: Dict[str, Any]):
    """
    Confirm a list of items and compute invoice.
    Expected payload:
    {
      "items": [
        {"class": "croissant", "qty": 2},
        {"class": "egg_tart", "qty": 1}
      ]
    }
    """
    meta = load_metadata()
    items = payload.get("items", [])
    if not isinstance(items, list) or len(items) == 0:
        raise HTTPException(status_code=400, detail="No items provided.")

    invoice_lines = []
    total = 0
    for it in items:
        cls = it.get("class")
        qty = int(it.get("qty", 1))
        if cls not in meta:
            raise HTTPException(status_code=400, detail=f"Unknown class: {cls}")
        price = int(meta[cls].get("price", 0))
        line_total = price * qty
        total += line_total
        invoice_lines.append({
            "class": cls,
            "display_name": meta[cls].get("display_name", cls),
            "qty": qty,
            "unit_price": price,
            "line_total": line_total
        })

    return {
        "invoice": invoice_lines,
        "total": total,
        "currency": "VND"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # import tại chỗ để tránh lỗi khi module không có sẵn lúc import top-level
    try:
        from detect import detect_bakery
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_bakery not available: {e}")

    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    result = detect_bakery(temp_path)

    items = []
    for box in result.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]

        items.append({
            "class": result.names[cls],
            "confidence": conf,
            "bbox": xyxy
        })

    return {"detections": items}

from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Tính đường dẫn tới thư mục dist của frontend
BACKEND_DIR = Path(__file__).resolve().parent
FRONT_DIST = BACKEND_DIR.parent / "smart_bakery_frontend" / "dist"

# Serve frontend build ở root (API đã được khai báo trước nên vẫn hoạt động)
app.mount("/", StaticFiles(directory=str(FRONT_DIST), html=True), name="frontend")

if __name__ == "__main__":
    # Run with: python app.py  (or: uvicorn app:app --reload)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
