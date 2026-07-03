"""
FastAPI service dự đoán bệnh lúa từ ảnh lá lúa bằng model YOLO classification.

Service này độc lập với backend NestJS. NestJS đóng vai trò gateway:
  Frontend -> NestJS /api/diseases/predict -> FastAPI /predict -> trả {class, confidence}
NestJS sẽ map class (slug) với bản ghi Disease trong MongoDB để lấy thuốc gợi ý.

Chạy:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .model import ModelBundle, load_model

app = FastAPI(
    title="Rice Disease Prediction Service",
    description="Dự đoán bệnh lúa từ ảnh lá lúa (YOLO classification)",
    version="1.0.0",
)

# Cho phép gọi từ NestJS / FE khi dev. Siết origin lại trong production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định dạng ảnh được chấp nhận và giới hạn dung lượng (5MB).
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_BYTES = 5 * 1024 * 1024

# Số lượng class trả về theo thứ tự confidence giảm dần (top-k).
TOP_K = int(os.getenv("TOP_K", "3"))

# Model được load 1 lần khi khởi động và tái dùng cho mọi request.
_bundle: Optional[ModelBundle] = None


@app.on_event("startup")
def _startup() -> None:
    global _bundle
    _bundle = load_model()


@app.get("/health")
def health() -> dict:
    """Health check cho NestJS / orchestrator kiểm tra service sống."""
    return {
        "status": "ok",
        "model_loaded": _bundle is not None,
        "device": _bundle.device if _bundle else None,
        "num_classes": len(_bundle.names) if _bundle else 0,
    }


@app.get("/classes")
def classes() -> dict:
    """Trả danh sách class model nhận biết (tên gốc + slug) — đối chiếu với DB."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")
    return {
        "classes": [
            {"name": _bundle.names[i], "slug": _bundle.slug_for(_bundle.names[i])}
            for i in sorted(_bundle.names)
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Nhận 1 ảnh lá lúa (multipart field `file`), trả top-k bệnh kèm độ tin cậy.

    Response:
      {
        "predictions": [
          { "class": "dao-on-la", "label": "Đạo ôn lá", "confidence": 0.94 },
          ...
        ],
        "top": { "class": "dao-on-la", "label": "Đạo ôn lá", "confidence": 0.94 }
      }
    """
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng ảnh không hợp lệ: {file.content_type}",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="Ảnh vượt quá 5MB")
    if not raw:
        raise HTTPException(status_code=400, detail="File ảnh rỗng")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh")

    predictions = _infer(image, _bundle)
    return {"predictions": predictions, "top": predictions[0]}


def _infer(image: Image.Image, bundle: ModelBundle) -> List[dict]:
    """Chạy YOLO-cls trên ảnh -> lấy top-k class từ results.probs."""
    results = bundle.model.predict(
        source=image,
        imgsz=bundle.img_size,
        device=bundle.device,
        verbose=False,
    )
    probs = results[0].probs  # đối tượng Probs của Ultralytics

    k = min(TOP_K, len(bundle.names))
    top_idx = probs.top5[:k]  # list index theo confidence giảm dần
    conf = probs.data.tolist()  # vector xác suất theo index class

    out: List[dict] = []
    for idx in top_idx:
        name = bundle.names[int(idx)]
        slug = bundle.slug_for(name)
        out.append(
            {
                "class": slug,
                "label": bundle.labels.get(slug, name),
                "confidence": round(float(conf[int(idx)]), 4),
            }
        )
    return out
