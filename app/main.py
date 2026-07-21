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
        "ood_enabled": bool(_bundle and _bundle.ood),
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
    Nhận 1 ảnh lá lúa (multipart field `file`), trả top-k bệnh kèm độ tin cậy
    và trạng thái từ tầng phát hiện ảnh lạ (OOD).

    Response:
      {
        "status": "KNOWN_DISEASE" | "NEED_MORE_INFORMATION" | "UNKNOWN_DISEASE",
        "message": "...",                       // câu trả lời gợi ý cho người dùng
        "predictions": [
          { "class": "dao-on-la", "label": "Đạo ôn lá", "confidence": 0.94 },
          ...
        ],
        "top": { ... },                          // null khi status = UNKNOWN_DISEASE
        "ood": { "energy_score": ..., "feature_distance": ..., ... }
      }

    NestJS chỉ nên tra thuốc trong DB khi status = KNOWN_DISEASE; hai trạng thái
    còn lại nên hiển thị `message` và xin thêm ảnh/mô tả.
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

    if _bundle.ood is not None:
        return _predict_with_ood(image, _bundle)

    # Không có file OOD -> chạy chế độ cũ, luôn coi là biết bệnh.
    predictions = _infer(image, _bundle)
    return {
        "status": "KNOWN_DISEASE",
        "message": _message("KNOWN_DISEASE", predictions),
        "predictions": predictions,
        "top": predictions[0],
        "ood": None,
    }


def _predict_with_ood(image: Image.Image, bundle: ModelBundle) -> dict:
    """Chạy 3 tín hiệu OOD, xếp hạng class theo xác suất đã hiệu chỉnh."""
    res = bundle.ood.analyze(image)

    ranked = sorted(res.probs_by_class.items(), key=lambda kv: -kv[1])
    k = min(TOP_K, len(ranked))
    predictions = []
    for name, prob in ranked[:k]:
        slug = bundle.slug_for(name)
        predictions.append(
            {
                "class": slug,
                "label": bundle.labels.get(slug, name),
                "confidence": round(float(prob), 4),
            }
        )

    return {
        "status": res.status,
        "message": _message(res.status, predictions),
        "predictions": predictions,
        # ảnh lạ thì không có bệnh nào đáng tra thuốc
        "top": predictions[0] if res.status != "UNKNOWN_DISEASE" else None,
        "ood": {
            "energy_score": res.energy,
            "energy_threshold": bundle.ood.e_thr,
            "energy_is_ood": res.energy_is_ood,
            "feature_distance": res.distance,
            "distance_threshold": bundle.ood.d_thr,
            "distance_is_ood": res.distance_is_ood,
            "top_prob_calibrated": res.top_prob_calibrated,
            "margin": res.margin,
        },
    }


def _message(status: str, preds: List[dict]) -> str:
    """Câu trả lời tiếng Việt cho từng trạng thái, để FE/chatbot hiển thị thẳng."""
    if status == "KNOWN_DISEASE":
        return f"Kết quả: {preds[0]['label']} ({preds[0]['confidence'] * 100:.1f}%)."
    if status == "NEED_MORE_INFORMATION":
        second = preds[1]["label"] if len(preds) > 1 else preds[0]["label"]
        return (
            "Chưa đủ tin cậy để kết luận. Một số đặc điểm gần với "
            f"{preds[0]['label']} hoặc {second}, nhưng cần thêm hình ảnh cận cảnh "
            "vết bệnh và mô tả triệu chứng."
        )
    return (
        "Ảnh này không giống các bệnh trong hệ thống. Có thể là bệnh khác, ảnh "
        "không phải lá lúa, hoặc điều kiện chụp chưa phù hợp. Vui lòng cung cấp "
        "thêm ảnh và mô tả, hoặc hỏi ý kiến chuyên gia."
    )


def _infer(image: Image.Image, bundle: ModelBundle) -> List[dict]:
    """Chạy YOLO-cls trên ảnh -> lấy top-k class từ results.probs (không có OOD)."""
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
