"""
Load model YOLO classification (Ultralytics) cho service dự đoán bệnh lúa.

Model là YOLOv8/v11-cls (file .pt train bằng Ultralytics). Ultralytics tự lưu tên
class trong model (model.names), nên không cần classes.json. Tuy nhiên tên class
lúc train (vd "dao_on_la", "LeafBlast"...) thường KHÔNG khớp Disease.slug trong
MongoDB, nên ta dùng `models/class_map.json` để ánh xạ tên class -> slug.

CÁCH DÙNG:
1. Đặt file model vào `models/`, ví dụ `models/rice_disease.pt`.
   Trỏ đường dẫn qua MODEL_PATH (mặc định models/rice_disease.pt).

2. (Khuyến nghị) Tạo `models/class_map.json` ánh xạ tên class YOLO -> slug khớp
   Disease.slug trong DB. Ví dụ nếu lúc train các thư mục ảnh tên là:
   Bacterialblight / Blast / Brownspot / Healthy thì:
     {
       "Bacterialblight": "bac-la",
       "Blast": "dao-on-la",
       "Brownspot": "dom-nau",
       "Healthy": "khoe-manh"
     }
   Nếu KHÔNG có file này, slug = chính tên class (đã lowercase, thay _/space bằng -).

3. (Tùy chọn) `models/labels.json` map slug -> tên hiển thị tiếng Việt.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

from .ood import OODDetector, try_load_detector

# Thư mục chứa model + metadata, nằm cạnh package app/.
MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "rice_disease.pt"))

# Kích thước ảnh đầu vào — phải khớp imgsz lúc train YOLO-cls (mặc định 224).
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))


@dataclass
class ModelBundle:
    """Gói mọi thứ cần để infer bằng YOLO-cls."""

    model: YOLO
    # names YOLO: index -> tên class gốc (model.names).
    names: Dict[int, str]
    # tên class gốc -> slug khớp Disease.slug.
    class_to_slug: Dict[str, str]
    # slug -> nhãn hiển thị tiếng Việt.
    labels: Dict[str, str]
    img_size: int
    device: str  # giữ để báo cáo ở /health (YOLO tự chọn device khi predict)
    # tầng phát hiện ảnh lạ; None nếu thiếu ood_params.json / feature_stats.npz.
    ood: "OODDetector | None" = None

    @property
    def classes(self) -> List[str]:
        """Danh sách slug theo thứ tự index của model — cho endpoint /classes."""
        return [self.slug_for(self.names[i]) for i in sorted(self.names)]

    def slug_for(self, class_name: str) -> str:
        """Tên class gốc -> slug (qua class_map nếu có, không thì chuẩn hóa)."""
        if class_name in self.class_to_slug:
            return self.class_to_slug[class_name]
        return _slugify(class_name)


def _slugify(name: str) -> str:
    """Chuẩn hóa tên class thành slug: 'Brown Spot' -> 'brown-spot'."""
    s = name.strip().lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    return s.strip("-")


def _load_json(name: str) -> dict:
    path = MODELS_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model() -> ModelBundle:
    """Load model YOLO-cls + metadata, trả ModelBundle dùng chung cho mọi request."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {MODEL_PATH}. "
            f"Đặt file model YOLO-cls (.pt) vào và/hoặc set MODEL_PATH trong .env."
        )

    device = os.getenv("DEVICE", "cuda:0" if _has_cuda() else "cpu")
    model = YOLO(str(MODEL_PATH))

    # model.names: index -> tên class. Ép key về int cho chắc.
    names = {int(k): v for k, v in model.names.items()}

    return ModelBundle(
        model=model,
        names=names,
        class_to_slug=_load_json("class_map.json"),
        labels=_load_json("labels.json"),
        img_size=IMG_SIZE,
        device=device,
        ood=try_load_detector(model, MODELS_DIR, device),
    )


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
