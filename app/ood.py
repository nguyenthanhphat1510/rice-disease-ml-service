"""
Tầng phát hiện OOD (ảnh lạ / bệnh ngoài danh mục) cho service dự đoán bệnh lúa.

Bản port từ `phan_loai_datn/ood/ood_detector.py` sang service: bỏ đường dẫn
tuyệt đối, tái dùng model đã load trong ModelBundle, và trả slug khớp
Disease.slug trong MongoDB thay vì tên class gốc.

Ba tín hiệu chạy trên cùng 1 lần forward:
  1. Temperature scaling  -> xác suất đã hiệu chỉnh (p_max, margin).
  2. Energy score         -> ảnh càng lạ, energy càng cao.
  3. Feature distance     -> cosine tới tâm class gần nhất trong không gian 1280-D.

Luật quyết định (theo tài liệu, mục 10-11):
  UNKNOWN_DISEASE       : energy VÀ distance đều vượt ngưỡng -> chắc chắn OOD.
  NEED_MORE_INFORMATION : 1 tín hiệu vượt, hoặc model phân vân giữa 2 bệnh.
  KNOWN_DISEASE         : cả 3 tín hiệu đồng ý -> tin cậy cao.

Ngưỡng nằm trong `models/ood_params.json`, tâm feature trong
`models/feature_stats.npz` — cả hai phải sinh ra từ CHÍNH file weights đang
dùng (models/rice_disease.pt), nếu train lại model thì phải tạo lại cả hai.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ngưỡng phụ cho vùng xám (chọn thực dụng, không phải từ tối ưu AUROC):
PROB_MARGIN = 0.15  # top1 - top2 < 15% -> 2 bệnh sát nhau, hỏi thêm
PROB_LOW = 0.55     # p_max sau hiệu chỉnh quá thấp -> hỏi thêm

IMG_SIZE = 224


def _preprocess(img: Image.Image) -> torch.Tensor:
    """Resize giữ tỉ lệ theo cạnh ngắn rồi center-crop — khớp pipeline lúc đo ngưỡng."""
    img = img.convert("RGB")
    w, h = img.size
    s = IMG_SIZE / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    l, t = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    img = img.crop((l, t, l + IMG_SIZE, t + IMG_SIZE))
    return torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)


@dataclass
class OODResult:
    status: str
    energy: float
    distance: float
    energy_is_ood: bool
    distance_is_ood: bool
    top_prob_calibrated: float
    margin: float
    # slug -> xác suất đã hiệu chỉnh bằng temperature
    probs_by_class: Dict[str, float]


class OODDetector:
    """Bọc model đã load để tính 3 tín hiệu OOD trong 1 lần forward."""

    def __init__(self, yolo_model, models_dir: Path, device: str):
        params_path = models_dir / "ood_params.json"
        stats_path = models_dir / "feature_stats.npz"

        p = json.loads(params_path.read_text(encoding="utf-8"))
        self.T: float = p["temperature"]
        self.e_thr: float = p["energy_threshold"]
        self.d_thr: float = p["distance_threshold"]
        # thứ tự class trong params phải khớp index của model
        self.names: List[str] = p["class_names"]

        mus = np.load(stats_path)["mus"]
        # chuẩn hóa sẵn tâm để tính cosine nhanh
        self.mus_n = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-8)

        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.net = yolo_model.model.to(self.device).eval()

        # hook lấy feature ngay trước lớp linear cuối (đầu vào classifier).
        self._feat: Dict[str, torch.Tensor] = {}
        self.net.model[10].linear.register_forward_pre_hook(self._hook)

    def _hook(self, module, inp):
        self._feat["f"] = inp[0].detach()

    @torch.no_grad()
    def analyze(self, img: Image.Image) -> OODResult:
        x = _preprocess(img).unsqueeze(0).to(self.device)
        out = self.net(x)
        logits = (out[1] if isinstance(out, (tuple, list)) else out)[0].float().cpu()
        feat = self._feat["f"][0].float().cpu().numpy()

        # 1. xác suất đã hiệu chỉnh
        probs = F.softmax(logits / self.T, 0).numpy()
        order = np.argsort(-probs)
        p_max = float(probs[order[0]])
        margin = float(probs[order[0]] - probs[order[1]])

        # 2. energy
        energy = float(-self.T * torch.logsumexp(logits / self.T, 0))

        # 3. cosine distance tới tâm gần nhất
        fn = feat / (np.linalg.norm(feat) + 1e-8)
        d_min = float(1.0 - (self.mus_n @ fn).max())

        energy_ood = energy > self.e_thr
        distance_ood = d_min > self.d_thr

        if energy_ood and distance_ood:
            status = "UNKNOWN_DISEASE"
        elif energy_ood or distance_ood:
            status = "NEED_MORE_INFORMATION"
        elif p_max < PROB_LOW or margin < PROB_MARGIN:
            status = "NEED_MORE_INFORMATION"
        else:
            status = "KNOWN_DISEASE"

        return OODResult(
            status=status,
            energy=round(energy, 3),
            distance=round(d_min, 3),
            energy_is_ood=bool(energy_ood),
            distance_is_ood=bool(distance_ood),
            top_prob_calibrated=round(p_max, 4),
            margin=round(margin, 4),
            probs_by_class={self.names[i]: float(probs[i]) for i in range(len(self.names))},
        )


def try_load_detector(yolo_model, models_dir: Path, device: str) -> Optional[OODDetector]:
    """Load detector nếu đủ file; thiếu file thì trả None để service vẫn chạy được."""
    if not (models_dir / "ood_params.json").exists():
        return None
    if not (models_dir / "feature_stats.npz").exists():
        return None
    return OODDetector(yolo_model, models_dir, device)
