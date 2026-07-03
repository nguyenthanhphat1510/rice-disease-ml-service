---
title: Rice Disease ML Service
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# ml-service — Dự đoán bệnh lúa (FastAPI + PyTorch)

Service Python độc lập, nhận ảnh lá lúa và trả về bệnh dự đoán kèm độ tin cậy.
Backend NestJS gọi service này qua HTTP (gateway), rồi map kết quả với DB `diseases`.

```
Frontend ──ảnh──► NestJS /api/diseases/predict ──ảnh──► ml-service /predict
                       │ map slug ↔ Disease (MongoDB) ◄── { class, confidence }
                       ▼
                  trả về FE: bệnh + thuốc gợi ý
```

## 1. Cài đặt

```bash
cd ml-service
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
cp .env.example .env
```

## 2. Gắn model của bạn (YOLO classification)

Model là YOLOv8/v11-cls train bằng Ultralytics. Tên class được lưu sẵn trong file
`.pt` (`model.names`), nên **không cần khai báo lại thứ tự class**.

1. Copy file model vào `models/`, ví dụ `models/rice_disease.pt`
   (thường là `runs/classify/train/weights/best.pt` sau khi train).
2. Tạo `models/class_map.json` ánh xạ **tên class YOLO → slug** khớp `Disease.slug`
   trong MongoDB. Tên class chính là tên các thư mục ảnh lúc train. Ví dụ:
   ```json
   { "Bacterialblight": "bac-la", "Blast": "dao-on-la", "Healthy": "khoe-manh" }
   ```
   Nếu bỏ qua file này, slug = tên class đã chuẩn hóa (lowercase, `_`/space → `-`).
   Xem tên class thật bằng: `GET /classes` sau khi chạy service.
3. (Tùy chọn) Sửa `models/labels.json` map slug → tên hiển thị tiếng Việt.
4. Trong `.env`: đặt `IMG_SIZE` đúng `imgsz` lúc train (mặc định 224); `DEVICE`
   để trống là tự dò GPU.

## 3. Chạy

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger: http://localhost:8000/docs
- Health:  http://localhost:8000/health
- Classes: http://localhost:8000/classes

Test nhanh:
```bash
curl -X POST http://localhost:8000/predict -F "file=@la-lua.jpg"
```

## API

| Method | Path       | Mô tả                                    |
|--------|------------|------------------------------------------|
| GET    | /health    | Kiểm tra service + model đã load chưa     |
| GET    | /classes   | Danh sách slug class model nhận biết      |
| POST   | /predict   | multipart `file` = ảnh → top-k bệnh       |
