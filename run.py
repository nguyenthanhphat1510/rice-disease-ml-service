"""
Chạy service dự đoán bệnh lúa cho tiện: `python run.py`.

Tương đương với:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Đổi host/port qua biến môi trường (hoặc .env của bạn):
    HOST=0.0.0.0 PORT=8080 python run.py

Đặt RELOAD=1 để uvicorn tự khởi động lại khi sửa code (chậm hơn vì phải nạp lại
model mỗi lần lưu file) — thường chỉ cần khi đang sửa app/.
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD") == "1",
    )
