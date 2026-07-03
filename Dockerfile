FROM python:3.11-slim

WORKDIR /app

# libgl1 + libglib2.0-0: OpenCV (ultralytics kéo theo) cần, tránh lỗi "libGL.so.1 not found"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài dependencies trước (tận dụng cache Docker khi chỉ đổi code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code + model
COPY . .

# Hugging Face Spaces bắt buộc app lắng nghe cổng 7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
