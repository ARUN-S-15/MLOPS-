FROM python:3.11-slim

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install \
    --default-timeout=1000 \
    --retries=20 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

RUN pip install \
    --default-timeout=1000 \
    --retries=20 \
    --no-cache-dir \
    -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "backend/app.py"]