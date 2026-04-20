FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/root/.insightface

# System deps:
#  - libgl1, libglib2.0-0: OpenCV runtime
#  - poppler-utils: pdf2image (PDF rendering)
#  - build-essential, cmake, libjpeg-dev: needed by face_recognition/dlib fallback
#  - libsm6, libxext6, libxrender1: extra OpenCV deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        poppler-utils \
        build-essential cmake \
        libjpeg-dev libpng-dev \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY README.md ./README.md

RUN mkdir -p /app/data/photos /app/data/thumbs /app/data/faces

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
