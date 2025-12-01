# Use CUDA 12.2 (Compatible with your RTX 3090 and Driver 570)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# FIX: Added 'ffmpeg' here
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy Requirements
COPY requirements.txt .

# 3. Install Python Packages
# Using 'pip3' alias
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 4. Copy Code & Create Dirs
COPY . .
RUN mkdir -p models temp_videos

# 5. Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]