# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (Python 3.11, poppler for PDF rendering, fonts)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    poppler-utils \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 support FIRST (before requirements.txt)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Copy and install remaining requirements (torch is already installed above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model and processor during build (avoids ~5GB download on every cold start)
RUN python3.11 -c "\
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; \
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct'); \
Qwen2_5_VLForConditionalGeneration.from_pretrained('allenai/olmOCR-2-7B-1025-FP8')"

# Copy application code
COPY app.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Ensure torch can find CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE 8080

# Health check (longer start period for model download on first run)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
