# Use NVIDIA CUDA devel image (needed for vLLM CUDA kernels)
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

# Install PyTorch with CUDA 12.8 support FIRST
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install vLLM (pulls in its own dependencies)
RUN pip install --no-cache-dir vllm

# Install olmocr and remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build (avoids ~5GB download on every cold start)
RUN python3.11 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('allenai/olmOCR-2-7B-1025-FP8'); \
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"

# Copy application code and entrypoint
COPY app.py .
COPY start.sh .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Ensure torch/vLLM can find CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE 8080

# Health check (longer start period for vLLM model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the entrypoint script (starts vLLM then FastAPI)
CMD ["./start.sh"]
