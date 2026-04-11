# =============================================================================
# ACE-Step 1.5 — NVIDIA DGX Spark (aarch64 / CUDA 13)
# =============================================================================
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

ARG ACESTEP_REF=main

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1

# ==================== System packages ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget \
        ffmpeg libsndfile1 libsndfile1-dev \
        libgl1 libglib2.0-0 \
        build-essential cmake \
        libopenblas-dev liblapack-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

# ==================== Conda + Python 3.12 ====================
RUN curl -fsSL -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh

ENV PATH="/opt/conda/envs/acestep/bin:/opt/conda/bin:${PATH}"

RUN conda create -y -n acestep python=3.12 pip && \
    conda clean -afy

# ==================== PyTorch (cu130, aarch64) ====================
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    pip install --no-cache-dir \
      torch==2.10.0 \
      torchvision==0.25.0 \
      torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cu130

# The base image (cuda:13.0.1) ships CUBLAS 13.0.2 which crashes on GB10
# (compute 12.1). Install newer CUBLAS from NVIDIA's pip index and ensure
# it loads before the broken system lib.
RUN pip install --no-cache-dir --index-url https://pypi.nvidia.com "nvidia-cublas>=13.1"
ENV LD_LIBRARY_PATH="/opt/conda/envs/acestep/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"

# ==================== Clone ACE-Step ====================
RUN git clone https://github.com/ACE-Step/ACE-Step-1.5.git /app && \
    cd /app && git checkout "${ACESTEP_REF}"

WORKDIR /app

# ==================== Python dependencies ====================
# Install from pyproject.toml but skip torch (already installed) and mlx (Apple only)
RUN pip install --no-cache-dir \
        "transformers>=4.51.0,<4.58.0" \
        "diffusers" \
        "gradio==6.2.0" \
        "matplotlib>=3.7.5" \
        "scipy>=1.10.1" \
        "soundfile>=0.13.1" \
        "loguru>=0.7.3" \
        "einops>=0.8.1" \
        "accelerate>=1.12.0" \
        "fastapi>=0.110.0" \
        "diskcache" \
        "uvicorn[standard]>=0.27.0" \
        "numba>=0.63.1" \
        "vector-quantize-pytorch>=1.27.15" \
        "toml" \
        "safetensors" \
        "modelscope" \
        "peft>=0.18.0" \
        "lycoris-lora" \
        "lightning>=2.0.0" \
        "tensorboard>=2.20.0" \
        "typer-slim>=0.21.1" \
        "xxhash" \
        "pyyaml" \
        "bitsandbytes>=0.49.0" \
        "triton>=3.4.0"

# torchcodec is NOT installed — pip wheel has ABI mismatch with torch 2.10.
# launch.py patches torchaudio.save to use soundfile backend directly.
# ACE-Step shells out to ffmpeg CLI for MP3 conversion.
RUN pip uninstall -y torchcodec 2>/dev/null; true

# Install nano-vllm from bundled source (no-deps to avoid pulling x86 flash-attn)
RUN pip install --no-cache-dir --no-deps /app/acestep/third_parts/nano-vllm

# Install the project itself (no-deps, we installed everything above)
RUN pip install --no-cache-dir --no-deps -e .

# ==================== Runtime directories ====================
RUN mkdir -p /app/checkpoints /app/gradio_outputs /app/acestep_output /app/lora_output

# ==================== Launch script ====================
COPY launch.py /app/launch.py

# ==================== Environment ====================
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ACESTEP_MODE=gradio
ENV ACESTEP_LLM_BACKEND=pt
ENV ACESTEP_INIT_SERVICE=true
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0

EXPOSE 7860

CMD ["python", "/app/launch.py"]
