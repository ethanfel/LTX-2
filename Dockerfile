###############################################################################
# LTX-2 Trainer image
#
# Multi-stage build:
#   * builder  – resolves & installs the uv workspace into /app/.venv
#   * runtime  – CUDA runtime + ffmpeg/opencv/soundfile system libs + the venv
#
# torch's PyPI wheels bundle their own CUDA libraries (torch 2.11 ships the
# cu13 / CUDA 13 build), so the base image only needs a matching CUDA runtime
# for driver compatibility. CUDA 13 + these wheels support Blackwell
# (sm_120 / RTX 50-series).
#
# The model checkpoint (.safetensors) and the Gemma text-encoder directory are
# NOT baked into the image — mount them at run time (see README / docker run).
###############################################################################

ARG CUDA_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

############################  builder  ########################################
FROM ${CUDA_IMAGE} AS builder

# uv: fast, reproducible installs straight from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Dependency layer — only the manifests, so this caches until deps change.
#    Every workspace member's pyproject (+README, referenced by build metadata)
#    must be present for uv to resolve the workspace.
COPY pyproject.toml uv.lock ./
COPY packages/ltx-core/pyproject.toml      packages/ltx-core/README.md      packages/ltx-core/
COPY packages/ltx-pipelines/pyproject.toml packages/ltx-pipelines/README.md packages/ltx-pipelines/
COPY packages/ltx-trainer/pyproject.toml   packages/ltx-trainer/README.md   packages/ltx-trainer/

# Install third-party dependencies only (no workspace packages yet) — best cache.
RUN uv sync --frozen --no-dev --no-install-workspace

# 2) Source layer — the workspace packages themselves.
COPY packages/ packages/
RUN uv sync --frozen --no-dev

############################  runtime  ########################################
FROM ${CUDA_IMAGE} AS runtime

LABEL org.opencontainers.image.source="https://github.com/ethanfel/LTX-2" \
      org.opencontainers.image.description="LTX-2 trainer — torch 2.11 / CUDA 13 (Blackwell-ready)"

# System libraries needed at run time:
#   ffmpeg                  – torchcodec / PyAV video & audio decode
#   libgl1, libglib2.0-0    – OpenCV (opencv-python)
#   libsndfile1             – soundfile audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv-managed interpreter + resolved virtualenv + workspace source (editable).
COPY --from=builder /python /python
COPY --from=builder /app /app

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.hf_cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app/packages/ltx-trainer

# Flexible toolbox image — override the command for the task you need, e.g.:
#   Single-GPU training:
#     docker run --gpus all -v $PWD/run:/workspace ltx2-trainer \
#       python scripts/train.py configs/t2v_lora_low_vram.yaml
#   Multi-GPU / FSDP:
#     docker run --gpus all -v $PWD/run:/workspace ltx2-trainer \
#       accelerate launch scripts/train.py configs/t2v_lora.yaml
#   Dataset preprocessing:
#     docker run --gpus all -v $PWD/run:/workspace ltx2-trainer \
#       python scripts/process_dataset.py ...
CMD ["python", "scripts/train.py", "--help"]
