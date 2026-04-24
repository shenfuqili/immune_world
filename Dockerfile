ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        build-essential ninja-build git ca-certificates \
        libhdf5-dev pkg-config \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/immune_world

# --- dependency layer (cache-friendly) ----------------------------------------
COPY pyproject.toml README.md ./
COPY src/immune_world/__init__.py src/immune_world/__init__.py
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch>=2.3,<2.5" --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install -e ".[dev]"

# flash-attn separately (has heavy nvcc build; needs torch already installed)
RUN python -m pip install "flash-attn>=2.6,<2.7" --no-build-isolation || \
    echo "flash-attn install skipped (base image lacks matching nvcc); training path uses fallback SDPA."

# --- source layer -------------------------------------------------------------
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY Makefile ./Makefile
COPY LICENSE ./LICENSE

RUN python -m pytest -q tests/test_package_import.py && \
    python -m ruff check . && \
    python -m mypy --strict src/immune_world

CMD ["python", "-m", "immune_world.cli.train", "experiment=_smoke"]
