# ==============================================================================
# Production Docker Container for Stage 3 Mamba & MambaLog Experiments
# ==============================================================================
# Multi-stage GPU container with pre-compiled mamba-ssm CUDA kernels
# ensuring zero-setup reproducibility across cloud and cluster environments.
# ==============================================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/stage3-mamba

# Upgrade pip and install build prerequisites
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# Install PyTorch 2.4.1 with CUDA 12.1 support (required for @torch.library.custom_op in mamba-ssm 2.3+)
RUN python3 -m pip install --no-cache-dir torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies with --no-build-isolation so CUDA kernels (mamba-ssm/causal-conv1d) see pre-installed torch
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy complete project codebase
COPY . /workspace/stage3-mamba/

# Set Python path for module imports
ENV PYTHONPATH=/workspace/stage3-mamba:/workspace/stage1-gpt2:/workspace:$PYTHONPATH

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh run_paper_experiments.sh run.sh run_all_3_stages_comparison.sh

# Default entrypoint runs verification or custom command
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["--fast-dev-run"]
