# Use official PyTorch image with CUDA devel to get nvcc compiler and runtime support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Avoid interactive prompts during apt packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for Python performance and runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace

# Copy and install python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entrypoint script and make it executable
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy the rest of the application code
COPY . /workspace/

# Set PYTHONPATH so internal modules resolve cleanly
ENV PYTHONPATH=/workspace

# Use the entrypoint to fix WSL2 driver library paths at runtime
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["bash"]
