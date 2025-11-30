# Stock Report Generator Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# GPU Support (Optional)
# Uncomment the following lines for GPU support
# Note: Requires nvidia-docker runtime and CUDA-compatible GPU
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# 
# For GPU support, use nvidia/cuda base image instead:
# FROM nvidia/cuda:11.8-devel-ubuntu22.04
# 
# GPU Requirements:
# - NVIDIA GPU with CUDA Compute Capability 6.0+
# - nvidia-docker2 or Docker with GPU support
# - CUDA Toolkit 11.8+ installed on host

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-lock.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy source code
COPY src/ ./src/
COPY setup.py ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p reports data images temp

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for web interface)
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.main"]
