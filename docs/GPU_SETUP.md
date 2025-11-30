# GPU Setup Guide for Stock Report Generator

This guide explains how to set up GPU acceleration for the Stock Report Generator to improve AI model performance.

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA Compute Capability 6.0 or higher
- **Minimum 4GB GPU memory** (8GB+ recommended for large models)
- **Compatible GPU models**: GTX 1060, RTX 2060, RTX 3070, RTX 4080, A100, etc.

### Software Requirements
- **CUDA Toolkit 11.8 or 12.0**
- **cuDNN 8.6 or higher**
- **NVIDIA drivers 520.61.05 or higher**
- **Python 3.10+**

## Installation Steps

### 1. Install CUDA Toolkit

#### Ubuntu/Debian:
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Install CUDA Toolkit
sudo apt update
sudo apt install cuda-toolkit-11-8
```

#### Windows:
1. Download CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow the setup wizard
3. Add CUDA to your PATH environment variable

### 2. Install cuDNN

1. Download cuDNN from [NVIDIA Developer](https://developer.nvidia.com/cudnn)
2. Extract and copy files to CUDA installation directory
3. Verify installation: `nvcc --version`

### 3. Install GPU Dependencies

```bash
# Install GPU-specific requirements
pip install -r requirements-gpu.txt

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"
```

## Docker GPU Support

### 1. Install nvidia-docker2

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Build GPU-enabled Docker Image

```bash
# Use the GPU-enabled Dockerfile
docker build -f docker/Dockerfile.gpu -t stock-report-generator:gpu .

# Run with GPU support
docker run --gpus all -it stock-report-generator:gpu
```

## Configuration

### Environment Variables

Add to your `.env` file:
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # Use first GPU (0,1,2,3 for multiple GPUs)
TF_FORCE_GPU_ALLOW_GROWTH=true  # Allow GPU memory growth
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimize memory usage
```

### Python Configuration

```python
# Enable GPU memory growth for TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set PyTorch device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Performance Optimization

### 1. Memory Management
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Monitor GPU usage
import gpustat
gpustat.monitor()
```

### 2. Batch Processing
```python
# Use larger batch sizes for GPU processing
batch_size = 32 if torch.cuda.is_available() else 8
```

### 3. Mixed Precision Training
```python
# Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    # Your model forward pass
    pass
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Clear cache: `torch.cuda.empty_cache()`

2. **Driver version mismatch**
   - Update NVIDIA drivers
   - Check CUDA compatibility

3. **Docker GPU not working**
   - Verify nvidia-docker2 installation
   - Check Docker daemon configuration

### Verification Commands

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check Python GPU support
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check GPU memory
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

## Performance Benchmarks

### Expected Speedup
- **Text Processing**: 2-3x faster
- **Model Inference**: 5-10x faster
- **Batch Processing**: 3-5x faster

### Memory Usage
- **Minimum**: 4GB GPU memory
- **Recommended**: 8GB+ GPU memory
- **Large Models**: 16GB+ GPU memory

## Support

For GPU-related issues:
1. Check NVIDIA driver compatibility
2. Verify CUDA installation
3. Review GPU memory usage
4. Check system requirements

## References

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [PyTorch GPU Support](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [Docker GPU Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
