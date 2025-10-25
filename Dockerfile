# Stock Report Generator Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

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
