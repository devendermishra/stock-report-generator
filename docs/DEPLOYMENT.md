# Deployment Documentation

This document provides comprehensive instructions for deploying the Stock Report Generator application across different environments and platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Manual Deployment](#manual-deployment)
- [API Server Deployment](#api-server-deployment)
- [UI Deployment](#ui-deployment)
- [Production Deployment](#production-deployment)
- [GPU-Accelerated Deployment](#gpu-accelerated-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Scaling Considerations](#scaling-considerations)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10/11
- **Python**: 3.10 or higher (3.11+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 2GB free space
- **CPU**: 2 cores minimum (4+ cores recommended)

**For GPU Support:**
- **NVIDIA GPU**: CUDA Compute Capability 6.0+
- **CUDA Toolkit**: 11.8 or 12.0
- **GPU Memory**: 4GB minimum (8GB+ recommended)
- **nvidia-docker2**: For Docker GPU support

### Required Software

1. **Python 3.10+** with pip
2. **Docker** (version 20.10+) - for containerized deployment
3. **Docker Compose** (version 2.0+) - for multi-container orchestration
4. **Git** - for cloning the repository

### API Keys

- **OpenAI API Key**: Required for LLM functionality
  - Get your key from: https://platform.openai.com/api-keys
  - Minimum account tier: Pay-as-you-go

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd stock-report-generator
```

### 2. Environment Variables

Create a `.env` file from the example:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: Model Configuration
DEFAULT_MODEL=gpt-4o-mini
MAX_TOKENS=4000
TEMPERATURE=0.1

# Optional: File Paths
OUTPUT_DIR=reports
TEMP_DIR=temp

# Optional: Rate Limiting
API_RATE_LIMIT_PER_MINUTE=2
MAX_REQUESTS_PER_MINUTE=60
REQUEST_DELAY=1.0

# Optional: Circuit Breaker Configuration
CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
CIRCUIT_BREAKER_TIME_WINDOW_SECONDS=120
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=60

# Optional: Metrics (Prometheus)
ENABLE_METRICS=false
METRICS_PORT=8000
```

### 3. Create Required Directories

```bash
mkdir -p reports data/inputs data/outputs data/processed data/raw temp logs
```

## Docker Deployment

### Standard Docker Deployment

#### 1. Build the Docker Image

```bash
docker build -f docker/Dockerfile -t stock-report-generator:latest .
```

#### 2. Run the Container

**Basic Usage:**
```bash
docker run --rm \
  -e OPENAI_API_KEY=your_api_key_here \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/data:/app/data \
  stock-report-generator:latest RELIANCE
```

**With Environment File:**
```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/data:/app/data \
  stock-report-generator:latest RELIANCE
```

**Interactive Mode:**
```bash
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/data:/app/data \
  stock-report-generator:latest bash
```

#### 3. Docker Run Options

**Volume Mounts:**
- `-v $(pwd)/reports:/app/reports` - Mount reports directory
- `-v $(pwd)/data:/app/data` - Mount data directory
- `-v $(pwd)/logs:/app/logs` - Mount logs directory (optional)

**Environment Variables:**
- `-e OPENAI_API_KEY=...` - Set API key directly
- `--env-file .env` - Load from .env file

**Port Mapping (for API):**
- `-p 8000:8000` - Expose API on port 8000

**Resource Limits:**
```bash
docker run --rm \
  --memory="4g" \
  --cpus="2" \
  --env-file .env \
  stock-report-generator:latest RELIANCE
```

## Docker Compose Deployment

### Standard Docker Compose

#### 1. Update docker-compose.yml

Ensure your `docker-compose.yml` includes environment variables:

```yaml
version: '3.8'

services:
  stock-report-generator:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: stock-report-generator
    volumes:
      - ./reports:/app/reports
      - ./data:/app/data
      - ./temp:/app/temp
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app/src
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    ports:
      - "8000:8000"
    command: python -m src.main
```

#### 2. Start Services

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

#### 3. Run Commands

```bash
# Execute a report generation
docker-compose run --rm stock-report-generator python -m src.main RELIANCE

# Access container shell
docker-compose exec stock-report-generator bash
```

**Note:** For UI deployment with Docker Compose, see the [UI Deployment](#ui-deployment) section for complete setup instructions including UI service configuration.


## Manual Deployment

### 1. Install Python Dependencies

**Using pip:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Using editable install:**
```bash
pip install -e .
```

**With GPU support:**
```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

### 2. Verify Installation

```bash
python -c "import langchain, langgraph, openai; print('Installation successful!')"
```

### 3. Run the Application

**CLI Mode:**
```bash
cd src
python main.py RELIANCE
```

**With Options:**
```bash
python main.py RELIANCE --skip-ai
python main.py RELIANCE --export-graph graph.png
```

**From Project Root:**
```bash
python -m src.main RELIANCE
```

## API Server Deployment

### 1. Start the API Server

**Development Mode:**
```bash
cd src
python api.py
```

**Using uvicorn directly:**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Production Mode:**
```bash
uvicorn src.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --no-access-log
```

### 2. Docker API Deployment

**Build API Image:**
```bash
docker build -f docker/Dockerfile -t stock-report-api:latest .
```

**Run API Container:**
```bash
docker run -d \
  --name stock-report-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/reports:/app/reports \
  stock-report-api:latest \
  uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 3. API Endpoints

Once deployed, the API provides:

- `GET /` - API information
- `GET /health` - Health check endpoint
- `GET /report/{symbol}` - Generate report (returns markdown)
- `GET /report/{symbol}?s=true` - Generate report with structured workflow
- `POST /pdf` - Convert markdown to PDF

**Example Usage:**
```bash
# Generate report
curl http://localhost:8000/report/RELIANCE

# Health check
curl http://localhost:8000/health


### 4. API with Reverse Proxy (Nginx)

**Nginx Configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

## UI Deployment

The Stock Report Generator includes a modern web-based UI for generating and viewing stock reports. The UI is a static web application that communicates with the API server.

### 1. UI Overview

The UI consists of:
- **Static HTML/CSS/JavaScript** files in the `ui/` directory
- **No build process required** - uses CDN for dependencies
- **Connects to API** at configurable endpoint
- **Features:**
  - Stock symbol input
  - Real-time report generation
  - Markdown rendering with syntax highlighting
  - Download as Markdown or PDF
  - Responsive design

### 2. Development Setup

#### Simple Local Server

**Using Python:**
```bash
cd ui
python3 -m http.server 3000
# Open http://localhost:3000 in browser
```

**Using Node.js:**
```bash
npx http-server ui -p 3000
# Open http://localhost:3000 in browser
```


#### Configure API Endpoint

Edit `ui/app.js` and update the API base URL:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change to your API URL
```

For production, use the full domain:
```javascript
const API_BASE_URL = 'https://api.yourdomain.com';
```

### 3. Docker Deployment for UI

#### Option 1: Nginx Container

Create `docker/Dockerfile.ui`:

```dockerfile
FROM nginx:alpine

# Copy UI files
COPY ui/ /usr/share/nginx/html/

# Copy custom nginx config (optional)
COPY docker/nginx-ui.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

Create `docker/nginx-ui.conf`:

```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Serve static files
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

**Build and Run:**
```bash
docker build -f docker/Dockerfile.ui -t stock-report-ui:latest .
docker run -d \
  --name stock-report-ui \
  -p 3000:80 \
  stock-report-ui:latest
```

#### Option 2: Simple HTTP Server Container

Create `docker/Dockerfile.ui-simple`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir http-server

# Copy UI files
COPY ui/ ./ui/

WORKDIR /app/ui

EXPOSE 3000

CMD ["python", "-m", "http.server", "3000"]
```

**Build and Run:**
```bash
docker build -f docker/Dockerfile.ui-simple -t stock-report-ui:simple .
docker run -d \
  --name stock-report-ui \
  -p 3000:3000 \
  stock-report-ui:simple
```

### 4. Docker Compose with UI

Update `docker-compose.yml` to include UI service:

```yaml
version: '3.8'

services:
  # API Service
  stock-report-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: stock-report-api
    volumes:
      - ./reports:/app/reports
      - ./data:/app/data
      - ./temp:/app/temp
    environment:
      - PYTHONPATH=/app/src
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    ports:
      - "8000:8000"
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000

  # UI Service
  stock-report-ui:
    build:
      context: .
      dockerfile: docker/Dockerfile.ui
    container_name: stock-report-ui
    ports:
      - "3000:80"
    depends_on:
      - stock-report-api
    environment:
      - API_BASE_URL=http://stock-report-api:8000
```

**Note:** For the UI to connect to the API in Docker Compose, you may need to:
1. Update `app.js` to use the API service name, or
2. Use environment variable injection at build time, or
3. Configure a reverse proxy (see below)

### 5. Production Deployment with Nginx Reverse Proxy

Deploy both API and UI behind a single Nginx reverse proxy:

**Nginx Configuration (`/etc/nginx/sites-available/stock-report`):**

```nginx
# Upstream API server
upstream api_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name yourdomain.com;

    # UI - Serve static files
    location / {
        root /var/www/stock-report-ui;
        try_files $uri $uri/ /index.html;
        index index.html;
    }

    # API - Proxy to backend
    location /api/ {
        proxy_pass http://api_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS headers (if needed)
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        
        # Timeouts for long-running requests
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://api_backend/health;
    }

    # Static assets caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        root /var/www/stock-report-ui;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

**Deployment Steps:**

1. **Copy UI files:**
```bash
sudo mkdir -p /var/www/stock-report-ui
sudo cp -r ui/* /var/www/stock-report-ui/
```

2. **Update API URL in app.js:**
```javascript
const API_BASE_URL = '/api'; // Use relative path for same domain
```

3. **Enable Nginx site:**
```bash
sudo ln -s /etc/nginx/sites-available/stock-report /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Environment-Based Configuration

For different environments, you can use environment variable injection:

**Build-time configuration (Docker):**

Create `docker/Dockerfile.ui-env`:

```dockerfile
FROM nginx:alpine

COPY ui/ /usr/share/nginx/html/

# Create script to inject environment variables
RUN echo '#!/bin/sh' > /docker-entrypoint.sh && \
    echo 'envsubst < /usr/share/nginx/html/app.js.template > /usr/share/nginx/html/app.js' >> /docker-entrypoint.sh && \
    echo 'exec nginx -g "daemon off;"' >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
```

**Runtime configuration (JavaScript):**

Update `app.js` to read from window config:

```javascript
// Read from window object (set by server)
const API_BASE_URL = window.API_BASE_URL || 'http://localhost:8000';
```

Inject in `index.html`:

```html
<script>
    window.API_BASE_URL = '{{API_BASE_URL}}';
</script>
<script src="app.js"></script>
```

### 7. Cloud Platform Deployment

#### AWS S3 + CloudFront

1. **Build UI:**
```bash
cd ui
# No build needed, just copy files
```

2. **Upload to S3:**
```bash
aws s3 sync ui/ s3://your-bucket-name/ --delete
```

3. **Configure CloudFront:**
   - Origin: S3 bucket
   - Behaviors: Cache static assets
   - Custom error pages: Route 404 to index.html

4. **Update API URL:**
   - Point to your API endpoint (API Gateway, ALB, etc.)

#### Netlify/Vercel

1. **Deploy UI:**
```bash
# Netlify
netlify deploy --dir=ui --prod

# Vercel
vercel --cwd ui
```

2. **Configure environment variables:**
   - Set `API_BASE_URL` in platform settings

3. **Update app.js** to use environment variable:
```javascript
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';
```

**Note:** For static hosting, you'll need to inject the API URL at build time or use a config file.

#### GitHub Pages

1. **Create `ui/config.js`:**
```javascript
window.API_BASE_URL = 'https://your-api-domain.com';
```

2. **Update `index.html` to load config:**
```html
<script src="config.js"></script>
<script src="app.js"></script>
```

3. **Deploy:**
```bash
git subtree push --prefix ui origin gh-pages
```

### 8. CORS Configuration

If UI and API are on different domains, configure CORS in the API:

The API already includes CORS middleware. For production, update `src/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-ui-domain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 9. UI Troubleshooting

#### API Connection Issues

**Error:** `Failed to generate report: Network error`

**Solutions:**
1. Verify API is running: `curl http://localhost:8000/health`
2. Check API URL in `app.js`
3. Check browser console for CORS errors
4. Verify firewall/security group allows connections

#### CORS Errors

**Error:** `Access to fetch at '...' has been blocked by CORS policy`

**Solutions:**
1. Update CORS settings in API (`src/api.py`)
2. Use same domain for UI and API
3. Configure reverse proxy to serve both

#### Static File Not Found

**Error:** `404 Not Found` for JS/CSS files

**Solutions:**
1. Verify file paths are correct
2. Check web server configuration
3. Ensure files are copied to deployment directory

#### PDF Download Fails

**Error:** `PDF generation failed`

**Solutions:**
1. Verify API `/pdf` endpoint is accessible
2. Check API logs for errors
3. Verify rate limits aren't exceeded
4. Check API has sufficient resources

### 10. UI Production Checklist

- [ ] API URL configured correctly
- [ ] CORS configured if needed
- [ ] HTTPS enabled
- [ ] Static assets cached
- [ ] Error handling tested
- [ ] Mobile responsiveness verified
- [ ] Browser compatibility tested
- [ ] Security headers configured
- [ ] Analytics/monitoring added (optional)

## Production Deployment

### 1. Production Considerations

**Environment Variables:**
- Set `ENABLE_METRICS=true` for monitoring
- Configure appropriate rate limits
- Use secure API key management (secrets manager)

**Resource Allocation:**
- Allocate sufficient memory (8GB+)
- Use multiple CPU cores (4+)
- Consider GPU for faster processing

**Logging:**
- Configure centralized logging
- Set appropriate log levels
- Rotate logs regularly

**Security:**
- Use HTTPS/TLS
- Implement authentication/authorization
- Restrict API access
- Secure environment variables

### 2. Production Docker Deployment

**Production Dockerfile (example):**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY setup.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Build and Deploy:**
```bash
docker build -f docker/Dockerfile.prod -t stock-report-generator:prod .
docker run -d \
  --name stock-report-prod \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env.prod \
  -v /path/to/reports:/app/reports \
  stock-report-generator:prod
```


## Cloud Platform Deployment

### AWS Deployment

#### EC2 Instance

1. **Launch EC2 Instance:**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.medium or larger
   - Security Group: Allow port 8000

2. **Install Dependencies:**
```bash
sudo apt update
sudo apt install -y python3-pip docker.io docker-compose
sudo usermod -aG docker $USER
```

3. **Deploy Application:**
```bash
git clone <repository-url>
cd stock-report-generator
docker-compose up -d
```

#### ECS (Elastic Container Service)

1. **Build and Push Image:**
```bash
aws ecr create-repository --repository-name stock-report-generator
docker build -t stock-report-generator .
docker tag stock-report-generator:latest <account-id>.dkr.ecr.<region>.amazonaws.com/stock-report-generator:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/stock-report-generator:latest
```

2. **Create ECS Task Definition** with:
   - Image: Your ECR image
   - Environment variables from Secrets Manager
   - Port mapping: 8000
   - Resource limits

3. **Deploy Service** with Application Load Balancer

### Google Cloud Platform (GCP)

#### Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/stock-report-generator

# Deploy to Cloud Run
gcloud run deploy stock-report-generator \
  --image gcr.io/PROJECT-ID/stock-report-generator \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key \
  --memory 4Gi \
  --cpu 2
```

#### Compute Engine

Similar to AWS EC2 deployment steps.

### Azure Deployment

#### Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry <registry-name> --image stock-report-generator:latest .

# Deploy to Container Instances
az container create \
  --resource-group <resource-group> \
  --name stock-report-generator \
  --image <registry-name>.azurecr.io/stock-report-generator:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server <registry-name>.azurecr.io \
  --environment-variables OPENAI_API_KEY=your-key \
  --ports 8000
```

#### Azure App Service

1. Create App Service with Container support
2. Configure environment variables
3. Deploy container image

## Monitoring and Health Checks

### 1. Health Check Endpoint

The API provides a health check endpoint:

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "generator_ai_initialized": true,
  "generator_structured_initialized": true,
  "circuit_breaker": {
    "state": "closed",
    "failure_count": 0,
    "config": {
      "failure_threshold": 3,
      "time_window_seconds": 120,
      "recovery_timeout_seconds": 60
    }
  }
}
```

### 2. Prometheus Metrics

Enable metrics in `.env`:
```bash
ENABLE_METRICS=true
METRICS_PORT=8000
```

Access metrics:
```bash
curl http://localhost:8000/metrics
```

### 3. Logging

**Log Files:**
- `logs/stock_report_generator.log` - Application logs
- `logs/prompts.log` - LLM prompts and responses (if enabled)

**Log Rotation:**
Configure logrotate or use container log management.

### 4. Docker Health Checks

The Dockerfile includes health checks. Monitor with:

```bash
docker ps  # Shows health status
docker inspect <container-id> | grep Health
```

## Scaling Considerations

### Horizontal Scaling

**Load Balancer Configuration:**
- Use multiple API instances behind a load balancer
- Configure sticky sessions if needed
- Set appropriate timeouts for long-running requests

**Docker Swarm:**
```bash
docker swarm init
docker stack deploy -c docker-compose.yml stock-report
docker service scale stock-report_stock-report-generator=3
```

**Kubernetes:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-report-generator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stock-report-generator
  template:
    metadata:
      labels:
        app: stock-report-generator
    spec:
      containers:
      - name: stock-report-generator
        image: stock-report-generator:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```

### Vertical Scaling

- Increase container memory and CPU limits
- Optimize model selection (faster models for higher throughput)

### Database Considerations

Currently, the application is stateless. For production:
- Consider adding a database for report caching
- Store session state for long-running requests
- Implement job queue for async processing

## Security Considerations

### 1. API Security

**Rate Limiting:**
- Configured via `API_RATE_LIMIT_PER_MINUTE`
- Default: 2 requests per minute per IP

**Authentication:**
- Add API key authentication for production
- Implement OAuth2/JWT if needed
- Use HTTPS/TLS

**CORS Configuration:**
- Update CORS settings in `src/api.py` for production
- Restrict allowed origins

### 2. Environment Variables

**Secure Storage:**
- Use secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
- Never commit `.env` files
- Rotate API keys regularly

**Docker Secrets:**
```bash
echo "your-api-key" | docker secret create openai_api_key -
```

### 3. Container Security

- Run containers as non-root user
- Use minimal base images
- Scan images for vulnerabilities
- Keep dependencies updated

### 4. Network Security

- Use private networks for containers
- Implement firewall rules
- Use VPN for internal services
- Enable DDoS protection

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Error:** `OPENAI_API_KEY not found`

**Solution:**
```bash
# Check environment variable
echo $OPENAI_API_KEY

# Verify .env file
cat .env | grep OPENAI_API_KEY

# Set in Docker
docker run -e OPENAI_API_KEY=your-key ...
```

#### 2. Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn src.api:app --port 8001
```

#### 3. Docker Build Fails

**Error:** Build errors or dependency issues

**Solution:**
```bash
# Clean build
docker build --no-cache -f docker/Dockerfile -t stock-report-generator .

# Check Dockerfile syntax
docker build --dry-run ...
```

#### 4. GPU Not Available

**Error:** `CUDA not available`

**Solution:**
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify container has GPU access
docker run --rm --gpus all stock-report-generator:gpu \
  python -c "import torch; print(torch.cuda.is_available())"
```

#### 5. Memory Issues

**Error:** Out of memory errors

**Solution:**
- Increase container memory limits
- Use smaller models
- Process reports in batches
- Enable swap space

#### 6. Circuit Breaker Open

**Error:** `Service temporarily unavailable - Circuit breaker is open`

**Solution:**
- Check underlying service health
- Wait for recovery timeout
- Review failure logs
- Adjust circuit breaker thresholds

### Debugging

**Enable Debug Logging:**
```bash
# Set log level
export LOG_LEVEL=DEBUG

# Or in .env
LOG_LEVEL=DEBUG
```

**View Container Logs:**
```bash
docker logs stock-report-generator
docker logs -f stock-report-generator  # Follow logs
```

**Access Container Shell:**
```bash
docker exec -it stock-report-generator bash
```

**Test API Endpoints:**
```bash
# Health check
curl http://localhost:8000/health

# Generate report
curl http://localhost:8000/report/RELIANCE

# Check metrics
curl http://localhost:8000/metrics
```

### Performance Tuning

**Optimize Model Selection:**
- Use `gpt-4o-mini` for faster responses
- Adjust `MAX_TOKENS` based on needs
- Configure appropriate `TEMPERATURE`

**Resource Allocation:**
- Allocate more CPU cores for parallel processing
- Increase memory for larger reports
- Use GPU for faster LLM inference

**Caching:**
- Implement report caching
- Cache API responses
- Use Redis for distributed caching

## Additional Resources

- [Main README](../README.md) - Project overview and usage
- [API Documentation](../README.md#api-documentation) - API reference
- [Agent Specialization](AGENT_SPECIALIZATION.md) - Agent details

## Support

For deployment issues:
1. Check logs in `logs/` directory
2. Review this documentation
3. Check [GitHub Issues](https://github.com/devendermishra/stock-report-generator/issues)
4. Create a new issue with deployment details

---

**Last Updated:** 2024-01-01
**Version:** 1.0.0

