# Docker Setup Guide for Cervical Cancer Prediction API

## What is Docker?

**Docker** is a platform that packages applications and their dependencies into standardized units called **containers**. Think of it as a shipping container for software - everything the app needs (code, libraries, config) is packed together and runs consistently anywhere.

### Key Concepts

| Term | Definition | Analogy |
|------|------------|---------|
| **Dockerfile** | Recipe/instructions to build an image | Recipe book |
| **Image** | Blueprint/template (read-only) | Master copy of software setup |
| **Container** | Running instance of an image | A running program |
| **Volume** | Persistent storage | External hard drive |
| **Port** | Access door to container | Door number on a building |

---

## Files Created

1. **Dockerfile** - Instructions to build the FastAPI app image
2. **docker-compose.yml** - Orchestrates multiple containers (API + MLflow UI)
3. **.dockerignore** - Excludes unnecessary files from build

---

## Prerequisites

### 1. Install Docker Desktop

**Download:** https://www.docker.com/products/docker-desktop/

**Verify installation:**
```bash
docker --version
docker-compose --version
```

### 2. Start Docker Desktop

- Open Docker Desktop application
- Wait for it to fully start (whale icon in system tray)
- Ensure it says "Docker is running"

---

## How to Use

### Option 1: Docker Compose (Recommended - Easiest)

This starts both the FastAPI API and MLflow UI together.

#### Start Services
```bash
# Build images and start containers
docker-compose up --build

# Or run in background (detached mode)
docker-compose up -d --build
```

#### View Logs
```bash
# View all logs
docker-compose logs -f

# View API logs only
docker-compose logs -f api

# View MLflow UI logs only
docker-compose logs -f mlflow-ui
```

#### Stop Services
```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

#### Rebuild After Code Changes
```bash
# Rebuild and restart
docker-compose up --build --force-recreate
```

---

### Option 2: Docker Commands (Manual)

If you prefer more control, use raw Docker commands.

#### Build the Image
```bash
docker build -t cervical-cancer-api .
```

#### Run the API Container
```bash
docker run -d \
  --name cancer-api \
  -p 8000:8000 \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  -v $(pwd)/mlruns:/app/mlruns \
  cervical-cancer-api
```

#### Run MLflow UI Container
```bash
docker run -d \
  --name mlflow-ui \
  -p 5000:5000 \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  -v $(pwd)/mlruns:/app/mlruns \
  -w /app \
  python:3.11-slim \
  bash -c "pip install mlflow && mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000"
```

#### View Logs
```bash
# API logs
docker logs -f cancer-api

# MLflow UI logs
docker logs -f mlflow-ui
```

#### Stop Containers
```bash
docker stop cancer-api mlflow-ui
docker rm cancer-api mlflow-ui
```

---

## Access the Services

Once running, access:

| Service | URL | Description |
|---------|-----|-------------|
| **FastAPI Swagger UI** | http://localhost:8000/docs | Interactive API documentation |
| **MLflow UI** | http://localhost:5000 | Model registry and experiments |
| **API Health Check** | http://localhost:8000/health | Service status |
| **API Models List** | http://localhost:8000/models | Available models |

---

## Understanding the Setup

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Docker Host (Your Computer)                            │
│                                                         │
│  ┌──────────────────┐          ┌──────────────────┐   │
│  │  API Container   │          │  MLflow Container│   │
│  │  Port 8000       │          │  Port 5000       │   │
│  │                  │          │                  │   │
│  │  - FastAPI app   │          │  - MLflow UI     │   │
│  │  - Python 3.11   │          │  - Model browser │   │
│  │  - Dependencies  │          │                  │   │
│  └────────┬─────────┘          └────────┬─────────┘   │
│           │                             │               │
│           └───────────┬─────────────────┘               │
│                       │                                 │
│            ┌──────────▼──────────┐                     │
│            │  Shared Volumes     │                     │
│            │  - mlflow.db        │                     │
│            │  - mlruns/          │                     │
│            └─────────────────────┘                     │
│                                                         │
│  Host Machine:                                          │
│  - ./mlflow.db → /app/mlflow.db (in containers)        │
│  - ./mlruns → /app/mlruns (in containers)              │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request → Host Port 8000 → API Container → Query MLflow DB
                                                    ↓
User Request → Host Port 5000 → MLflow Container → Read MLflow DB
```

### Volumes (Persistent Storage)

```
Host Machine              Container
─────────────             ──────────
./mlflow.db  ──────────→  /app/mlflow.db  (SQLite database)
./mlruns/    ──────────→  /app/mlruns/    (Model artifacts)
```

**Why volumes?**
- Data persists even if container is deleted
- Can share data between containers
- Easy to backup and migrate

---

## Common Commands

### Docker Compose
```bash
# Start services
docker-compose up --build

# Start in background
docker-compose up -d --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart a specific service
docker-compose restart api

# Execute command in running container
docker-compose exec api python --version
```

### Docker Images
```bash
# List images
docker images

# Remove image
docker rmi cervical-cancer-api

# Remove all unused images
docker image prune -a
```

### Docker Containers
```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# View container logs
docker logs -f cervical-cancer-api

# Execute command in container
docker exec -it cervical-cancer-api bash

# Stop container
docker stop cervical-cancer-api

# Remove container
docker rm cervical-cancer-api
```

### Docker Volumes
```bash
# List volumes
docker volume ls

# Remove unused volumes
docker volume prune
```

---

## Troubleshooting

### Problem: "Docker is not running"
**Solution:** Start Docker Desktop application

### Problem: "Port 8000 already in use"
**Solution:** Change port in docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # Use 8001 on host instead
```

### Problem: "Permission denied" on volumes
**Solution:** Check file permissions on host machine

### Problem: "Image build fails"
**Solution:** 
```bash
# Rebuild without cache
docker-compose build --no-cache

# Or remove old images first
docker-compose down
docker system prune -a
docker-compose up --build
```

### Problem: "Container keeps restarting"
**Solution:** Check logs for errors
```bash
docker logs cervical-cancer-api
```

---

## Testing the Deployment

### 1. Test API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_name": "Linear_SVM",
  "model_version": 1,
  "model_stage": "Production",
  "features_count": 29
}
```

### 2. Test Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\":25,\"number_of_sexual_partners\":1,...}"
```

### 3. Test MLflow UI
Open browser: http://localhost:5000

You should see the MLflow UI with all registered models.

---

## Production Deployment

### Security Considerations

1. **Don't use root user** - Create non-root user in Dockerfile:
```dockerfile
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

2. **Use specific tags** - Don't use `latest`:
```yaml
image: python:3.11.5-slim
```

3. **Scan for vulnerabilities**:
```bash
docker scan cervical-cancer-api
```

4. **Use secrets management** - Don't hardcode credentials

### Performance Optimization

1. **Use multi-stage builds** - Reduce image size
2. **Optimize layer caching** - Order Dockerfile commands strategically
3. **Use .dockerignore** - Exclude unnecessary files
4. **Choose slim images** - Smaller base images

### Deployment Options

| Platform | How to Deploy |
|----------|---------------|
| **AWS ECS** | Push image to ECR, deploy to ECS |
| **Google Cloud Run** | Push to GCR, deploy to Cloud Run |
| **Azure Container Instances** | Push to ACR, deploy to ACI |
| **Kubernetes** | Push to registry, create K8s manifests |
| **Heroku** | `heroku container:push web` |
| **DigitalOcean App Platform** | Connect GitHub repo |

---

## Next Steps

1. ✅ Install Docker Desktop
2. ✅ Test locally with `docker-compose up --build`
3. ✅ Verify both services are running
4. ✅ Test predictions via Swagger UI
5. ✅ Push image to registry (Docker Hub, AWS ECR, etc.)
6. ✅ Deploy to cloud platform

---

## Quick Reference Card

```bash
# Start everything
docker-compose up --build

# Stop everything
docker-compose down

# Rebuild after changes
docker-compose up --build --force-recreate

# View logs
docker-compose logs -f

# Access points
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
```

---

## Benefits You Get

✅ **Consistency** - Same environment everywhere (dev, staging, prod)
✅ **Portability** - Run on any machine with Docker
✅ **Isolation** - No conflicts with other software
✅ **Scalability** - Easy to run multiple instances
✅ **Version control** - Images can be tagged and versioned
✅ **Fast deployment** - One command to start everything
✅ **Easy rollback** - Switch to previous image version instantly