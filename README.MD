# OlmOCR Deployment on GCP

OCR API powered by [allenai/olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) model, deployed on Google Cloud Platform.

## 📋 Prerequisites

- Google Cloud Platform account
- `gcloud` CLI installed and configured
- Docker installed (for local testing)
- Git

## 🏗️ Project Structure

```
olmocr-deployment/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── .dockerignore         # Docker build exclusions
├── .gitignore           # Git exclusions
├── cloudbuild.yaml      # GCP Cloud Build configuration
├── app.yaml             # App Engine configuration (optional)
└── README.md            # This file
```

## 🚀 Deployment Options

### Option 1: Cloud Run (Recommended)

Cloud Run is serverless, scales to zero, and is cost-effective for variable workloads.

#### Step 1: Set up GCP Project

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE_NAME="olmocr-api"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

#### Step 2: Build and Deploy

```bash
# Build the container
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --memory 8Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --allow-unauthenticated
```

#### Step 3: Get Service URL

```bash
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
```

### Option 2: Google Kubernetes Engine (GKE)

For production workloads requiring more control.

#### Step 1: Create GKE Cluster

```bash
gcloud container clusters create olmocr-cluster \
  --region $REGION \
  --machine-type n1-standard-4 \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5
```

#### Step 2: Build and Push Image

```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
```

#### Step 3: Deploy to GKE

```bash
# Get cluster credentials
gcloud container clusters get-credentials olmocr-cluster --region $REGION

# Deploy
kubectl create deployment olmocr --image=gcr.io/$PROJECT_ID/$SERVICE_NAME

# Expose service
kubectl expose deployment olmocr --type LoadBalancer --port 80 --target-port 8080
```

### Option 3: Compute Engine (VM-based)

For persistent instances with full control.

#### Step 1: Create VM Instance

```bash
gcloud compute instances create olmocr-vm \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --boot-disk-size 50GB \
  --image-family debian-11 \
  --image-project debian-cloud \
  --tags http-server,https-server
```

#### Step 2: SSH and Setup

```bash
# SSH into instance
gcloud compute ssh olmocr-vm --zone us-central1-a

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io git
sudo usermod -aG docker $USER

# Clone repo and build
git clone <your-repo-url>
cd olmocr-deployment
sudo docker build -t olmocr .
sudo docker run -d -p 80:8080 olmocr
```

## 🧪 Local Testing

```bash
# Build Docker image
docker build -t olmocr .

# Run container
docker run -p 8080:8080 olmocr

# Test the API
curl http://localhost:8080/health
```

## 📝 API Usage

### Health Check

```bash
curl https://your-service-url.run.app/health
```

### Single Image OCR

```bash
curl -X POST https://your-service-url.run.app/ocr \
  -F "file=@/path/to/image.jpg"
```

### Batch OCR

```bash
curl -X POST https://your-service-url.run.app/ocr/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Python Example

```python
import requests

url = "https://your-service-url.run.app/ocr"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 💰 Cost Optimization

### Cloud Run
- **Scales to zero**: No charges when idle
- **Pay per request**: Only charged for actual usage
- **Memory**: 8GB recommended for model loading
- **Estimated cost**: ~$0.10-0.50 per hour of active use

### Tips
1. Use Cloud Run for variable workloads
2. Set max instances to control costs
3. Monitor usage with Cloud Monitoring
4. Consider reserved instances for consistent loads

## 🔧 Configuration

### Environment Variables

You can customize the deployment with these environment variables:

```bash
# In Dockerfile or deployment config
ENV MODEL_NAME="allenai/olmOCR-2-7B-1025-FP8"
ENV MAX_NEW_TOKENS=512
ENV PORT=8080
```

### Scaling

For Cloud Run:
```bash
gcloud run services update $SERVICE_NAME \
  --min-instances 1 \
  --max-instances 20 \
  --concurrency 10
```

## 📊 Monitoring

```bash
# View logs
gcloud run services logs read $SERVICE_NAME --region $REGION

# Monitor metrics
gcloud monitoring dashboards list
```

## 🐛 Troubleshooting

### Model loading timeout
- Increase `--timeout` in Cloud Run deployment
- Use larger instance size

### Out of memory
- Increase `--memory` to 16Gi or 32Gi
- Reduce batch size

### Slow response
- Add `--min-instances 1` to keep instance warm
- Use GPU-enabled instances for GKE/GCE

## 🔐 Security

### Enable Authentication

```bash
# Require authentication
gcloud run services update $SERVICE_NAME \
  --no-allow-unauthenticated \
  --region $REGION
```

### API Key Protection

Add API key middleware to `app.py`:

```python
from fastapi import Header, HTTPException

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return await call_next(request)
```

## 📚 Resources

- [OlmOCR Model Card](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)

## 📄 License

Follow the license terms of the OlmOCR model and your GCP usage agreement.