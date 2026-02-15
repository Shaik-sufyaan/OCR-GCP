#!/bin/bash

# OlmOCR Deployment Script for GCP Cloud Run (with GPU)
# This script automates the deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
REGION="us-east4"
SERVICE_NAME="olmocr-api"
MEMORY="32Gi"
CPU="8"
GPU="1"
GPU_TYPE="nvidia-l4"
MAX_INSTANCES="5"
TIMEOUT="300"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    print_error "No GCP project selected. Please set a project:"
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

print_info "Using GCP Project: $PROJECT_ID"
print_info "Region: $REGION"
print_info "Service Name: $SERVICE_NAME"
print_info "GPU: $GPU x $GPU_TYPE"
print_info "Memory: $MEMORY | CPU: $CPU"

# Confirm deployment
read -p "Do you want to proceed with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Deployment cancelled."
    exit 0
fi

# Enable required APIs
print_info "Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the container
print_info "Building container image (this may take 10-15 minutes)..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME --timeout=1800s

if [ $? -ne 0 ]; then
    print_error "Container build failed!"
    exit 1
fi

print_info "Container built successfully!"

# Deploy to Cloud Run with GPU
print_info "Deploying to Cloud Run with GPU..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --memory $MEMORY \
  --cpu $CPU \
  --gpu $GPU \
  --gpu-type $GPU_TYPE \
  --timeout $TIMEOUT \
  --max-instances $MAX_INSTANCES \
  --no-cpu-throttling \
  --allow-unauthenticated

if [ $? -ne 0 ]; then
    print_error "Deployment to Cloud Run failed!"
    exit 1
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format 'value(status.url)')

print_info "Deployment successful!"
echo ""
echo "=========================================="
print_info "Service URL: $SERVICE_URL"
echo "=========================================="
echo ""
print_info "Test your API with:"
echo "curl $SERVICE_URL/health"
echo ""
print_info "OCR an image:"
echo "curl -X POST $SERVICE_URL/ocr -F 'file=@image.jpg'"
echo ""
print_info "OCR a PDF (specific pages):"
echo "curl -X POST '$SERVICE_URL/ocr/pdf?pages=1-3' -F 'file=@document.pdf'"
echo ""
