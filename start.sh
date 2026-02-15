#\!/bin/bash
set -e

echo "Starting vLLM server..."
vllm serve allenai/olmOCR-2-7B-1025-FP8 \
    --port 8001 \
    --served-model-name olmocr \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype auto &

VLLM_PID=$\!

echo "Waiting for vLLM server to be ready..."
TIMEOUT=300
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready\! (took ${ELAPSED}s)"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: vLLM server failed to start within ${TIMEOUT}s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "Starting FastAPI server on port 8080..."
exec uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
