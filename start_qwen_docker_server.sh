#!/bin/bash

echo "🚀 Starting Qwen Docker Autonomous Driving Server"
echo "=================================================="

# Check if Qwen Docker is accessible
echo "🔍 Checking connection to Qwen Docker container..."
QWEN_URL="http://192.168.140.179:8080"

if curl -s "$QWEN_URL/health" > /dev/null 2>&1; then
    echo "✅ Qwen Docker container is accessible at $QWEN_URL"
else
    echo "❌ Cannot connect to Qwen Docker container at $QWEN_URL"
    echo ""
    echo "🔧 Setup Instructions:"
    echo "1. On Windows PC, run: run_qwen_docker.bat"
    echo "2. Wait for model download (first time only)"
    echo "3. Update QWEN_DOCKER_HOST in server if needed"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies if needed
echo "📦 Installing Python dependencies..."
pip3 install fastapi uvicorn opencv-python numpy pydantic python-multipart requests pillow

echo ""
echo "🚀 Starting Qwen Docker server on port 8000..."
echo "📡 Connecting to Qwen Docker at: $QWEN_URL"
echo "🌐 Server will be available at: http://0.0.0.0:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 server/autonomous_server_qwen_docker.py 