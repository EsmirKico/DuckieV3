#!/bin/bash

echo "🚀 Starting DuckieBot Autonomous Driving with LM Studio + Qwen"
echo "================================================================"

# Check if LM Studio is accessible
echo "🔗 Testing LM Studio connection..."
curl -s http://192.168.140.1:1234/v1/models > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ LM Studio is accessible!"
else
    echo "❌ Cannot connect to LM Studio at http://192.168.140.1:1234"
    echo "   Please make sure:"
    echo "   1. LM Studio is running"
    echo "   2. A vision model (Qwen2-VL) is loaded"
    echo "   3. Server is accessible on port 1234"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip show fastapi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r server/lmstudio_requirements.txt
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🚗 Starting Autonomous Driving Server with LM Studio..."
echo "   Server will run on: http://0.0.0.0:8000"
echo "   LM Studio URL: http://192.168.140.1:1234"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================================"

# Start the server
python3 server/autonomous_server_lmstudio.py 