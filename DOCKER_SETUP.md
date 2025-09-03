# 🐳 Qwen Docker Autonomous Driving Setup

This guide explains how to run Qwen 2.5 VL in a Docker container on a Windows PC and connect the DuckieBot to it for autonomous driving.

## 🎯 Architecture Overview

```
DuckieBot (Linux/ROS) ←→ Processing Server (Linux) ←→ Qwen Docker (Windows/vLLM)
```

1. **DuckieBot**: Captures camera images, sends to processing server
2. **Processing Server**: Receives images, forwards to Qwen Docker, processes responses
3. **Qwen Docker**: Runs Qwen 2.5 VL model via vLLM for vision analysis

## 🖥️ Windows PC Setup (Docker)

### Prerequisites
- Windows 10/11 with WSL2
- Docker Desktop for Windows
- NVIDIA GPU with CUDA support (recommended)
- At least 16GB RAM (32GB recommended for 7B model)

### Step 1: Install Docker Desktop
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
2. Install and enable WSL2 integration
3. Enable NVIDIA GPU support in Docker settings

### Step 2: Run Qwen Container
1. Copy `run_qwen_docker.bat` to your Windows PC
2. Open PowerShell as Administrator
3. Navigate to the directory with the batch file
4. Run:
   ```cmd
   run_qwen_docker.bat
   ```

### First Run (Model Download)
The first time you run this, it will:
- Download the vLLM Docker image (~5GB)
- Download Qwen 2.5 VL 7B model (~15GB)
- Start the inference server

**⏱️ Initial setup can take 30-60 minutes depending on internet speed**

### Verify Installation
Open a web browser and go to:
- `http://localhost:8000/docs` - API documentation
- `http://localhost:8000/health` - Health check

## 🐧 Linux Server Setup

### Step 1: Update Configuration
Edit the server configuration to point to your Windows PC:

```bash
# In server/autonomous_server_qwen_docker.py
QWEN_DOCKER_HOST = "192.168.1.100"  # Replace with your Windows PC IP
QWEN_DOCKER_PORT = 8000
```

To find your Windows PC IP:
```cmd
ipconfig
```
Look for the "IPv4 Address" under your network adapter.

### Step 2: Start the Processing Server
```bash
./start_qwen_docker_server.sh
```

This will:
- Check connection to Qwen Docker
- Install Python dependencies
- Start the processing server on port 8002

### Step 3: Test the Setup
```bash
python3 server/test_qwen_docker_api.py
```

This will test:
- ✅ Direct connection to Qwen Docker
- ✅ Server health endpoints
- ✅ Full autonomous driving pipeline

## 🤖 DuckieBot Setup

The DuckieBot configuration automatically uses the Docker setup:
- Server runs on port 8002
- Higher timeouts for Docker processing
- Optimized speeds for ground movement

## 🚀 Running the System

### 1. Start Windows Docker Container
On Windows PC:
```cmd
run_qwen_docker.bat
```
Wait for "Application startup complete" message.

### 2. Start Processing Server
On Linux server:
```bash
./start_qwen_docker_server.sh
```

### 3. Launch DuckieBot
On DuckieBot:
```bash
./run_autonomous_driving.sh
```

## 🔧 Configuration Options

### Docker Container Settings
Edit `run_qwen_docker.bat`:

```cmd
REM Adjust GPU memory
--gpus '"device=0"' ^

REM Change model size
--model Qwen/Qwen2.5-VL-3B-Instruct ^

REM Adjust max context length
--max-model-len 4096 ^
```

### Server Settings
Edit `server/autonomous_server_qwen_docker.py`:

```python
# Connection settings
QWEN_DOCKER_HOST = "your.windows.pc.ip"
QWEN_DOCKER_PORT = 8000

# Performance tuning
LANE_CONFIDENCE_THRESHOLD = 0.6
SAFE_DISTANCE_THRESHOLD = 2.0
```

## 📊 Performance Tips

### For Faster Inference
1. **Use smaller model**: Qwen2.5-VL-3B instead of 7B
2. **Reduce max length**: `--max-model-len 4096`
3. **GPU optimization**: Ensure CUDA is working
4. **More VRAM**: Close other GPU applications

### For Better Accuracy
1. **Use larger model**: Qwen2.5-VL-7B or 14B
2. **Increase context**: `--max-model-len 8192`
3. **Lower temperature**: `temperature: 0.05` in API calls

## 🐛 Troubleshooting

### Docker Container Won't Start
```bash
# Check Docker is running
docker ps

# Check GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Free up memory
docker system prune -a
```

### Connection Issues
```bash
# Test network connectivity
ping your.windows.pc.ip

# Check firewall (Windows)
# Allow port 8000 in Windows Firewall

# Test direct API
curl http://your.windows.pc.ip:8000/health
```

### Poor Performance
1. **Check GPU usage**: Task Manager → Performance → GPU
2. **Monitor memory**: Docker Desktop → Settings → Resources
3. **Reduce concurrent requests**: Lower DuckieBot frame rate
4. **Use CPU only**: Remove `--gpus all` for testing

## 📁 File Structure

```
duckie_v2/
├── run_qwen_docker.bat              # Windows Docker launcher
├── start_qwen_docker_server.sh      # Linux server launcher
├── server/
│   ├── autonomous_server_qwen_docker.py  # Main processing server
│   └── test_qwen_docker_api.py      # Test script
└── src/object_follower/
    ├── scripts/autonomous_detector.py    # DuckieBot client (port 8002)
    └── launch/autonomous_driving.launch  # ROS launch file
```

## 🔄 Migration from LM Studio

If migrating from LM Studio setup:
1. Stop LM Studio server
2. Update server to use Docker (port 8002)
3. Test with `test_qwen_docker_api.py`
4. Launch DuckieBot system

The Docker setup provides:
- ✅ Better performance with vLLM
- ✅ Easier deployment
- ✅ GPU optimization
- ✅ OpenAI-compatible API
- ✅ Better error handling

## 🆘 Support

For issues:
1. Check logs: `docker logs qwen-vl-server`
2. Test API directly: `http://windows.pc.ip:8000/docs`
3. Verify network connectivity between machines
4. Check available GPU memory and system resources

**Happy autonomous driving! 🚗🤖** 