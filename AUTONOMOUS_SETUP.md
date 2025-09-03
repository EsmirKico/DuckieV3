# DuckieBot Autonomous Driving with Qwen2.5-VL

Transform your tennis ball following DuckieBot into a fully autonomous vehicle using Qwen2.5 vision model!

## 🚗 Features

- **Lane Following**: Detects yellow/white lane lines and follows Duckietown tracks
- **Stop Line Detection**: Recognizes red stop lines and stops appropriately  
- **Obstacle Avoidance**: Detects duckies and other obstacles, navigates around them
- **Vision Intelligence**: Uses Qwen2.5-VL for sophisticated scene understanding
- **Real-time Control**: Maintains the same high-performance ROS architecture

## 🔧 Setup Instructions

### Part 1: Server Setup (PC with GPU recommended)

1. **Install Qwen2.5-VL Dependencies**:
```bash
cd /home/lesgrossman/ducky/duckie_v2
pip install -r server/autonomous_requirements.txt
```

2. **Update API Configuration** (if needed):
Edit `server/autonomous_server.py` line 29 to change model size:
```python
# For faster inference (recommended for testing):
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# For better accuracy (requires more GPU memory):
# model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

3. **Start the Autonomous Server**:
```bash
python server/autonomous_server.py
```

Expected output:
```
Loading Qwen2.5-VL model...
Model loaded successfully!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Part 2: DuckieBot Setup

1. **Connect to DuckieBot**:
```bash
ssh duckie@[DUCKIEBOT_NAME].local
```

2. **Access Container**:
```bash
docker ps
docker exec -it [CONTAINER_ID] /bin/bash
```

3. **Clone/Update Repository**:
```bash
git clone https://github.com/Sumeet-2023/duckie_v2.git
cd duckie_v2
# OR if already cloned:
# git pull origin main
```

4. **Configure Server IP**:
Edit `src/object_follower/scripts/autonomous_detector.py` line 20:
```python
API_IP = 'YOUR_SERVER_IP_HERE'  # Replace with your PC's IP address
```

5. **Make Scripts Executable**:
```bash
chmod +x src/object_follower/scripts/autonomous_detector.py
chmod +x src/object_follower/scripts/autonomous_monitor.py
chmod +x run_autonomous_driving.sh
```

6. **Launch Autonomous Driving**:
```bash
./run_autonomous_driving.sh
```

## 🎯 How It Works

### System Architecture:
```
DuckieBot Camera → Qwen2.5-VL Server → Autonomous Detector → Motor Controller → Robot Movement
```

### Detection Capabilities:

1. **Lane Detection**:
   - Yellow line (left boundary)
   - White line (right boundary) 
   - Lane center calculation
   - Steering angle computation

2. **Stop Line Detection**:
   - Red lines across the road
   - Distance estimation
   - Automatic stopping

3. **Obstacle Detection**:
   - Yellow rubber duckies
   - Other road obstacles
   - Avoidance path planning

### Control Logic Priority:
1. **Stop Line** (highest priority) → Full stop
2. **Obstacle Avoidance** → Navigate around obstacles
3. **Lane Following** → Follow lane center
4. **Emergency Stop** → Safe fallback

## 📊 Expected Output

### Successful Lane Following:
```
[INFO] 🤖 Autonomous Status: State=lane_following, Lane Offset=-0.12, Confidence=0.85, Speed=0.30, Steering=-0.24
[INFO] Autonomous: State=lane_following, Lane Offset=-0.12, Steering=-0.24, Speed=0.30, Confidence=0.85
```

### Obstacle Detection:
```
[INFO] 🚨 OBSTACLE DETECTED! Starting avoidance maneuver...
[INFO] 🤖 Autonomous Status: State=avoiding_obstacle, Lane Offset=0.05, Confidence=0.72, Speed=0.15, Steering=0.80
```

### Stop Line Detection:
```
[INFO] 🛑 STOP LINE DETECTED! Stopping robot...
[INFO] 🤖 Autonomous Status: State=stopped, Lane Offset=0.02, Confidence=0.91, Speed=0.00, Steering=0.00
```

## ⚙️ Configuration Options

### Speed and Steering Sensitivity:
Edit `src/object_follower/launch/autonomous_driving.launch`:
```xml
<param name="max_speed" value="0.4"/>  <!-- Max speed (0.1-0.8) -->
<param name="kp_lateral" value="0.8"/>  <!-- Steering responsiveness -->
```

### Vision Model Parameters:
Edit `server/autonomous_server.py`:
```python
SAFE_DISTANCE_THRESHOLD = 0.5  # Obstacle avoidance distance
STOP_LINE_CONFIDENCE_THRESHOLD = 0.8  # Stop line detection sensitivity
```

## 🚨 Troubleshooting

### Server Issues:
- **Out of GPU memory**: Use Qwen2.5-VL-3B instead of 7B model
- **Slow inference**: Reduce image resolution in autonomous_detector.py
- **API timeout**: Increase `api_timeout` parameter

### DuckieBot Issues:
- **Poor lane detection**: Improve lighting, clean camera lens
- **Erratic steering**: Reduce `kp_lateral` in launch file
- **Robot not moving**: Check motor topics with `rostopic list`

### Network Issues:
- **API connection failed**: Verify server IP and firewall settings
- **Slow response**: Check network bandwidth and latency

## 🔄 Switching Between Modes

### Tennis Ball Following (Original):
```bash
./run_object_detection.sh
```

### Autonomous Driving (New):
```bash
./run_autonomous_driving.sh
```

## 📈 Performance Monitoring

Monitor system performance:
```bash
# Lane detection rate
rostopic hz /autonomous/lane_info

# Driving commands
rostopic echo /cmd_vel

# System status
rostopic echo /autonomous/driving_state
```

## 🎯 Next Steps

1. **Test in Simulation**: Use Duckietown simulator first
2. **Tune Parameters**: Adjust PID values for your specific track
3. **Add Features**: Implement intersection handling, traffic lights
4. **Safety First**: Always have manual override ready

---

**Ready to Transform Your DuckieBot! 🚀**

Your robot will now intelligently navigate Duckietown tracks, avoid obstacles, and stop at red lines using state-of-the-art vision AI! 