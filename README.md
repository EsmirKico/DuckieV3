# DuckieBot Object Following System - User Guide

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the System](#running-the-system)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)

## Overview

This system enables DuckieBot to autonomously follow objects, particularly tennis balls, using computer vision and AI. The system provides:

- YOLOv8-based object detection via API
- Enhanced PID control for smooth following behavior
- Real-time performance monitoring
- Obstacle avoidance and safety features
- Kalman filter tracking for robust object following

## Prerequisites

### Hardware Requirements

- DuckieBot with functional camera
- Network connectivity (WiFi)
- Tennis ball for testing

### Software Requirements

- DuckieBot with `dt-duckiebot-interface` container running
- Server machine with YOLOv8 API running on port 8000
- SSH access to DuckieBot

### Network Setup

- DuckieBot and server machine on the same network
- Server machine IP address

## Setup Instructions

### Step 1: Connect to DuckieBot

```bash
ssh duckie@[DUCKIEBOT_NAME].local
```

Replace `[DUCKIEBOT_NAME]` with your DuckieBot's hostname.

### Step 2: Access Container

Check running containers:
```bash
docker ps
```

Locate the container with image `duckietown/dt-duckiebot-interface` and note its Container ID.

Enter the container:
```bash
docker exec -it [CONTAINER_ID] /bin/bash
```

### Step 3: Clone Repository

```bash
git clone https://github.com/Sumeet-2023/duckie_v2.git
cd duckie_v2
```

### Step 4: Configure API Server

Open the configuration file:
```bash
nano src/object_follower/scripts/enhanced_object_detector.py
```

Update the server IP address by changing the following variables:
```python
API_IP = '172.20.10.3'
API_PORT = 8000
API_ENDPOINT = '/detect'
```

Replace `API_IP` with your actual server IP address.

### Step 5: Launch System

```bash
./run_object_detection.sh
```

### Step 6: Test Functionality

Place a tennis ball in front of the DuckieBot camera. The robot should detect and begin following the object.

## Running the System

### Expected Startup Output

```
[INFO] Enhanced Object Detector initialized - Using API: http://YOUR_IP:8000/detect
[INFO] Enhanced Motor Controller - DuckieBot deployment ready
[INFO] Performance Monitor started - Tracking system metrics
```

### Successful Detection

```
[INFO] TARGET DETECTED!
[INFO] Following: pos=(0.23,0.15) dist=0.85m → vel=(0.15,-0.45)
[INFO] Target: True | Stability: 78.5%
```

### Performance Monitoring

The system continuously reports:
- Detection rate (Hz)
- Stability score (0-100%)
- Control performance metrics
- Position tracking accuracy

## Configuration Options

### Target Distance

Modify following distance in launch files:
```xml
<param name="target_distance" value="0.15"/>  <!-- Distance in meters -->
```

### Control Sensitivity

Adjust PID parameters for different behaviors:
```xml
<param name="kp_lateral" value="1.2"/>
<param name="ki_lateral" value="0.08"/>
<param name="kd_lateral" value="0.4"/>
```

### Speed Settings

```xml
<param name="max_speed" value="0.20"/>  <!-- Max speed in m/s -->
```

### Safety Parameters

```xml
<param name="safe_distance_pixels" value="40"/>  <!-- Obstacle avoidance -->
```

## Troubleshooting

### API Connection Issues

**Symptoms:** "API call failed" or "API call timed out"

**Solutions:**
- Verify server IP address configuration
- Ensure YOLOv8 API server is running on port 8000
- Test network connectivity: `ping YOUR_SERVER_IP`
- Check server firewall settings

### Camera Feed Problems

**Symptoms:** "No camera feed" or "Waiting for data"

**Solutions:**
- Check camera hardware connection
- Restart DuckieBot interface: `docker restart [CONTAINER_ID]`
- Verify camera topics: `rostopic list | grep camera`

### Motor Control Issues

**Symptoms:** Robot not moving despite detection

**Solutions:**
- Check motor driver status: `rostopic echo /[DUCKIEBOT_NAME]/wheels_driver_node/wheels_cmd`
- Verify joystick override is disabled
- Check obstacle detection: `rostopic echo /object_follower/obstacle_detected`

### Poor Tracking Performance

**Symptoms:** Unstable or erratic following behavior

**Solutions:**
- Improve lighting conditions
- Use bright, contrasting objects
- Adjust PID parameters in launch file
- Reduce maximum speed for smoother control

### Debug Commands

Monitor system status:
```bash
# List active topics
rostopic list

# Monitor detection status
rostopic echo /object_follower/target_found

# Check performance metrics
rostopic echo /object_follower/performance

# Verify camera feed rate
rostopic hz /[DUCKIEBOT_NAME]/camera_node/image/compressed
```
