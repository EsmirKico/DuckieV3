#!/usr/bin/env python3
"""
🚀 QWEN DOCKER AUTONOMOUS DRIVING SERVER
Direct connection to Qwen 2.5 VL running in Docker container with vLLM
"""

import uvicorn
import cv2
import numpy as np
import base64
import io
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
QWEN_DOCKER_HOST = "192.168.140.179"  # Your Windows PC IP
QWEN_DOCKER_PORT = 8080
QWEN_API_URL = f"http://{QWEN_DOCKER_HOST}:{QWEN_DOCKER_PORT}/completion"

# Control parameters
LANE_CONFIDENCE_THRESHOLD = 0.6
SAFE_DISTANCE_THRESHOLD = 2.0
MAX_STEERING_ANGLE = 1.0

# === DATA MODELS ===
@dataclass
class LaneInfo:
    confidence: float
    lane_center_offset: float
    lane_angle: float
    detected: bool

@dataclass
class ObstacleInfo:
    detected: bool
    distance: float
    position_x: float
    avoidance_direction: str

class AutonomousDriveResponse(BaseModel):
    steering_angle: float
    target_speed: float
    driving_state: str
    lane_confidence: float
    obstacle_detected: bool
    obstacle_distance: float
    emergency_stop: bool
    debug_info: Dict[str, Any]

# === GLOBAL STATE ===
app = FastAPI(title="Qwen Docker Autonomous Driving Server", version="1.0.0")

# Video recording
video_writer = None
video_filename = None

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string for API transmission"""
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise

def query_qwen_docker(image_b64: str) -> Dict[str, Any]:
    """Query Qwen 2.5 VL running in Docker container via llama.cpp server"""
    try:
        logger.info("Sending request to Qwen Docker container (llama.cpp)...")
        
        # Prepare the request payload for llama.cpp server (fixed prompt)
        prompt = f"""<|im_start|>system
You are an autonomous driving system for a DuckieBot. Analyze camera images and provide precise driving commands.<|im_end|>
<|im_start|>user
<img src="data:image/jpeg;base64,{image_b64}">

Analyze this DuckieBot camera image and provide driving commands.

Look for:
1. LANE LINES: Yellow/white lines marking the road
2. OBSTACLES: Duckies, other robots, people in the path
3. STOP LINES: Red lines across the road
4. TRAFFIC SIGNS: Stop signs, turn indicators

Respond with ONLY valid JSON:
{{"lane_detected": true, "lane_confidence": 0.8, "lane_center_offset": 0.0, "lane_angle": 0.0, "obstacle_detected": false, "obstacle_distance": 10.0, "obstacle_position_x": 0.0, "stop_line_detected": false, "recommended_speed": 0.5, "recommended_steering": 0.0, "avoidance_direction": "stop", "driving_state": "lane_following"}}<|im_end|>
<|im_start|>assistant"""

        payload = {
            "prompt": prompt,
            "n_predict": 500,
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.9,
            "stop": ["<|im_end|>"],
            "stream": False
        }
        
        # Make the API call
        response = requests.post(
            QWEN_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Qwen API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="Qwen API request failed")
        
        result = response.json()
        logger.info("Qwen Docker response received successfully")
        
        # Extract the content from llama.cpp response
        content = result.get("content", "")
        
        # Parse the JSON response
        try:
            # Clean up the response - extract JSON part
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                vision_data = json.loads(json_content)
                logger.info(f"Parsed Qwen response: {vision_data}")
                return vision_data
            else:
                raise json.JSONDecodeError("No JSON found in response", content, 0)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Qwen JSON response: {e}")
            logger.error(f"Raw content: {content}")
            # Return safe defaults
            return {
                "lane_detected": False,
                "lane_confidence": 0.0,
                "lane_center_offset": 0.0,
                "lane_angle": 0.0,
                "obstacle_detected": False,
                "obstacle_distance": 10.0,
                "obstacle_position_x": 0.0,
                "stop_line_detected": False,
                "recommended_speed": 0.3,
                "recommended_steering": 0.0,
                "avoidance_direction": "stop",
                "driving_state": "stopped"
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Qwen API: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling Qwen API: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def parse_vision_response(vision_data: Dict[str, Any]) -> Tuple[LaneInfo, ObstacleInfo]:
    """Parse Qwen vision response into structured data"""
    try:
        # Parse lane information
        lane_info = LaneInfo(
            confidence=float(vision_data.get("lane_confidence", 0.0)),
            lane_center_offset=float(vision_data.get("lane_center_offset", 0.0)),
            lane_angle=float(vision_data.get("lane_angle", 0.0)),
            detected=bool(vision_data.get("lane_detected", False))
        )
        
        # Parse obstacle information
        obstacle_info = ObstacleInfo(
            detected=bool(vision_data.get("obstacle_detected", False)),
            distance=float(vision_data.get("obstacle_distance", 10.0)),
            position_x=float(vision_data.get("obstacle_position_x", 0.0)),
            avoidance_direction=str(vision_data.get("avoidance_direction", "stop"))
        )
        
        return lane_info, obstacle_info
        
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing vision response: {e}")
        # Return safe defaults
        return (
            LaneInfo(confidence=0.0, lane_center_offset=0.0, lane_angle=0.0, detected=False),
            ObstacleInfo(detected=False, distance=10.0, position_x=0.0, avoidance_direction="stop")
        )

def calculate_driving_commands(lane_info: LaneInfo, obstacle_info: ObstacleInfo, vision_data: Dict[str, Any]) -> Tuple[float, float, str]:
    """Calculate steering angle, target speed, and driving state"""
    try:
        # Default values
        steering_angle = 0.0
        target_speed = 0.6  # Default higher speed for ground movement
        driving_state = "lane_following"
        
        # Handle stop line detection
        if vision_data.get("stop_line_detected", False):
            steering_angle = 0.0
            target_speed = 0.0
            driving_state = "stopped"
            logger.info("Stop line detected - stopping")
            return steering_angle, target_speed, driving_state
        
        # Handle lane following if lane is detected with good confidence
        if lane_info.detected and lane_info.confidence > LANE_CONFIDENCE_THRESHOLD:
            # Calculate steering based on lane position and angle
            steering_angle = -lane_info.lane_center_offset * 0.8  # Position correction
            steering_angle += -lane_info.lane_angle * 0.02  # Add angle correction
            
            # Limit steering angle
            steering_angle = np.clip(steering_angle, -1.0, 1.0)
            
            # Adjust speed based on curvature
            if abs(steering_angle) > 0.5:
                target_speed = 0.4  # Moderate speed for sharp turns
            else:
                target_speed = 0.7  # Higher normal speed for ground movement
            
            driving_state = "lane_following"
        else:
            # Poor lane detection - stop safely
            steering_angle = 0.0
            target_speed = 0.0
            driving_state = "stopped"
        
        # Handle obstacle avoidance (overrides lane following)
        if obstacle_info.detected and obstacle_info.distance < SAFE_DISTANCE_THRESHOLD:
            if obstacle_info.avoidance_direction == "left":
                steering_angle = -0.8  # Turn left
                target_speed = 0.4  # Moderate speed for obstacle avoidance
                driving_state = "avoiding_obstacle"
            elif obstacle_info.avoidance_direction == "right":
                steering_angle = 0.8  # Turn right
                target_speed = 0.4  # Moderate speed for obstacle avoidance
                driving_state = "avoiding_obstacle"
            else:  # stop
                steering_angle = 0.0
                target_speed = 0.0
                driving_state = "stopped"
        
        # Apply recommended values from Qwen if they seem reasonable
        recommended_speed = vision_data.get("recommended_speed", target_speed)
        recommended_steering = vision_data.get("recommended_steering", steering_angle)
        
        if 0.0 <= recommended_speed <= 1.0:
            target_speed = max(target_speed, recommended_speed * 0.7)  # Scale up for ground movement
        
        if -1.0 <= recommended_steering <= 1.0:
            # Blend with calculated steering
            steering_angle = 0.7 * steering_angle + 0.3 * recommended_steering
        
        # Final safety limits
        steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        target_speed = np.clip(target_speed, 0.0, 0.8)
        
        return steering_angle, target_speed, driving_state
        
    except Exception as e:
        logger.error(f"Error calculating driving commands: {e}")
        # Safe fallback
        return 0.0, 0.0, "stopped"

def start_recording(image_shape):
    """Start video recording"""
    global video_writer, video_filename
    
    try:
        video_filename = f"autonomous_stream.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 5.0, (image_shape[1], image_shape[0]))
        logger.info(f"Started recording autonomous driving stream to {video_filename}")
    except Exception as e:
        logger.error(f"Failed to start video recording: {e}")

def record_frame(image):
    """Record a frame to video"""
    global video_writer
    
    try:
        if video_writer is not None:
            video_writer.write(image)
    except Exception as e:
        logger.error(f"Failed to record frame: {e}")

def stop_recording():
    """Stop video recording"""
    global video_writer, video_filename
    
    try:
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            logger.info(f"Video saved to {video_filename}")
    except Exception as e:
        logger.error(f"Failed to stop video recording: {e}")

@app.post("/autonomous_drive", response_model=AutonomousDriveResponse)
async def autonomous_drive(file: UploadFile = File(...)):
    """Main autonomous driving endpoint"""
    try:
        logger.info("Received autonomous drive request")
        
        # Read and decode image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        logger.info(f"Processing image of size: {image.shape}")
        
        # Start recording if not already started
        if video_writer is None:
            start_recording(image.shape)
        
        # Record this frame
        record_frame(image)
        
        # Encode image for API
        image_b64 = encode_image_to_base64(image)
        
        # Query Qwen
        vision_data = query_qwen_docker(image_b64)
        
        # Parse response
        lane_info, obstacle_info = parse_vision_response(vision_data)
        
        logger.info(f"Parsed successfully - Lane conf: {lane_info.confidence:.2f}, Obstacles: {obstacle_info.detected}")
        
        # Calculate driving commands
        steering_angle, target_speed, driving_state = calculate_driving_commands(lane_info, obstacle_info, vision_data)
        
        logger.info(f"Successfully processed image - State: {driving_state}, Steering: {steering_angle:.3f}")
        
        # Prepare response
        response = AutonomousDriveResponse(
            steering_angle=steering_angle,
            target_speed=target_speed,
            driving_state=driving_state,
            lane_confidence=lane_info.confidence,
            obstacle_detected=obstacle_info.detected,
            obstacle_distance=obstacle_info.distance,
            emergency_stop=False,
            debug_info={
                "lane_center_offset": lane_info.lane_center_offset,
                "lane_angle": lane_info.lane_angle,
                "obstacle_position_x": obstacle_info.position_x,
                "avoidance_direction": obstacle_info.avoidance_direction,
                "vision_response": vision_data
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in autonomous_drive: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return safe emergency response
        return AutonomousDriveResponse(
            steering_angle=0.0,
            target_speed=0.0,
            driving_state="error",
            lane_confidence=0.0,
            obstacle_detected=True,
            obstacle_distance=0.0,
            emergency_stop=True,
            debug_info={"error": str(e), "error_type": str(type(e))}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test connection to Qwen Docker (llama.cpp health endpoint)
        test_response = requests.get(f"http://{QWEN_DOCKER_HOST}:{QWEN_DOCKER_PORT}/health", timeout=5)
        qwen_status = "connected" if test_response.status_code == 200 else "disconnected"
    except:
        qwen_status = "disconnected"
    
    return {
        "status": "healthy",
        "qwen_docker_status": qwen_status,
        "qwen_url": f"http://{QWEN_DOCKER_HOST}:{QWEN_DOCKER_PORT}",
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("🚀 Starting Autonomous Driving Server with Qwen Docker")
    logger.info(f"📡 Qwen Docker URL: http://{QWEN_DOCKER_HOST}:{QWEN_DOCKER_PORT}")
    
    # Test connection to Qwen Docker
    try:
        test_response = requests.get(f"http://{QWEN_DOCKER_HOST}:{QWEN_DOCKER_PORT}/health", timeout=5)
        if test_response.status_code == 200:
            logger.info("✅ Qwen Docker (llama.cpp) connection verified!")
        else:
            logger.warning("⚠️ Qwen Docker health check failed")
    except Exception as e:
        logger.warning(f"⚠️ Could not connect to Qwen Docker: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    stop_recording()
    logger.info("Autonomous driving stream recording stopped and saved.")

if __name__ == "__main__":
    print("🚀 Using Qwen Docker for Vision Model")
    print(f"📡 Qwen Docker URL: http://{QWEN_DOCKER_HOST}:{QWEN_DOCKER_PORT}")
    print("🚀 Starting Autonomous Driving Server with Qwen Docker")
    
    uvicorn.run(app, host="192.168.140.144", port=8000) 