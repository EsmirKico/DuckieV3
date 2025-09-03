from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import atexit
import os
import requests
import json
from PIL import Image as PILImage
import io
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# LM Studio Configuration
LM_STUDIO_URL = "http://192.168.140.179:8080"
LM_STUDIO_CHAT_ENDPOINT = f"{LM_STUDIO_URL}/v1/chat/completions"

# Video recording variables
video_writer = None
video_initialized = False
video_filename = "autonomous_stream.avi"
video_codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps = 20

# Duckietown parameters
LANE_WIDTH_PIXELS = 150  # Approximate lane width in pixels
SAFE_DISTANCE_THRESHOLD = 0.5  # meters
STOP_LINE_CONFIDENCE_THRESHOLD = 0.8

print("🚀 Using LM Studio for Qwen Vision Model (ROBUST VERSION)")
print(f"📡 LM Studio URL: {LM_STUDIO_URL}")

class LaneInfo(BaseModel):
    lane_center_offset: float  # -1 to 1, where 0 is perfect center
    lane_angle: float  # -45 to 45 degrees, steering angle needed
    confidence: float  # 0 to 1

class ObstacleInfo(BaseModel):
    detected: bool
    type: str  # "duckie", "other"
    position_x: float  # -1 to 1 (left to right)
    distance: float  # meters
    avoidance_direction: str  # "left", "right", "stop"

class StopLineInfo(BaseModel):
    detected: bool
    distance: float  # meters to stop line
    confidence: float

class AutonomousResponse(BaseModel):
    # Lane following
    lane_info: LaneInfo
    
    # Obstacle detection
    obstacle_info: ObstacleInfo
    
    # Stop line detection
    stop_line_info: StopLineInfo
    
    # Driving commands
    steering_angle: float  # -1 to 1 (left to right)
    target_speed: float   # 0 to 1 (0 = stop, 1 = max speed)
    
    # State information
    driving_state: str  # "lane_following", "stopping", "avoiding_obstacle", "stopped"
    debug_info: str
    
    # Optional debug image
    debug_image_base64: Optional[str] = None

def cleanup_resources():
    """Clean up video writer and OpenCV windows"""
    global video_writer
    
    try:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {video_filename}")
        
        cv2.destroyAllWindows()
        print("Autonomous driving stream recording stopped and saved.")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

atexit.register(cleanup_resources)

def write_frame_to_video(frame):
    """Initialize video writer and write frame to video file"""
    global video_writer, video_initialized
    
    try:
        if not video_initialized:
            height, width = frame.shape[:2]
            video_writer = cv2.VideoWriter(video_filename, video_codec, video_fps, (width, height))
            video_initialized = True
            print(f"Started recording autonomous driving stream to {video_filename}")
        
        if video_writer is not None:
            video_writer.write(frame)
    except Exception as e:
        logger.error(f"Error writing video frame: {e}")

def test_lm_studio_connection():
    """Test if LM Studio is accessible"""
    try:
        response = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ LM Studio connected! Available models: {len(models.get('data', []))}")
            return True
        else:
            print(f"❌ LM Studio responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        return False

def get_safe_default_response():
    """Return a safe default response when vision analysis fails"""
    return AutonomousResponse(
        lane_info=LaneInfo(lane_center_offset=0.0, lane_angle=0.0, confidence=0.0),
        obstacle_info=ObstacleInfo(detected=False, type="none", position_x=0.0, distance=10.0, avoidance_direction="none"),
        stop_line_info=StopLineInfo(detected=False, distance=10.0, confidence=0.0),
        steering_angle=0.0,
        target_speed=0.0,
        driving_state="stopped",
        debug_info="Vision analysis failed - using safe defaults"
    )

def analyze_duckietown_scene_lmstudio(image):
    """Use LM Studio Qwen to analyze Duckietown scene"""
    
    try:
        # Convert OpenCV image to base64
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create detailed prompt for Duckietown analysis
        prompt = """You are an AI system controlling a DuckieBot autonomous vehicle. Analyze this Duckietown road scene image and provide a JSON response for autonomous driving decisions.

IMPORTANT: You must respond with ONLY a valid JSON object, no other text.

Analyze the image for:
1. Lane lines: Yellow line (left), white line (right)
2. Lane center position relative to robot
3. Red stop lines across the road
4. Yellow rubber ducks or other obstacles
5. Required steering and speed commands

Respond with this EXACT JSON format:

{
    "lane_lines": {
        "yellow_line_detected": true/false,
        "white_line_detected": true/false,
        "lane_center_offset": number from -1 to 1 (negative=robot left of center, positive=robot right of center, 0=perfect center),
        "steering_angle_needed": number from -45 to 45 (degrees, negative=turn left, positive=turn right),
        "confidence": number from 0 to 1
    },
    "obstacles": {
        "duckie_detected": true/false,
        "duckie_position_x": number from -1 to 1 (left to right in image),
        "duckie_distance": number (estimated meters, 0.1 to 5.0),
        "other_obstacles": true/false,
        "avoidance_action": "left"/"right"/"stop"/"none"
    },
    "stop_line": {
        "red_line_detected": true/false,
        "distance_to_line": number (estimated meters, 0.1 to 5.0),
        "should_stop": true/false,
        "confidence": number from 0 to 1
    },
    "road_state": {
        "on_road": true/false,
        "road_type": "straight"/"curve_left"/"curve_right"/"intersection",
        "visibility": "good"/"poor"
    }
}

Remember: Respond with ONLY the JSON object, no explanations or additional text."""

        # Prepare the request for LM Studio
        payload = {
            "model": "qwen/qwen2.5-vl-7b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
            "stream": False
        }
        
        logger.info("Sending request to LM Studio...")
        
        # Send request to LM Studio
        response = requests.post(
            LM_STUDIO_CHAT_ENDPOINT, 
            json=payload, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            logger.info("LM Studio response received successfully")
            return content.strip()
        else:
            logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error("LM Studio API timeout")
        return None
    except Exception as e:
        logger.error(f"Error calling LM Studio API: {e}")
        logger.error(traceback.format_exc())
        return None

def parse_vision_response(response_text, image_shape):
    """Parse LM Studio Qwen response and convert to driving commands"""
    
    # Initialize default values
    lane_info = LaneInfo(
        lane_center_offset=0.0,
        lane_angle=0.0,
        confidence=0.0
    )
    
    obstacle_info = ObstacleInfo(
        detected=False,
        type="none",
        position_x=0.0,
        distance=10.0,
        avoidance_direction="none"
    )
    
    stop_line_info = StopLineInfo(
        detected=False,
        distance=10.0,
        confidence=0.0
    )
    
    try:
        if not response_text:
            logger.warning("Empty response from LM Studio")
            return lane_info, obstacle_info, stop_line_info
            
        # Clean the response text
        response_text = response_text.strip()
        
        # Remove any markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Try to parse JSON directly
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            # Try to extract JSON from within the text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
            else:
                logger.error(f"Could not find valid JSON in response: {response_text[:200]}...")
                return lane_info, obstacle_info, stop_line_info
        
        # Parse lane information
        if "lane_lines" in analysis:
            lane_data = analysis["lane_lines"]
            lane_info.lane_center_offset = float(lane_data.get("lane_center_offset", 0.0))
            lane_info.lane_angle = float(lane_data.get("steering_angle_needed", 0.0))
            lane_info.confidence = float(lane_data.get("confidence", 0.0))
        
        # Parse obstacle information
        if "obstacles" in analysis:
            obs_data = analysis["obstacles"]
            obstacle_info.detected = obs_data.get("duckie_detected", False) or obs_data.get("other_obstacles", False)
            if obstacle_info.detected:
                obstacle_info.type = "duckie" if obs_data.get("duckie_detected", False) else "other"
                # Handle null values from Qwen
                pos_x = obs_data.get("duckie_position_x", 0.0)
                distance = obs_data.get("duckie_distance", 10.0)
                obstacle_info.position_x = float(pos_x) if pos_x is not None else 0.0
                obstacle_info.distance = float(distance) if distance is not None else 10.0
                obstacle_info.avoidance_direction = obs_data.get("avoidance_action", "none")
        
        # Parse stop line information
        if "stop_line" in analysis:
            stop_data = analysis["stop_line"]
            stop_line_info.detected = stop_data.get("red_line_detected", False)
            # Handle null values from Qwen
            distance = stop_data.get("distance_to_line", 10.0)
            confidence = stop_data.get("confidence", 0.0)
            stop_line_info.distance = float(distance) if distance is not None else 10.0
            stop_line_info.confidence = float(confidence) if confidence is not None else 0.0

        logger.info(f"Parsed successfully - Lane conf: {lane_info.confidence:.2f}, Obstacles: {obstacle_info.detected}")

    except Exception as e:
        logger.error(f"Error parsing LM Studio response: {e}")
        logger.error(f"Raw response: {response_text[:500]}...")
        logger.error(traceback.format_exc())
    
    return lane_info, obstacle_info, stop_line_info

def calculate_driving_commands(lane_info, obstacle_info, stop_line_info):
    """Calculate steering and speed commands based on perception"""
    
    # Default values
    steering_angle = 0.0
    target_speed = 0.6  # Default higher speed for ground movement
    driving_state = "lane_following"
    
    try:
        # Priority 1: Stop line detection
        if stop_line_info.detected and stop_line_info.distance < 0.5 and stop_line_info.confidence > 0.7:
            steering_angle = 0.0
            target_speed = 0.0
            driving_state = "stopped"
            return steering_angle, target_speed, driving_state
        
        # Priority 2: Obstacle avoidance
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
                driving_state = "stopping"
            return steering_angle, target_speed, driving_state
        
        # Priority 3: Lane following
        if lane_info.confidence > 0.3:
            # Convert lane offset and angle to steering command
            steering_angle = -lane_info.lane_center_offset * 0.8  # Proportional to offset
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
        
    except Exception as e:
        logger.error(f"Error calculating driving commands: {e}")
        # Safe fallback
        steering_angle = 0.0
        target_speed = 0.0
        driving_state = "stopped"
    
    return steering_angle, target_speed, driving_state

@app.post("/autonomous_drive", response_model=AutonomousResponse)
async def autonomous_drive(file: UploadFile = File(...)):
    
    logger.info("Received autonomous drive request")
    
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            return get_safe_default_response()
        
        logger.info(f"Processing image of size: {image.shape}")
        
        # Analyze scene with LM Studio Qwen
        vision_response = analyze_duckietown_scene_lmstudio(image)
        
        if vision_response is None:
            logger.error("LM Studio analysis failed")
            return get_safe_default_response()
        
        # Parse vision response
        lane_info, obstacle_info, stop_line_info = parse_vision_response(vision_response, image.shape)
        
        # Calculate driving commands
        steering_angle, target_speed, driving_state = calculate_driving_commands(lane_info, obstacle_info, stop_line_info)
        
        # Create debug visualization
        debug_image = image.copy()
        
        # Draw lane information
        if lane_info.confidence > 0.3:
            center_x = int(image.shape[1] / 2 + lane_info.lane_center_offset * 100)
            cv2.circle(debug_image, (center_x, image.shape[0] - 50), 10, (0, 255, 0), -1)
            cv2.putText(debug_image, f"Lane Offset: {lane_info.lane_center_offset:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw obstacle information
        if obstacle_info.detected:
            obs_x = int(image.shape[1] / 2 + obstacle_info.position_x * 200)
            cv2.rectangle(debug_image, (obs_x-30, 100), (obs_x+30, 160), (0, 0, 255), 3)
            cv2.putText(debug_image, f"Obstacle: {obstacle_info.type}", 
                       (obs_x-30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw stop line information
        if stop_line_info.detected:
            cv2.putText(debug_image, f"STOP LINE - {stop_line_info.distance:.1f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw driving commands
        cv2.putText(debug_image, f"Steering: {steering_angle:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(debug_image, f"Speed: {target_speed:.2f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(debug_image, f"State: {driving_state}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add LM Studio indicator
        cv2.putText(debug_image, "LM Studio + Qwen (ROBUST)", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Write frame to video file
        write_frame_to_video(debug_image)
        
        # Display the debug image
        cv2.imshow("Autonomous Driving (LM Studio + Qwen - ROBUST)", debug_image)
        cv2.waitKey(1)
        
        debug_info = f"LM Studio: {vision_response[:100]}..." if vision_response else "No LM Studio response"
        
        logger.info(f"Successfully processed image - State: {driving_state}, Steering: {steering_angle:.3f}")
        
        return AutonomousResponse(
            lane_info=lane_info,
            obstacle_info=obstacle_info,
            stop_line_info=stop_line_info,
            steering_angle=steering_angle,
            target_speed=target_speed,
            driving_state=driving_state,
            debug_info=debug_info
        )
        
    except Exception as e:
        logger.error(f"Critical error in autonomous_drive: {e}")
        logger.error(traceback.format_exc())
        return get_safe_default_response()

@app.get("/")
def read_root():
    return {"message": "Duckietown Autonomous Driving API with LM Studio + Qwen is running (ROBUST VERSION)"}

@app.get("/test_lmstudio")
def test_lmstudio():
    """Test LM Studio connection"""
    connected = test_lm_studio_connection()
    return {"lm_studio_connected": connected, "url": LM_STUDIO_URL}

if __name__ == "__main__":
    print("🚀 Starting ROBUST Autonomous Driving Server with LM Studio + Qwen")
    
    # Test LM Studio connection on startup
    if test_lm_studio_connection():
        print("✅ LM Studio connection verified!")
    else:
        print("⚠️  Warning: LM Studio not accessible. Please check:")
        print(f"   - LM Studio is running at {LM_STUDIO_URL}")
        print("   - A vision model (Qwen2-VL) is loaded")
        print("   - Network connectivity is working")
    
    import uvicorn
    uvicorn.run(app, host="192.168.140.179", port=8000) 