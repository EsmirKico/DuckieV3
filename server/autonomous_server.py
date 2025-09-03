from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import atexit
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image as PILImage
import io

app = FastAPI()

# Video recording variables
video_writer = None
video_initialized = False
video_filename = "autonomous_stream.avi"
video_codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps = 20

# Load Qwen2.5-VL model
print("Loading Qwen2.5-VL model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", 
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print("Model loaded successfully!")

# Duckietown parameters
LANE_WIDTH_PIXELS = 150  # Approximate lane width in pixels
SAFE_DISTANCE_THRESHOLD = 0.5  # meters
STOP_LINE_CONFIDENCE_THRESHOLD = 0.8

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
    
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {video_filename}")
    
    cv2.destroyAllWindows()
    print("Autonomous driving stream recording stopped and saved.")

atexit.register(cleanup_resources)

def write_frame_to_video(frame):
    """Initialize video writer and write frame to video file"""
    global video_writer, video_initialized
    
    if not video_initialized:
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(video_filename, video_codec, video_fps, (width, height))
        video_initialized = True
        print(f"Started recording autonomous driving stream to {video_filename}")
    
    if video_writer is not None:
        video_writer.write(frame)

def analyze_duckietown_scene(image):
    """Use Qwen2.5-VL to analyze Duckietown scene"""
    
    # Convert OpenCV image to PIL
    pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create detailed prompt for Duckietown analysis
    prompt = """Analyze this Duckietown road scene carefully. Provide analysis in this exact JSON format:

{
    "lane_lines": {
        "yellow_line_detected": true/false,
        "white_line_detected": true/false, 
        "lane_center_offset": number from -1 to 1 (negative=left, positive=right, 0=centered),
        "steering_angle_needed": number from -45 to 45 (degrees to correct),
        "confidence": number from 0 to 1
    },
    "obstacles": {
        "duckie_detected": true/false,
        "duckie_position_x": number from -1 to 1 (left to right in image),
        "duckie_distance": number (estimated meters),
        "other_obstacles": true/false,
        "avoidance_action": "left"/"right"/"stop"/"none"
    },
    "stop_line": {
        "red_line_detected": true/false,
        "distance_to_line": number (estimated meters),
        "should_stop": true/false,
        "confidence": number from 0 to 1
    },
    "road_state": {
        "on_road": true/false,
        "road_type": "straight"/"curve_left"/"curve_right"/"intersection",
        "visibility": "good"/"poor"
    }
}

Look for:
- Yellow line on the left, white line on the right (standard Duckietown)
- Red lines across the road (stop lines)
- Yellow rubber duckies or other obstacles
- Road curvature and navigation needs

Be precise with measurements and confident in your analysis."""

    # Process with Qwen2.5-VL
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text_inputs,
            images=[pil_image],
            return_tensors="pt"
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response_text
        
    except Exception as e:
        print(f"Error in Qwen2.5-VL analysis: {e}")
        return None

def parse_vision_response(response_text, image_shape):
    """Parse Qwen2.5-VL response and convert to driving commands"""
    
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
        # Try to extract JSON from response
        import json
        import re
        
        # Look for JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            analysis = json.loads(json_str)
            
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
                    obstacle_info.position_x = float(obs_data.get("duckie_position_x", 0.0))
                    obstacle_info.distance = float(obs_data.get("duckie_distance", 10.0))
                    obstacle_info.avoidance_direction = obs_data.get("avoidance_action", "none")
            
            # Parse stop line information
            if "stop_line" in analysis:
                stop_data = analysis["stop_line"]
                stop_line_info.detected = stop_data.get("red_line_detected", False)
                stop_line_info.distance = float(stop_data.get("distance_to_line", 10.0))
                stop_line_info.confidence = float(stop_data.get("confidence", 0.0))
    
    except Exception as e:
        print(f"Error parsing vision response: {e}")
        print(f"Raw response: {response_text}")
    
    return lane_info, obstacle_info, stop_line_info

def calculate_driving_commands(lane_info, obstacle_info, stop_line_info):
    """Calculate steering and speed commands based on perception"""
    
    # Default values
    steering_angle = 0.0
    target_speed = 0.3  # Default moderate speed
    driving_state = "lane_following"
    
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
            target_speed = 0.2  # Slow down
            driving_state = "avoiding_obstacle"
        elif obstacle_info.avoidance_direction == "right":
            steering_angle = 0.8  # Turn right
            target_speed = 0.2  # Slow down
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
            target_speed = 0.2  # Slow down for sharp turns
        else:
            target_speed = 0.4  # Normal speed
        
        driving_state = "lane_following"
    else:
        # Poor lane detection - stop safely
        steering_angle = 0.0
        target_speed = 0.0
        driving_state = "stopped"
    
    return steering_angle, target_speed, driving_state

@app.post("/autonomous_drive", response_model=AutonomousResponse)
async def autonomous_drive(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return AutonomousResponse(
            lane_info=LaneInfo(lane_center_offset=0.0, lane_angle=0.0, confidence=0.0),
            obstacle_info=ObstacleInfo(detected=False, type="none", position_x=0.0, distance=10.0, avoidance_direction="none"),
            stop_line_info=StopLineInfo(detected=False, distance=10.0, confidence=0.0),
            steering_angle=0.0,
            target_speed=0.0,
            driving_state="stopped",
            debug_info="Invalid image"
        )
    
    # Analyze scene with Qwen2.5-VL
    vision_response = analyze_duckietown_scene(image)
    
    if vision_response is None:
        return AutonomousResponse(
            lane_info=LaneInfo(lane_center_offset=0.0, lane_angle=0.0, confidence=0.0),
            obstacle_info=ObstacleInfo(detected=False, type="none", position_x=0.0, distance=10.0, avoidance_direction="none"),
            stop_line_info=StopLineInfo(detected=False, distance=10.0, confidence=0.0),
            steering_angle=0.0,
            target_speed=0.0,
            driving_state="stopped",
            debug_info="Vision analysis failed"
        )
    
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
    
    # Write frame to video file
    write_frame_to_video(debug_image)
    
    # Display the debug image
    cv2.imshow("Autonomous Driving (Server Side)", debug_image)
    cv2.waitKey(1)
    
    debug_info = f"Vision: {vision_response[:100]}..." if vision_response else "No vision response"
    
    return AutonomousResponse(
        lane_info=lane_info,
        obstacle_info=obstacle_info,
        stop_line_info=stop_line_info,
        steering_angle=steering_angle,
        target_speed=target_speed,
        driving_state=driving_state,
        debug_info=debug_info
    )

@app.get("/")
def read_root():
    return {"message": "Duckietown Autonomous Driving API with Qwen2.5-VL is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 