from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from typing import List, Optional
import atexit
import os

app = FastAPI()

# Video recording variables
video_writer = None
video_initialized = False
video_filename = "detection_stream.avi"
video_codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps = 20

# Load your trained YOLOv8 model
model = YOLO("model/best.pt")

# Camera parameters (same as your robot)
FOCAL_LENGTH = 525.0  # pixels
KNOWN_OBJECT_WIDTH = 0.067  # meters (tennis ball diameter ~6.7cm)

class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int

class DetectionResponse(BaseModel):
    target_found: bool
    detections: List[Detection]
    best_detection: Optional[Detection] = None
    # Normalized coordinates for robot control (-1 to 1)
    target_position_x: Optional[float] = None  
    target_position_y: Optional[float] = None
    estimated_distance: Optional[float] = None
    debug_image_base64: Optional[str] = None

def cleanup_resources():
    """Clean up video writer and OpenCV windows"""
    global video_writer
    
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {video_filename}")
    
    cv2.destroyAllWindows()
    print("Detection stream recording stopped and saved.")

# Register cleanup function to be called at program exit
atexit.register(cleanup_resources)

def write_frame_to_video(frame):
    """Initialize video writer and write frame to video file"""
    global video_writer, video_initialized
    
    if not video_initialized:
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(video_filename, video_codec, video_fps, (width, height))
        video_initialized = True
        print(f"Started recording detection stream to {video_filename}")
    
    if video_writer is not None:
        video_writer.write(frame)

def estimate_distance(pixel_width):
    """Estimate distance using camera parameters"""
    if pixel_width > 0:
        distance = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH) / pixel_width
        return max(0.1, min(distance, 5.0))  # Clamp between 10cm and 5m
    return 0.0

@app.post("/detect", response_model=DetectionResponse)
async def detect_tennis_ball(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return DetectionResponse(target_found=False, detections=[])
    
    # Run YOLOv8 inference
    results = model(image, conf=0.5, imgsz=416)  # confidence threshold
    
    detections = []
    best_detection = None
    best_confidence = 0.0
    
    # Process detections
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                
                detection = Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(conf),
                    center_x=center_x, center_y=center_y,
                    width=width, height=height
                )
                detections.append(detection)
                
                # Track best detection (highest confidence)
                if conf > best_confidence:
                    best_confidence = float(conf)
                    best_detection = detection
    
    # Calculate robot control values if detection found
    target_found = len(detections) > 0
    target_position_x = None
    target_position_y = None
    estimated_distance = None
    
    if target_found and best_detection is not None:
        img_height, img_width = image.shape[:2]
        target_position_x = (best_detection.center_x - img_width // 2) / (img_width // 2)
        target_position_y = (best_detection.center_y - img_height // 2) / (img_height // 2)
        estimated_distance = estimate_distance(best_detection.width)
        cv2.rectangle(image, (best_detection.x1, best_detection.y1), 
                     (best_detection.x2, best_detection.y2), (0, 255, 0), 2)
        cv2.circle(image, (best_detection.center_x, best_detection.center_y), 5, (0, 0, 255), -1)
        cv2.putText(image, f"Dist: {estimated_distance:.2f}m", 
                   (best_detection.x1, best_detection.y1-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Conf: {best_detection.confidence:.2f}", 
                   (best_detection.x1, best_detection.y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Write frame to video file (regardless of detection status)
    write_frame_to_video(image)
    
    # Display the marked detection in a persistent window
    cv2.imshow("Detection Stream (Server Side)", image)
    cv2.waitKey(1)
    
    # Encode debug image as base64 (optional, for debugging)
    _, buffer = cv2.imencode('.jpg', image)
    
    return DetectionResponse(
        target_found=target_found,
        detections=detections,
        best_detection=best_detection,
        target_position_x=target_position_x,
        target_position_y=target_position_y,
        estimated_distance=estimated_distance,
    )

@app.get("/")
def read_root():
    return {"message": "Tennis Ball Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
