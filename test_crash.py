#!/usr/bin/env python3

import cv2
import requests
import numpy as np

# Create a simple test image (like a DuckieBot might send)
def create_test_image():
    """Create a simple test image with lane-like features"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw road surface (gray)
    cv2.rectangle(image, (100, 200), (540, 480), (80, 80, 80), -1)
    
    # Draw yellow line (left)
    cv2.line(image, (150, 480), (200, 200), (0, 255, 255), 8)
    
    # Draw white line (right)
    cv2.line(image, (450, 480), (400, 200), (255, 255, 255), 8)
    
    # Add some noise to make it realistic
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def test_autonomous_api():
    """Test the autonomous driving API that's causing crashes"""
    
    print("🧪 Testing autonomous driving API...")
    
    # Create test image
    image = create_test_image()
    
    # Save test image for debugging
    cv2.imwrite('test_image.jpg', image)
    print("📸 Created test image: test_image.jpg")
    
    # Encode image for API
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'file': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        print("📡 Sending request to autonomous API...")
        response = requests.post(
            "http://localhost:8000/autonomous_drive", 
            files=files, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response received!")
            print(f"Driving State: {result.get('driving_state', 'unknown')}")
            print(f"Steering: {result.get('steering_angle', 0.0):.3f}")
            print(f"Speed: {result.get('target_speed', 0.0):.3f}")
            
            lane_info = result.get('lane_info', {})
            print(f"Lane Confidence: {lane_info.get('confidence', 0.0):.2f}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (this might be normal for first request)")
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - server may have crashed")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_autonomous_api() 