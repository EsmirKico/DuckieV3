#!/usr/bin/env python3

import cv2
import requests
import numpy as np

def test_robust_server():
    """Test the robust server with error handling"""
    
    print("🧪 Testing ROBUST autonomous driving server...")
    
    # Create test image (like a DuckieBot might send)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw road surface (gray)
    cv2.rectangle(image, (100, 200), (540, 480), (80, 80, 80), -1)
    
    # Draw yellow line (left)
    cv2.line(image, (150, 480), (200, 200), (0, 255, 255), 8)
    
    # Draw white line (right)
    cv2.line(image, (450, 480), (400, 200), (255, 255, 255), 8)
    
    # Encode image for API
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'file': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        print("📡 Sending request to ROBUST server (port 8001)...")
        response = requests.post(
            "http://localhost:8001/autonomous_drive", 
            files=files, 
            timeout=45  # Longer timeout for Qwen2.5-VL-7B
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ROBUST server response received!")
            print(f"State: {result.get('driving_state', 'unknown')}")
            print(f"Steering: {result.get('steering_angle', 0.0):.3f}")
            print(f"Speed: {result.get('target_speed', 0.0):.3f}")
            
            lane_info = result.get('lane_info', {})
            print(f"Lane Confidence: {lane_info.get('confidence', 0.0):.2f}")
            print(f"Lane Offset: {lane_info.get('lane_center_offset', 0.0):.3f}")
            
            debug_info = result.get('debug_info', 'No debug info')
            print(f"Debug: {debug_info[:100]}...")
            
            print("🎉 ROBUST server is working properly!")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - Qwen2.5-VL-7B is processing")
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - server may have crashed")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_robust_server() 