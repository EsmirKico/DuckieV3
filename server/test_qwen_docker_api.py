#!/usr/bin/env python3
"""
Test script for Qwen Docker Autonomous Driving API
"""

import requests
import cv2
import numpy as np
import json
import time
from pathlib import Path

# Configuration
SERVER_URL = "http://192.168.140.144:8000"
QWEN_DOCKER_URL = "http://192.168.140.179:8080"

def test_qwen_docker_connection():
    """Test direct connection to Qwen Docker"""
    print("🔍 Testing direct connection to Qwen Docker...")
    try:
        response = requests.get(f"{QWEN_DOCKER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Qwen Docker is running and accessible")
        else:
            print(f"⚠️ Qwen Docker responded with status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Qwen Docker: {e}")
        print("Make sure Docker container is running on Windows PC!")

def test_server_health():
    """Test server health endpoint"""
    print("\n🔍 Testing server health...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Server is healthy")
            print(f"   Qwen Docker Status: {health_data.get('qwen_docker_status', 'unknown')}")
            print(f"   Qwen URL: {health_data.get('qwen_url', 'unknown')}")
        else:
            print(f"⚠️ Server health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")

def test_autonomous_drive():
    """Test autonomous drive endpoint with a sample image"""
    print("\n🔍 Testing autonomous drive endpoint...")
    
    # Create a test image (simulating camera input)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some lane lines (yellow and white)
    cv2.line(test_image, (100, 480), (200, 200), (0, 255, 255), 5)  # Yellow line (left)
    cv2.line(test_image, (540, 480), (440, 200), (255, 255, 255), 5)  # White line (right)
    
    # Draw a duckie obstacle
    cv2.circle(test_image, (320, 300), 30, (0, 255, 0), -1)  # Green circle for duckie
    
    # Save test image
    cv2.imwrite("test_image.jpg", test_image)
    print("📸 Created test image with lane lines and obstacle")
    
    try:
        # Send request
        with open("test_image.jpg", "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            
            print("📤 Sending request to autonomous drive API...")
            start_time = time.time()
            
            response = requests.post(
                f"{SERVER_URL}/autonomous_drive",
                files=files,
                timeout=60  # Give plenty of time for Qwen
            )
            
            duration = time.time() - start_time
            print(f"⏱️ Response time: {duration:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Autonomous drive API test successful!")
                print(f"   Steering: {result.get('steering_angle', 0):.3f}")
                print(f"   Speed: {result.get('target_speed', 0):.3f}")
                print(f"   State: {result.get('driving_state', 'unknown')}")
                print(f"   Lane Confidence: {result.get('lane_confidence', 0):.2f}")
                print(f"   Obstacle Detected: {result.get('obstacle_detected', False)}")
                print(f"   Emergency Stop: {result.get('emergency_stop', False)}")
                
                # Show debug info
                debug_info = result.get('debug_info', {})
                if debug_info:
                    print("\n🔧 Debug Information:")
                    for key, value in debug_info.items():
                        if key != 'vision_response':  # Skip the large vision response
                            print(f"   {key}: {value}")
                
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    print("🚀 Testing Qwen Docker Autonomous Driving API")
    print("=" * 50)
    
    # Test 1: Direct Qwen Docker connection
    test_qwen_docker_connection()
    
    # Test 2: Server health
    test_server_health()
    
    # Test 3: Autonomous drive endpoint
    test_autonomous_drive()
    
    print("\n" + "=" * 50)
    print("🏁 Testing complete!")
    print("\nInstructions for Windows PC:")
    print("1. Install Docker Desktop")
    print("2. Open PowerShell as Administrator")
    print("3. Run: run_qwen_docker.bat")
    print("4. Wait for model to download and start")
    print("5. Update QWEN_DOCKER_HOST in server to your Windows PC IP")

if __name__ == "__main__":
    main() 