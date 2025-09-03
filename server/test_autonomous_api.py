import requests
import cv2
import numpy as np
import json

API_URL = "http://localhost:8000/autonomous_drive"

def test_autonomous_api():
    """Test the autonomous driving API with webcam or image file"""
    
    print("🚗 Testing Autonomous Driving API...")
    
    # Try webcam first
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No webcam found. Please test with real DuckieBot camera.")
        return
    
    print("📹 Using webcam for testing. Press 'q' to quit.")
    print("🎯 Point camera at Duckietown track, lanes, or obstacles to test detection.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Encode frame as JPEG for sending to API
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

        try:
            response = requests.post(API_URL, files=files, timeout=10)
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                print("\n" + "="*50)
                print("🤖 AUTONOMOUS DRIVING API RESPONSE:")
                print("="*50)
                
                # Lane info
                lane_info = result.get('lane_info', {})
                print(f"🛣️  Lane Offset: {lane_info.get('lane_center_offset', 0.0):.3f}")
                print(f"📐 Lane Angle: {lane_info.get('lane_angle', 0.0):.1f}°")
                print(f"🎯 Lane Confidence: {lane_info.get('confidence', 0.0):.2f}")
                
                # Obstacle info
                obstacle_info = result.get('obstacle_info', {})
                if obstacle_info.get('detected', False):
                    print(f"🚨 Obstacle: {obstacle_info.get('type', 'unknown')}")
                    print(f"📍 Position: {obstacle_info.get('position_x', 0.0):.2f}")
                    print(f"📏 Distance: {obstacle_info.get('distance', 0.0):.2f}m")
                    print(f"↗️  Avoidance: {obstacle_info.get('avoidance_direction', 'none')}")
                
                # Stop line info
                stop_info = result.get('stop_line_info', {})
                if stop_info.get('detected', False):
                    print(f"🛑 Stop Line Detected!")
                    print(f"📏 Distance: {stop_info.get('distance', 0.0):.2f}m")
                    print(f"🎯 Confidence: {stop_info.get('confidence', 0.0):.2f}")
                
                # Driving commands
                print(f"🚗 Steering: {result.get('steering_angle', 0.0):.3f}")
                print(f"⚡ Speed: {result.get('target_speed', 0.0):.3f}")
                print(f"🔄 State: {result.get('driving_state', 'unknown')}")
                
                # Draw basic visualization
                cv2.putText(frame, f"State: {result.get('driving_state', 'unknown')}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Steering: {result.get('steering_angle', 0.0):.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Speed: {result.get('target_speed', 0.0):.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                if lane_info.get('confidence', 0.0) > 0.3:
                    cv2.putText(frame, "LANE DETECTED", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                print(f"❌ API call failed: {response.status_code} {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ API call timed out - this is normal for first request (model loading)")
        except Exception as e:
            print(f"❌ Exception calling API: {e}")

        cv2.imshow("Autonomous Driving API Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    print("🚀 Starting Qwen2.5-VL Autonomous Driving API Test")
    print("📋 Make sure the server is running: python server/autonomous_server.py")
    print()
    
    # Test basic connection first
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ Server is running!")
            test_autonomous_api()
        else:
            print("❌ Server not responding correctly")
    except:
        print("❌ Cannot connect to server. Please start: python server/autonomous_server.py") 