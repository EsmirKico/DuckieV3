import requests
import cv2
import numpy as np
import json

API_URL = "http://localhost:8000/autonomous_drive"
LM_STUDIO_URL = "http://192.168.140.1:1234"

def test_lm_studio_direct():
    """Test LM Studio connection directly"""
    print("🔗 Testing direct LM Studio connection...")
    try:
        response = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ LM Studio connected! Models available: {len(models.get('data', []))}")
            for model in models.get('data', []):
                print(f"   📦 {model.get('id', 'Unknown')}")
            return True
        else:
            print(f"❌ LM Studio error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        return False

def test_autonomous_api():
    """Test the autonomous driving API with webcam"""
    
    print("🚗 Testing LM Studio Autonomous Driving API...")
    
    # Try webcam first
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No webcam found. Testing with synthetic image...")
        # Create a simple test image with lane-like features
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw yellow line (left)
        cv2.line(test_image, (200, 480), (250, 200), (0, 255, 255), 5)
        # Draw white line (right)
        cv2.line(test_image, (400, 480), (350, 200), (255, 255, 255), 5)
        # Draw road surface
        cv2.rectangle(test_image, (200, 200), (400, 480), (50, 50, 50), -1)
        
        test_single_image(test_image)
        return
    
    print("📹 Using webcam for testing. Press 'q' to quit.")
    print("🎯 Point camera at Duckietown track, lanes, or obstacles to test detection.")
    print("💡 Tips:")
    print("   - Good lighting helps vision model performance")
    print("   - Clear lane markings work best")
    print("   - Try drawing yellow/white lines on paper for testing")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame_count += 1
        
        # Test every 10th frame to avoid overwhelming the system
        if frame_count % 10 != 0:
            cv2.imshow("LM Studio Autonomous API Test (Processing every 10th frame)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Encode frame as JPEG for sending to API
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

        try:
            print(f"\n🔄 Processing frame {frame_count}...")
            response = requests.post(API_URL, files=files, timeout=30)
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                print("="*60)
                print("🤖 LM STUDIO AUTONOMOUS DRIVING RESPONSE:")
                print("="*60)
                
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
                else:
                    print("✅ No obstacles detected")
                
                # Stop line info
                stop_info = result.get('stop_line_info', {})
                if stop_info.get('detected', False):
                    print(f"🛑 Stop Line Detected!")
                    print(f"📏 Distance: {stop_info.get('distance', 0.0):.2f}m")
                    print(f"🎯 Confidence: {stop_info.get('confidence', 0.0):.2f}")
                else:
                    print("🟢 No stop lines detected")
                
                # Driving commands
                steering = result.get('steering_angle', 0.0)
                speed = result.get('target_speed', 0.0)
                state = result.get('driving_state', 'unknown')
                
                print(f"\n🚗 DRIVING COMMANDS:")
                print(f"   Steering: {steering:.3f} ({'LEFT' if steering < -0.1 else 'RIGHT' if steering > 0.1 else 'STRAIGHT'})")
                print(f"   Speed: {speed:.3f} ({'FORWARD' if speed > 0.1 else 'STOPPED'})")
                print(f"   State: {state.upper()}")
                
                # Draw visualization on frame
                cv2.putText(frame, f"State: {state}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Steering: {steering:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Speed: {speed:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Lane Conf: {lane_info.get('confidence', 0.0):.2f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if lane_info.get('confidence', 0.0) > 0.3:
                    cv2.putText(frame, "LANE DETECTED", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, "LM Studio + Qwen", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Show debug info
                debug_info = result.get('debug_info', '')
                if debug_info:
                    print(f"🔍 Debug: {debug_info[:80]}...")
                
            else:
                print(f"❌ API call failed: {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print("⏰ API call timed out (this can happen with vision models)")
        except Exception as e:
            print(f"❌ Exception calling API: {e}")

        cv2.imshow("LM Studio Autonomous API Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed!")

def test_single_image(image):
    """Test with a single image"""
    print("🖼️  Testing with single image...")
    
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'file': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response received!")
            print(f"Lane Confidence: {result.get('lane_info', {}).get('confidence', 0.0):.2f}")
            print(f"Driving State: {result.get('driving_state', 'unknown')}")
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 LM Studio + Qwen Autonomous Driving API Test")
    print("="*60)
    
    # Step 1: Test LM Studio connection
    if not test_lm_studio_direct():
        print("\n❌ LM Studio is not accessible. Please:")
        print("   1. Start LM Studio")
        print("   2. Load a vision model (like Qwen2-VL)")
        print("   3. Ensure it's running on http://192.168.140.1:1234")
        exit(1)
    
    print("\n" + "="*60)
    
    # Step 2: Test server connection
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ Autonomous server is running!")
        else:
            print("❌ Server not responding correctly")
            exit(1)
    except:
        print("❌ Cannot connect to server. Please start:")
        print("   python server/autonomous_server_lmstudio.py")
        exit(1)
    
    print("\n" + "="*60)
    
    # Step 3: Test LM Studio integration
    try:
        response = requests.get("http://localhost:8000/test_lmstudio")
        result = response.json()
        if result.get('lm_studio_connected', False):
            print("✅ LM Studio integration working!")
        else:
            print("⚠️  LM Studio integration has issues")
    except:
        print("⚠️  Could not test LM Studio integration")
    
    print("\n" + "="*60)
    print("🎬 Starting main test...")
    test_autonomous_api() 