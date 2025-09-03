#!/usr/bin/env python3

"""
Enhanced Object Detector Node for DuckieBot - Phase 3
Uses YOLOv8 API for tennis ball detection
"""

import rospy
import cv2
import numpy as np
import requests
import base64
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge
import threading

API_IP = '172.20.10.3'
API_PORT = 8000
API_ENDPOINT = '/detect'

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain  
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Target value (0 for center)
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def update(self, current_value, dt=None):
        """Calculate PID output"""
        if dt is None:
            dt = 0.1  # Default time step
            
        # Calculate error (distance from center)
        error = self.setpoint - current_value
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term (accumulated error over time)
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # Derivative term (rate of error change)
        derivative = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        
        # Calculate output
        output = proportional + integral_term + derivative
        
        # Save for next iteration
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0.0
        self.integral = 0.0


class EnhancedObjectDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('enhanced_object_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()

        # PID Controllers for centering the ball
        self.session = requests.Session()
        self.session.headers.update({'Connection': 'keep-alive'})

        self.x_pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=0.0)
        self.y_pid = PIDController(kp=0.8, ki=0.05, kd=0.03, setpoint=0.0)
        
        # Publishers
        self.target_pub = rospy.Publisher('/object_follower/target_position', Point, queue_size=1)
        self.target_found_pub = rospy.Publisher('/object_follower/target_found', Bool, queue_size=1)
        self.distance_pub = rospy.Publisher('/object_follower/target_distance', Float32, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/object_follower/debug_image', Image, queue_size=1)
        self.detection_info_pub = rospy.Publisher('/object_follower/detection_info', String, queue_size=1)
        self.control_pub = rospy.Publisher('/object_follower/control_cmd', Point, queue_size=1)
        
        # Subscribers - Handle both local and DuckieBot camera topics
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback, 
                                 queue_size=1, buff_size=2**24)
        
        # DuckieBot-specific topic (with robot namespace)
        robot_name = rospy.get_param('~robot_name', 'ducky')
        compressed_topic = f"/{robot_name}/camera_node/image/compressed"
        self.compressed_image_sub = rospy.Subscriber(compressed_topic, CompressedImage, 
                                           self.compressed_image_callback, 
                                           queue_size=1, buff_size=2**24)
        
        # Fallback for generic topic
        self.compressed_fallback_sub = rospy.Subscriber('/camera_node/image/compressed', CompressedImage, self.compressed_image_callback, queue_size=1, buff_size=2**24)
        
        # API Configuration
        self.api_url = f"http://{API_IP}:{API_PORT}{API_ENDPOINT}"  # API endpoint
        self.api_timeout = rospy.get_param('~api_timeout', 0.5)  # seconds
        
        # Enhanced tracking parameters
        self.last_known_position = None
        self.tracking_confidence = 0.0
        self.consecutive_detections = 0
        self.max_tracking_loss = 10  # frames
        self.tracking_loss_count = 0
        
        # Kalman filter for object tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman_initialized = False
        
        # Performance metrics
        self.detection_count = 0
        self.api_failure_count = 0
        self.start_time = rospy.Time.now()
        self.last_control_time = rospy.Time.now()

        self.last_process_time = 0
        self.min_process_interval = 0.1  # Process at most 10 FPS
        
        rospy.loginfo(f"Enhanced Object Detector initialized - Using API: {self.api_url}")
    
    def compressed_image_callback(self, msg):
        """Handle compressed images from DuckieBot camera"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                rospy.loginfo_throttle(5, f"Enhanced detector processing compressed image: {cv_image.shape}")
                self.detect_with_api(cv_image)
            else:
                rospy.logwarn("Failed to decode compressed image")
                
        except Exception as e:
            rospy.logerr(f"Error processing compressed image: {str(e)}")
    
    def image_callback(self, msg):
        """Handle regular images (for local testing)"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.loginfo_throttle(5, f"Enhanced detector processing image: {cv_image.shape}")
            self.detect_with_api(cv_image)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def call_detection_api(self, image):
        """Call the YOLOv8 detection API"""
        try:
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Send to API
            files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
            response = self.session.post(self.api_url, files=files, timeout=self.api_timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                rospy.logwarn(f"API returned status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            rospy.logwarn(f"API call timed out after {self.api_timeout} seconds")
            self.api_failure_count += 1
            return None
        except Exception as e:
            rospy.logwarn(f"API call failed: {str(e)}")
            self.api_failure_count += 1
            return None
    
    def detect_with_api(self, image):
        """Use API for detection with asynchronous threaded processing"""

        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_process_time < self.min_process_interval:
            return

        if hasattr(self, '_processing') and self._processing:
            rospy.loginfo_throttle(1, "Skipping frame - API still processing")
            return

        self._processing = True
        self.last_process_time = current_time

        # Process asynchronously in separate thread (non-blocking)
        thread = threading.Thread(target=self._async_detect, args=(image.copy(),))
        thread.daemon = True  # Dies when main thread dies
        thread.start()

    def _async_detect(self, image):
        """Asynchronous detection processing in separate thread"""
        try:
            height, width = image.shape[:2]
            if width > 640:
                scale = 640.0 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            result = self.call_detection_api(image)

            if result and result.get('target_found', False):
                best_detection = result.get('best_detection')
                if best_detection:
                    center_x = best_detection['center_x']
                    center_y = best_detection['center_y']
                    confidence = best_detection['confidence']

                    # Kalman filter prediction and update
                    if not self.kalman_initialized:
                        self.kalman.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
                        self.kalman.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)
                        self.kalman_initialized = True

                    prediction = self.kalman.predict()
                    measurement = np.array([[center_x], [center_y]], dtype=np.float32)
                    self.kalman.correct(measurement)

                    # Calculate PID control for centering
                    current_time = rospy.Time.now()
                    dt = (current_time - self.last_control_time).to_sec()
                    self.last_control_time = current_time

                    # Get normalized position (-1 to 1, where 0 is center)
                    x_position = result['target_position_x']  # Current X position
                    y_position = result['target_position_y']  # Current Y position

                    # Calculate PID outputs (how much to turn/move)
                    x_control = self.x_pid.update(x_position, dt)  # Left/Right control
                    y_control = self.y_pid.update(y_position, dt)  # Up/Down control (if needed)

                    # Create control command
                    control_cmd = Point()
                    control_cmd.x = x_control      # Turning speed (left/right)
                    control_cmd.y = 0.0           # Forward speed (can add distance-based control)
                    control_cmd.z = y_control     # Tilt control (if your robot supports it)
                    
                    # Publish control command for robot movement
                    self.control_pub.publish(control_cmd)

                    # Also publish original detection for other nodes
                    target_point = Point()
                    target_point.x = x_position
                    target_point.y = y_position
                    target_point.z = result['estimated_distance']

                    # Publish target information
                    self.target_pub.publish(target_point)
                    self.distance_pub.publish(Float32(result['estimated_distance']))
                    self.target_found_pub.publish(Bool(True))

                    # Update tracking
                    self.last_known_position = target_point
                    self.tracking_confidence = confidence
                    self.consecutive_detections += 1
                    self.tracking_loss_count = 0
                    self.detection_count += 1

                    # Publish detection info
                    info_msg = f"Method: YOLOv8_API, Confidence: {confidence:.2f}, Distance: {result['estimated_distance']:.2f}m"
                    self.detection_info_pub.publish(String(info_msg))

            else:
                # Handle tracking loss or API failure
                self.tracking_loss_count += 1
                if self.tracking_loss_count > self.max_tracking_loss:
                    self.consecutive_detections = 0
                    self.tracking_confidence = 0.0
                    self.target_found_pub.publish(Bool(False))
                else:
                    # Use prediction if recently lost and Kalman is initialized
                    if self.kalman_initialized:
                        prediction = self.kalman.predict()
                        debug_image = image.copy()
                        cv2.circle(debug_image, (int(prediction[0]), int(prediction[1])), 
                                  10, (255, 255, 0), 2)
                        cv2.putText(debug_image, "PREDICTED", (int(prediction[0]), int(prediction[1])-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                        try:
                            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                            self.debug_image_pub.publish(debug_msg)
                        except Exception as e:
                            rospy.logwarn(f"Could not publish prediction debug image: {str(e)}")
                    else:
                        self.target_found_pub.publish(Bool(False))

        finally:
            self._processing = False


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = EnhancedObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
