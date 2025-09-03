#!/usr/bin/env python3

"""
Autonomous Driving Detector Node for DuckieBot
Uses Qwen2.5-VL API for lane following, stop line detection, and obstacle avoidance
"""

import rospy
import cv2
import numpy as np
import requests
import base64
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge
import threading
import json
from duckietown_msgs.msg import Twist2DStamped, WheelsCmdStamped

# API Configuration
API_IP = '192.168.140.144'
API_PORT = 8002  # Using Qwen Docker server
API_ENDPOINT = '/autonomous_drive'

class LaneFollowingController:
    def __init__(self):
        # Lane following PID parameters (different from object following)
        self.kp_steering = 1.2  # Steering response to lane offset
        self.ki_steering = 0.05
        self.kd_steering = 0.15
        
        self.steering_integral = 0.0
        self.prev_steering_error = 0.0
        
    def update(self, lane_offset, dt=0.1):
        """Calculate steering command for lane following"""
        # Lane offset is from -1 to 1, where 0 is perfect center
        error = lane_offset  # We want to minimize the offset
        
        # PID calculation
        self.steering_integral += error * dt
        self.steering_integral = np.clip(self.steering_integral, -0.5, 0.5)  # Anti-windup
        
        derivative = (error - self.prev_steering_error) / dt if dt > 0 else 0
        
        steering_output = (self.kp_steering * error + 
                         self.ki_steering * self.steering_integral + 
                         self.kd_steering * derivative)
        
        self.prev_steering_error = error
        
        return np.clip(steering_output, -1.0, 1.0)
    
    def reset(self):
        self.steering_integral = 0.0
        self.prev_steering_error = 0.0

class AutonomousDriving:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('autonomous_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize lane following controller
        self.lane_controller = LaneFollowingController()
        
        # Session for API calls
        self.session = requests.Session()
        self.session.headers.update({'Connection': 'keep-alive'})
        
        # Get robot name for proper topic namespacing
        self.robot_name = rospy.get_param('~robot_name', 'ducky')
        rospy.loginfo(f"Using robot name: {self.robot_name}")
        
        # Publishers for autonomous driving (with robot namespace)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.duckiebot_vel_pub = rospy.Publisher(f'/{self.robot_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        self.wheels_pub = rospy.Publisher(f'/{self.robot_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        
        # Autonomous driving status publishers
        self.lane_info_pub = rospy.Publisher('/autonomous/lane_info', String, queue_size=1)
        self.obstacle_pub = rospy.Publisher('/autonomous/obstacle_detected', Bool, queue_size=1)
        self.stop_line_pub = rospy.Publisher('/autonomous/stop_line_detected', Bool, queue_size=1)
        self.driving_state_pub = rospy.Publisher('/autonomous/driving_state', String, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/autonomous/debug_image', Image, queue_size=1)
        
        # Publishers compatible with existing motor controller
        self.target_pub = rospy.Publisher('/object_follower/target_position', Point, queue_size=1)
        self.target_found_pub = rospy.Publisher('/object_follower/target_found', Bool, queue_size=1)
        self.distance_pub = rospy.Publisher('/object_follower/target_distance', Float32, queue_size=1)
        
        # Subscribers - Handle both local and DuckieBot camera topics
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback, 
                                 queue_size=1, buff_size=2**24)
        
        # DuckieBot-specific topic (with robot namespace)
        compressed_topic = f"/{self.robot_name}/camera_node/image/compressed"
        self.compressed_image_sub = rospy.Subscriber(compressed_topic, CompressedImage, 
                                           self.compressed_image_callback, 
                                           queue_size=1, buff_size=2**24)
        
        # Fallback for generic topic
        self.compressed_fallback_sub = rospy.Subscriber('/camera_node/image/compressed', CompressedImage, self.compressed_image_callback, queue_size=1, buff_size=2**24)
        
        # API Configuration
        self.api_url = f"http://{API_IP}:{API_PORT}{API_ENDPOINT}"
        self.api_timeout = rospy.get_param('~api_timeout', 10.0)  # Longer timeout for Qwen2.5-VL-7B model
        
        # Autonomous driving state
        self.current_state = "lane_following"
        self.last_lane_info = None
        self.obstacle_detected = False
        self.stop_line_detected = False
        
        # Performance tracking
        self.last_process_time = 0
        self.min_process_interval = 0.5  # Process at ~2 FPS (optimized for Qwen2.5-VL-7B with 15s timeout)
        self._processing = False
        
        # Safety parameters
        self.max_speed = rospy.get_param('~max_speed', 0.8)
        self.emergency_stop_enabled = False
        
        rospy.loginfo(f"Autonomous Driving initialized - Using API: {self.api_url}")
    
    def compressed_image_callback(self, msg):
        """Handle compressed images from DuckieBot camera"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                rospy.loginfo_throttle(5, f"Autonomous driving processing compressed image: {cv_image.shape}")
                self.process_autonomous_driving(cv_image)
            else:
                rospy.logwarn("Failed to decode compressed image")
                
        except Exception as e:
            rospy.logerr(f"Error processing compressed image: {str(e)}")
    
    def image_callback(self, msg):
        """Handle regular images (for local testing)"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.loginfo_throttle(5, f"Autonomous driving processing image: {cv_image.shape}")
            self.process_autonomous_driving(cv_image)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def call_autonomous_api(self, image):
        """Call the Qwen2.5-VL autonomous driving API"""
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
            return None
        except Exception as e:
            rospy.logwarn(f"API call failed: {str(e)}")
            return None
    
    def process_autonomous_driving(self, image):
        """Process autonomous driving with rate limiting"""
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_process_time < self.min_process_interval:
            return

        if hasattr(self, '_processing') and self._processing:
            rospy.loginfo_throttle(1, "Skipping frame - API still processing")
            return

        self._processing = True
        self.last_process_time = current_time

        # Process asynchronously in separate thread
        thread = threading.Thread(target=self._async_autonomous_drive, args=(image.copy(),))
        thread.daemon = True
        thread.start()

    def _async_autonomous_drive(self, image):
        """Asynchronous autonomous driving processing"""
        try:
            # Resize image if too large to improve processing speed
            height, width = image.shape[:2]
            if width > 640:
                scale = 640.0 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            # Call Qwen2.5-VL API
            result = self.call_autonomous_api(image)

            if result:
                self.process_driving_response(result, image)
            else:
                # API failed - use safe driving defaults instead of emergency stop
                rospy.logwarn("API failed, using safe driving defaults")
                self.publish_safe_driving_defaults()

        except Exception as e:
            rospy.logerr(f"Error in autonomous driving: {str(e)}")
            # Use safe defaults instead of emergency stop
            self.publish_safe_driving_defaults()
        finally:
            self._processing = False

    def process_driving_response(self, result, image):
        """Process the autonomous driving API response"""
        try:
            # Extract information from API response
            lane_info = result.get('lane_info', {})
            obstacle_info = result.get('obstacle_info', {})
            stop_line_info = result.get('stop_line_info', {})
            
            steering_angle = result.get('steering_angle', 0.0)
            target_speed = result.get('target_speed', 0.0)
            driving_state = result.get('driving_state', 'stopped')
            
            # Update internal state
            self.current_state = driving_state
            self.obstacle_detected = obstacle_info.get('detected', False)
            self.stop_line_detected = stop_line_info.get('detected', False)
            
            # Publish autonomous driving information
            lane_msg = json.dumps(lane_info)
            self.lane_info_pub.publish(String(lane_msg))
            self.obstacle_pub.publish(Bool(self.obstacle_detected))
            self.stop_line_pub.publish(Bool(self.stop_line_detected))
            self.driving_state_pub.publish(String(driving_state))
            
            # Convert to motor commands and publish
            self.publish_motor_commands(steering_angle, target_speed, lane_info)
            
            # Publish compatibility messages for existing motor controller
            self.publish_compatibility_messages(lane_info, obstacle_info)
            
            # Log driving status
            lane_offset = lane_info.get('lane_center_offset', 0.0)
            confidence = lane_info.get('confidence', 0.0)
            
            rospy.loginfo_throttle(2, f"Autonomous: State={driving_state}, "
                                 f"Lane Offset={lane_offset:.2f}, "
                                 f"Steering={steering_angle:.2f}, "
                                 f"Speed={target_speed:.2f}, "
                                 f"Confidence={confidence:.2f}")
            
        except Exception as e:
            rospy.logerr(f"Error processing driving response: {str(e)}")
            # Use safe defaults instead of emergency stop
            self.publish_safe_driving_defaults()

    def publish_motor_commands(self, steering_angle, target_speed, lane_info):
        """Publish motor commands for autonomous driving"""
        from std_msgs.msg import Header
        
        # Convert API commands to ROS velocity commands
        linear_vel = target_speed * self.max_speed  # Forward speed
        angular_vel = steering_angle * 2.0  # Angular velocity for steering
        
        # Safety limits
        linear_vel = np.clip(linear_vel, 0.0, self.max_speed)
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        # Emergency stop override (disabled - robot keeps moving with safe defaults)
        
        current_time = rospy.Time.now()
        
        # 1. Standard ROS Twist message (for debugging)
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)
        
        # 2. DuckieBot Twist2DStamped message (primary control)
        duckiebot_msg = Twist2DStamped()
        duckiebot_msg.header = Header()
        duckiebot_msg.header.stamp = current_time
        duckiebot_msg.header.frame_id = "base_link"
        duckiebot_msg.v = linear_vel
        duckiebot_msg.omega = angular_vel
        self.duckiebot_vel_pub.publish(duckiebot_msg)
        
        # 3. DuckieBot WheelsCmdStamped message (backup control)
        wheel_distance = 0.1  # meters between wheels
        vel_left = linear_vel - angular_vel * wheel_distance / 2.0
        vel_right = linear_vel + angular_vel * wheel_distance / 2.0
        
        wheels_msg = WheelsCmdStamped()
        wheels_msg.header = Header()
        wheels_msg.header.stamp = current_time
        wheels_msg.header.frame_id = "base_link"
        wheels_msg.vel_left = vel_left
        wheels_msg.vel_right = vel_right
        self.wheels_pub.publish(wheels_msg)
        
        # Log commands for debugging
        rospy.loginfo_throttle(2, 
            f"🚗 Motor Commands: linear={linear_vel:.3f}, angular={angular_vel:.3f} "
            f"→ wheels L={vel_left:.3f}, R={vel_right:.3f} on robot '{self.robot_name}'")

    def publish_safe_driving_defaults(self):
        """Publish safe driving defaults when vision fails"""
        from std_msgs.msg import Header
        current_time = rospy.Time.now()
        
        # Safe default: moderate forward movement, no steering
        linear_vel = 0.4  # Moderate forward speed for ground movement
        angular_vel = 0.0  # No steering
        
        # Publish to all motor control topics
        # 1. Standard Twist
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)
        
        # 2. DuckieBot Twist2DStamped
        duckiebot_msg = Twist2DStamped()
        duckiebot_msg.header = Header()
        duckiebot_msg.header.stamp = current_time
        duckiebot_msg.header.frame_id = "base_link"
        duckiebot_msg.v = linear_vel
        duckiebot_msg.omega = angular_vel
        self.duckiebot_vel_pub.publish(duckiebot_msg)
        
        # 3. DuckieBot WheelsCmdStamped
        wheel_distance = 0.1
        vel_left = linear_vel - angular_vel * wheel_distance / 2.0
        vel_right = linear_vel + angular_vel * wheel_distance / 2.0
        
        wheels_msg = WheelsCmdStamped()
        wheels_msg.header = Header()
        wheels_msg.header.stamp = current_time
        wheels_msg.header.frame_id = "base_link"
        wheels_msg.vel_left = vel_left
        wheels_msg.vel_right = vel_right
        self.wheels_pub.publish(wheels_msg)
        
        # Publish status messages
        self.driving_state_pub.publish(String("safe_default"))
        self.target_found_pub.publish(Bool(False))
        
        rospy.loginfo_throttle(2, f"🔒 Safe driving: forward={linear_vel:.2f}, steering={angular_vel:.2f} (increased for ground movement)")

    def publish_compatibility_messages(self, lane_info, obstacle_info):
        """Publish messages compatible with existing motor controller"""
        # Convert lane following to "target position" for compatibility
        lane_offset = lane_info.get('lane_center_offset', 0.0)
        confidence = lane_info.get('confidence', 0.0)
        
        if confidence > 0.3:  # Good lane detection
            # Convert lane offset to target position
            target_point = Point()
            target_point.x = lane_offset  # Lane offset (-1 to 1)
            target_point.y = 0.0  # Not used for lane following
            target_point.z = 1.0  # Fixed distance for lane following
            
            self.target_pub.publish(target_point)
            self.target_found_pub.publish(Bool(True))
            self.distance_pub.publish(Float32(1.0))
        else:
            # Poor lane detection
            self.target_found_pub.publish(Bool(False))

    def emergency_stop(self):
        """Gentle stop - reduce to safe speed instead of full stop"""
        
        from std_msgs.msg import Header
        current_time = rospy.Time.now()
        
        # Gentle stop: slow forward movement instead of full stop
        linear_vel = 0.2  # Slow but sufficient forward speed for ground movement
        angular_vel = 0.0  # No steering
        
        # Publish gentle stop commands to all topics
        # 1. Standard Twist
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)
        
        # 2. DuckieBot Twist2DStamped
        duckiebot_msg = Twist2DStamped()
        duckiebot_msg.header = Header()
        duckiebot_msg.header.stamp = current_time
        duckiebot_msg.header.frame_id = "base_link"
        duckiebot_msg.v = linear_vel
        duckiebot_msg.omega = angular_vel
        self.duckiebot_vel_pub.publish(duckiebot_msg)
        
        # 3. DuckieBot WheelsCmdStamped
        wheel_distance = 0.1
        vel_left = linear_vel - angular_vel * wheel_distance / 2.0
        vel_right = linear_vel + angular_vel * wheel_distance / 2.0
        
        wheels_msg = WheelsCmdStamped()
        wheels_msg.header = Header()
        wheels_msg.header.stamp = current_time
        wheels_msg.header.frame_id = "base_link"
        wheels_msg.vel_left = vel_left
        wheels_msg.vel_right = vel_right
        self.wheels_pub.publish(wheels_msg)
        
        # Publish compatibility messages
        self.target_found_pub.publish(Bool(False))
        self.driving_state_pub.publish(String("gentle_stop"))
        
        rospy.logwarn(f"⚠️ GENTLE STOP - slow movement on robot '{self.robot_name}'")

    def resume_driving(self):
        """Resume autonomous driving after emergency stop"""
        self.emergency_stop_enabled = False
        self.lane_controller.reset()
        rospy.loginfo("Autonomous driving resumed")

    def run(self):
        """Main running loop"""
        rospy.loginfo("Autonomous driving started - ready for lane following")
        rospy.spin()

if __name__ == '__main__':
    try:
        autonomous = AutonomousDriving()
        autonomous.run()
    except rospy.ROSInterruptException:
        pass 