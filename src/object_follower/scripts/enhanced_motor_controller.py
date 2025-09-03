#!/usr/bin/env python3

"""
Enhanced Motor Controller Node for DuckieBot - Phase 2
Optimized for DuckieBot deployment with official duckietown_msgs
"""

import rospy
import numpy as np
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, Header
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

class EnhancedMotorController:
    def __init__(self):
        rospy.init_node('enhanced_motor_controller', anonymous=True)
        
        # Publishers - Full DuckieBot integration with official messages
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.wheels_pub = rospy.Publisher('/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.duckiebot_vel_pub = rospy.Publisher('/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        
        # Performance monitoring publishers
        self.performance_pub = rospy.Publisher('/object_follower/performance', Float32, queue_size=1)
        
        # Subscribers
        self.target_sub = rospy.Subscriber('/object_follower/target_position', Point, self.target_callback)
        self.target_found_sub = rospy.Subscriber('/object_follower/target_found', Bool, self.target_found_callback)
        self.distance_sub = rospy.Subscriber('/object_follower/target_distance', Float32, self.distance_callback)
        self.obstacle_sub = rospy.Subscriber('/object_follower/obstacle_detected', Bool, self.obstacle_callback)
        
        # Enhanced control parameters
        self.max_speed = rospy.get_param('~max_speed', 1.5)
        self.target_distance = rospy.get_param('~target_distance', 1.0)
        self.distance_tolerance = rospy.get_param('~distance_tolerance', 0.15)
        
        # Robot parameters
        self.wheel_distance = rospy.get_param('~wheel_distance', 0.1)  # meters between wheels
        
        self.kp_lateral = rospy.get_param('~kp_lateral', 1.5)  # Reduced from 2.5
        self.ki_lateral = rospy.get_param('~ki_lateral', 0.08) # Reduced from 0.15
        self.kd_lateral = rospy.get_param('~kd_lateral', 0.3)  # Reduced from 0.8
        
        self.kp_distance = rospy.get_param('~kp_distance', 1.5)
        self.ki_distance = rospy.get_param('~ki_distance', 0.08)
        self.kd_distance = rospy.get_param('~kd_distance', 0.3)
        
        # State variables
        self.target_position = None
        self.target_found = False
        self.current_distance = 0.0
        self.obstacle_detected = False
        
        # Enhanced PID state
        self.lateral_error_integral = 0.0
        self.lateral_error_previous = 0.0
        self.distance_error_integral = 0.0
        self.distance_error_previous = 0.0
        
        # Performance tracking
        self.control_history = []
        self.start_time = rospy.Time.now()
        
        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20 Hz

        # Enhanced PID parameters for centering mode
        self.centering_mode = rospy.get_param('~centering_mode', True)  # Enable centering

        # More aggressive lateral control for centering
        self.kp_lateral_centering = rospy.get_param('~kp_lateral_centering', 2.2)  # Reduced from 3.5
        self.ki_lateral_centering = rospy.get_param('~ki_lateral_centering', 0.08)  # Reduced from 0.2
        self.kd_lateral_centering = rospy.get_param('~kd_lateral_centering', 0.6)   # Reduced from 1.2


        # Distance-based speed scaling
        self.min_speed_factor = rospy.get_param('~min_speed_factor', 0.3)  # Minimum speed when turning
        self.centering_threshold = rospy.get_param('~centering_threshold', 0.15)  # Consider "centered" within ±0.15

        # Anti-fidgeting parameters
        self.lateral_deadband = rospy.get_param('~lateral_deadband', 0.03)  # Ignore small errors
        self.min_angular_vel = rospy.get_param('~min_angular_vel', 0.05)     # Minimum turn command
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.7)   # Output smoothing

        #Previous command for smoothing
        self.prev_angular_vel = 0.0
        self.prev_linear_vel = 0.0

        self.stop_distance = rospy.get_param('~stop_distance', 0.3)  # Stop when < 30cm
        self.slow_distance = rospy.get_param('~slow_distance', 0.6)   # Slow when < 60cm
        
        rospy.loginfo("Enhanced Motor Controller - DuckieBot deployment ready")
    
    def target_callback(self, msg):
        self.target_position = msg
    
    def target_found_callback(self, msg):
        self.target_found = msg.data
        if not self.target_found:
            self.reset_pid_state()
    
    def distance_callback(self, msg):
        self.current_distance = msg.data
    
    def obstacle_callback(self, msg):
        if msg.data and not self.obstacle_detected:
            rospy.logwarn("🚧 Obstacle detected! Emergency stop!")
        self.obstacle_detected = msg.data
    
    def reset_pid_state(self):
        """Reset PID integral terms when target is lost"""
        self.lateral_error_integral = 0.0
        self.distance_error_integral = 0.0
    
    def control_loop(self, event):
        """Enhanced control loop with performance monitoring"""
        if self.obstacle_detected:
            self.emergency_stop()
            return
        
        if not self.target_found or self.target_position is None:
            self.stop_robot()
            return
        
        # Adaptive PID tuning every few cycles
        # if len(self.control_history) % 10 == 0:
        #     self.adaptive_pid_tuning()

        if self.current_distance > 0 and self.current_distance < self.stop_distance:
            rospy.loginfo_throttle(2, f"Ball too close ({self.current_distance:.2f}m)! Stopping.")
            self.publish_velocity_commands(0.0, 0.0)
            return
        
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_enhanced_control_commands()
        
        # Apply velocity limits (higher angular limits for faster turning)
        linear_vel = np.clip(linear_vel, -self.max_speed, self.max_speed)
        angular_vel = np.clip(angular_vel, -1.5, 1.5)  # Increased from 2.0 to 3.0
        
        # Publish commands to all topics
        self.publish_velocity_commands(linear_vel, angular_vel)
        
        # Track performance
        self.track_performance(linear_vel, angular_vel)

    def adaptive_pid_tuning(self):
        """Dynamically adjust PID gains based on performance"""
        if len(self.control_history) < 20:
            return

        recent_errors = [abs(d['lateral_error']) for d in self.control_history[-10:]]
        avg_error = np.mean(recent_errors)

        # If consistently off-center, increase gains
        if avg_error > self.centering_threshold * 1.5:
            self.kp_lateral_centering = min(5.0, self.kp_lateral_centering * 1.1)
            rospy.loginfo_throttle(5, f"Increased centering gain to {self.kp_lateral_centering:.2f}")

        # If oscillating, reduce gains
        error_changes = [abs(recent_errors[i] - recent_errors[i-1]) for i in range(1, len(recent_errors))]
        if np.mean(error_changes) > 0.3:
            self.kp_lateral_centering = max(2.0, self.kp_lateral_centering * 0.95)
            self.kd_lateral_centering = min(2.0, self.kd_lateral_centering * 1.1)

    
    def calculate_enhanced_control_commands(self):
        """Enhanced PID control with anti-fidgeting measures"""
        dt = 0.05  # 20 Hz control loop

        # Use centering-specific PID gains if enabled
        if self.centering_mode:
            kp_lat = self.kp_lateral_centering
            ki_lat = self.ki_lateral_centering 
            kd_lat = self.kd_lateral_centering
        else:
            kp_lat = self.kp_lateral
            ki_lat = self.ki_lateral
            kd_lat = self.kd_lateral

        # DISTANCE-BASED GAIN REDUCTION for close targets
        distance_factor = 1.0
        if self.current_distance < 0.5:  # Ball is very close (< 50cm)
            distance_factor = max(0.3, self.current_distance / 0.5)  # Reduce gains dramatically
            kp_lat *= distance_factor
            ki_lat *= distance_factor
            rospy.loginfo_throttle(2, f"Close ball detected! Reducing gains by factor: {distance_factor:.2f}")

        # Lateral control (steering)
        lateral_error = self.target_position.x  # -1 to 1, 0 is center

        # ENHANCED DEADBAND when ball is close
        close_deadband = self.lateral_deadband
        if self.current_distance < 0.8:  # Increase deadband when close
            close_deadband = self.lateral_deadband * (2.0 - self.current_distance)  

        # Apply deadband - ignore small errors
        if abs(lateral_error) < close_deadband:
            lateral_error = 0.0
            self.lateral_error_integral *= 0.8  # Decay integral when in deadband

        # PID calculations (CORRECT ORDER)
        self.lateral_error_integral += lateral_error * dt
        lateral_error_derivative = (lateral_error - self.lateral_error_previous) / dt

        # Anti-windup for integral term
        self.lateral_error_integral = np.clip(self.lateral_error_integral, -0.3, 0.3)

        angular_vel = -(kp_lat * lateral_error + 
                       ki_lat * self.lateral_error_integral + 
                       kd_lat * lateral_error_derivative)

        # Apply minimum command threshold
        min_angular_threshold = self.min_angular_vel
        if self.current_distance < 0.8:
            min_angular_threshold *= (2.0 - self.current_distance)  # Higher threshold when close

        if abs(angular_vel) < min_angular_threshold:
            angular_vel = 0.0

        self.lateral_error_previous = lateral_error  # Update AFTER using it

        # Distance control (forward/backward)
        distance_error = self.current_distance - self.target_distance
        self.distance_error_integral += distance_error * dt
        distance_error_derivative = (distance_error - self.distance_error_previous) / dt

        # Anti-windup
        self.distance_error_integral = np.clip(self.distance_error_integral, -1.0, 1.0)

        linear_vel = (self.kp_distance * distance_error + 
                     self.ki_distance * self.distance_error_integral + 
                     self.kd_distance * distance_error_derivative)

        self.distance_error_previous = distance_error

        # GENTLE speed scaling (less aggressive)
        if abs(self.target_position.x) > self.centering_threshold:
            speed_factor = max(0.9, 1.0 - 0.5 * abs(self.target_position.x))
        else:
            speed_factor = 1.0 - 0.2 * abs(self.target_position.x)

        linear_vel *= speed_factor

        # SMOOTH OUTPUT - blend with previous commands
        angular_vel = self.smoothing_factor * self.prev_angular_vel + (1 - self.smoothing_factor) * angular_vel
        linear_vel = self.smoothing_factor * self.prev_linear_vel + (1 - self.smoothing_factor) * linear_vel

        # Store for next iteration
        self.prev_angular_vel = angular_vel
        self.prev_linear_vel = linear_vel

        return linear_vel, angular_vel


    def track_performance(self, linear_vel, angular_vel):
        """Track control performance metrics"""
        control_data = {
            'timestamp': rospy.Time.now().to_sec(),
            'linear_vel': linear_vel,
            'angular_vel': angular_vel,
            'lateral_error': self.target_position.x if self.target_position else 0,
            'distance_error': abs(self.current_distance - self.target_distance)
        }
        
        self.control_history.append(control_data)
        
        # Keep only last 100 measurements
        if len(self.control_history) > 100:
            self.control_history.pop(0)
        
        # Calculate and publish performance score
        if len(self.control_history) >= 10:
            performance_score = self.calculate_performance_score()
            self.performance_pub.publish(Float32(performance_score))
    
    def calculate_performance_score(self):
        """Calculate performance score with centering emphasis"""
        if not self.control_history:
            return 0.0

        recent_data = self.control_history[-20:]

        # Calculate metrics
        lateral_errors = [abs(d['lateral_error']) for d in recent_data]
        distance_errors = [d['distance_error'] for d in recent_data]
        velocity_changes = [abs(recent_data[i]['linear_vel'] - recent_data[i-1]['linear_vel']) 
                           for i in range(1, len(recent_data))]

        # ENHANCED: Centering-focused scoring
        centering_score = max(0, 100 - np.mean(lateral_errors) * 300)  # Higher penalty for off-center
        distance_score = max(0, 100 - np.mean(distance_errors) * 100)
        smoothness_score = max(0, 100 - np.mean(velocity_changes) * 400)

        # Check if consistently centered
        centered_count = sum(1 for err in lateral_errors[-10:] if abs(err) < self.centering_threshold)
        consistency_bonus = (centered_count / 10) * 20  # Up to 20 bonus points

        # Weighted average emphasizing centering
        overall_score = (0.6 * centering_score + 0.25 * distance_score + 0.15 * smoothness_score + consistency_bonus)

        return min(100.0, max(0.0, overall_score))

    
    def publish_velocity_commands(self, linear_vel, angular_vel):
        """Publish velocity commands to all supported topics"""
        current_time = rospy.Time.now()
        
        # 1. Standard ROS Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)
        
        # 2. DuckieBot Twist2DStamped message
        duckiebot_msg = Twist2DStamped()
        duckiebot_msg.header = Header()
        duckiebot_msg.header.stamp = current_time
        duckiebot_msg.header.frame_id = "base_link"
        duckiebot_msg.v = linear_vel
        duckiebot_msg.omega = angular_vel
        self.duckiebot_vel_pub.publish(duckiebot_msg)
        
        # 3. DuckieBot WheelsCmdStamped message
        # Convert linear and angular velocity to wheel velocities
        vel_left = linear_vel - angular_vel * self.wheel_distance / 2.0
        vel_right = linear_vel + angular_vel * self.wheel_distance / 2.0
        
        wheels_msg = WheelsCmdStamped()
        wheels_msg.header = Header()
        wheels_msg.header.stamp = current_time
        wheels_msg.header.frame_id = "base_link"
        wheels_msg.vel_left = vel_left
        wheels_msg.vel_right = vel_right
        self.wheels_pub.publish(wheels_msg)
    
    def stop_robot(self):
        """Stop the robot"""
        self.publish_velocity_commands(0.0, 0.0)
    
    def emergency_stop(self):
        """Emergency stop with immediate brake"""
        self.publish_velocity_commands(0.0, 0.0)
        self.reset_pid_state()

if __name__ == '__main__':
    try:
        controller = EnhancedMotorController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass