#!/usr/bin/env python3

"""
Performance Monitor for Object Follower System - Phase 2
Tracks and analyzes system performance metrics
"""

import rospy
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, String
import numpy as np
import json
from collections import deque

class PerformanceMonitor:
    def __init__(self):
        rospy.init_node('performance_monitor', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/object_follower/target_position', Point, self.target_callback)
        rospy.Subscriber('/object_follower/target_found', Bool, self.target_found_callback)
        rospy.Subscriber('/object_follower/target_distance', Float32, self.distance_callback)
        rospy.Subscriber('/object_follower/detection_info', String, self.detection_info_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # Performance metrics storage
        self.target_distance = rospy.get_param('~target_distance', 1.0)
        self.window_size = 100  # Number of samples to keep
        
        # Metric queues
        self.position_errors = deque(maxlen=self.window_size)
        self.distance_errors = deque(maxlen=self.window_size)
        self.velocities = deque(maxlen=self.window_size)
        self.angular_velocities = deque(maxlen=self.window_size)
        self.detection_states = deque(maxlen=self.window_size)
        self.response_times = deque(maxlen=50)
        
        # Tracking variables
        self.last_target_found = False
        self.target_loss_count = 0
        self.target_acquisition_count = 0
        self.total_detections = 0
        self.start_time = rospy.Time.now()
        self.last_detection_time = None
        
        # Control performance
        self.control_smoothness = deque(maxlen=self.window_size)
        self.previous_linear_vel = 0.0
        self.previous_angular_vel = 0.0
        
        # Detection method tracking
        self.detection_methods = {}
        
        # Statistics timer
        self.stats_timer = rospy.Timer(rospy.Duration(10.0), self.print_detailed_stats)
        self.quick_stats_timer = rospy.Timer(rospy.Duration(2.0), self.print_quick_stats)
        
        rospy.loginfo("Performance Monitor started - Tracking system metrics")
        
    def target_callback(self, msg):
        # Calculate position error (how far from center)
        position_error = abs(msg.x)  # Distance from center (-1 to 1 range)
        self.position_errors.append(position_error)
        
        # Record response time if target was just acquired
        if not self.last_target_found and len(self.detection_states) > 0 and self.detection_states[-1]:
            current_time = rospy.Time.now()
            if self.last_detection_time:
                response_time = (current_time - self.last_detection_time).to_sec()
                self.response_times.append(response_time)
    
    def target_found_callback(self, msg):
        current_found = msg.data
        self.detection_states.append(current_found)
        
        # Track target acquisition and loss
        if current_found and not self.last_target_found:
            self.target_acquisition_count += 1
            self.last_detection_time = rospy.Time.now()
        elif not current_found and self.last_target_found:
            self.target_loss_count += 1
        
        if current_found:
            self.total_detections += 1
            
        self.last_target_found = current_found
    
    def distance_callback(self, msg):
        # Calculate distance error
        distance_error = abs(msg.data - self.target_distance)
        self.distance_errors.append(distance_error)
    
    def detection_info_callback(self, msg):
        # Parse detection method information
        try:
            info = msg.data
            if "Method:" in info:
                method = info.split("Method: ")[1].split(",")[0]
                if method in self.detection_methods:
                    self.detection_methods[method] += 1
                else:
                    self.detection_methods[method] = 1
        except:
            pass
    
    def cmd_vel_callback(self, msg):
        # Track velocity smoothness and control performance
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(linear_vel**2 + angular_vel**2)
        self.velocities.append(velocity_magnitude)
        self.angular_velocities.append(abs(angular_vel))
        
        # Calculate control smoothness (rate of change)
        linear_smoothness = abs(linear_vel - self.previous_linear_vel)
        angular_smoothness = abs(angular_vel - self.previous_angular_vel)
        total_smoothness = linear_smoothness + angular_smoothness
        self.control_smoothness.append(total_smoothness)
        
        self.previous_linear_vel = linear_vel
        self.previous_angular_vel = angular_vel
    
    def calculate_stability_score(self):
        """Calculate system stability score (0-100)"""
        if not self.position_errors or not self.distance_errors:
            return 0
        
        # Position stability (lower error = higher score)
        avg_pos_error = np.mean(self.position_errors)
        pos_score = max(0, 100 * (1 - avg_pos_error))
        
        # Distance stability
        avg_dist_error = np.mean(self.distance_errors)
        dist_score = max(0, 100 * (1 - avg_dist_error / self.target_distance))
        
        # Control smoothness (lower variation = higher score)
        if self.control_smoothness:
            smoothness_score = max(0, 100 * (1 - np.mean(self.control_smoothness) / 2.0))
        else:
            smoothness_score = 50
        
        # Detection consistency
        if self.detection_states:
            detection_rate = sum(self.detection_states) / len(self.detection_states)
            detection_score = detection_rate * 100
        else:
            detection_score = 0
        
        # Overall score (weighted average)
        overall_score = (pos_score * 0.3 + dist_score * 0.3 + 
                        smoothness_score * 0.2 + detection_score * 0.2)
        
        return overall_score
    
    def print_quick_stats(self, event):
        """Print quick status update"""
        if not self.position_errors:
            return
        
        stability_score = self.calculate_stability_score()
        current_detection = self.detection_states[-1] if self.detection_states else False
        
        status_icon = "Target Hit" if current_detection else "Target Not Hit"
        quality_icon = "Super" if stability_score > 80 else "Good" if stability_score > 60 else "Bad"
        
        rospy.loginfo(f"{status_icon} Target: {current_detection} | {quality_icon} Stability: {stability_score:.1f}%")
    
    def print_detailed_stats(self, event):
        """Print detailed performance statistics"""
        if not self.position_errors:
            rospy.loginfo("📊 Waiting for data...")
            return
            
        # Calculate all metrics
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        
        # Position metrics
        avg_pos_error = np.mean(self.position_errors)
        std_pos_error = np.std(self.position_errors)
        max_pos_error = np.max(self.position_errors)
        
        # Distance metrics
        avg_dist_error = np.mean(self.distance_errors) if self.distance_errors else 0
        std_dist_error = np.std(self.distance_errors) if self.distance_errors else 0
        
        # Velocity metrics
        avg_velocity = np.mean(self.velocities) if self.velocities else 0
        avg_angular_vel = np.mean(self.angular_velocities) if self.angular_velocities else 0
        
        # Control smoothness
        avg_smoothness = np.mean(self.control_smoothness) if self.control_smoothness else 0
        
        # Detection metrics
        detection_rate = self.total_detections / max(elapsed_time, 1.0)
        if self.detection_states:
            uptime_percentage = (sum(self.detection_states) / len(self.detection_states)) * 100
        else:
            uptime_percentage = 0
        
        # Response time
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        
        # Stability score
        stability_score = self.calculate_stability_score()
        
        # Print comprehensive report
        rospy.loginfo("=" * 60)
        rospy.loginfo("PERFORMANCE REPORT (Last 10 seconds)")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"TRACKING PERFORMANCE:")
        rospy.loginfo(f"   Position Error: {avg_pos_error:.3f} ± {std_pos_error:.3f} (max: {max_pos_error:.3f})")
        rospy.loginfo(f"   Distance Error: {avg_dist_error:.3f} ± {std_dist_error:.3f} meters")
        rospy.loginfo(f"   Target Uptime: {uptime_percentage:.1f}%")
        rospy.loginfo(f"   Detection Rate: {detection_rate:.1f} Hz")
        
        rospy.loginfo(f"CONTROL PERFORMANCE:")
        rospy.loginfo(f"   Average Speed: {avg_velocity:.3f} m/s")
        rospy.loginfo(f"   Average Turn Rate: {avg_angular_vel:.3f} rad/s")
        rospy.loginfo(f"   Control Smoothness: {avg_smoothness:.3f} (lower is smoother)")
        
        rospy.loginfo(f"RESPONSE METRICS:")
        rospy.loginfo(f"   Target Acquisitions: {self.target_acquisition_count}")
        rospy.loginfo(f"   Target Losses: {self.target_loss_count}")
        rospy.loginfo(f"   Avg Response Time: {avg_response_time:.3f}s")
        
        rospy.loginfo(f"OVERALL ASSESSMENT:")
        stability_grade = "EXCELLENT" if stability_score > 85 else "GOOD" if stability_score > 70 else "FAIR" if stability_score > 50 else "NEEDS IMPROVEMENT"
        rospy.loginfo(f"   Stability Score: {stability_score:.1f}% ({stability_grade})")
        
        if self.detection_methods:
            rospy.loginfo(f"DETECTION METHODS: {dict(self.detection_methods)}")
        
        rospy.loginfo("=" * 60)
    
    def get_performance_summary(self):
        """Return performance summary as dictionary"""
        return {
            'stability_score': self.calculate_stability_score(),
            'avg_position_error': np.mean(self.position_errors) if self.position_errors else 0,
            'avg_distance_error': np.mean(self.distance_errors) if self.distance_errors else 0,
            'detection_uptime': (sum(self.detection_states) / len(self.detection_states) * 100) if self.detection_states else 0,
            'target_acquisitions': self.target_acquisition_count,
            'target_losses': self.target_loss_count,
            'avg_velocity': np.mean(self.velocities) if self.velocities else 0
        }

if __name__ == '__main__':
    try:
        monitor = PerformanceMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass