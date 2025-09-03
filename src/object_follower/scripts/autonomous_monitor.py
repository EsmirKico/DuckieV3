#!/usr/bin/env python3

"""
Autonomous Driving Monitor for DuckieBot
Monitors lane following, obstacle detection, and stop line detection
"""

import rospy
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32, String
import json

class AutonomousMonitor:
    def __init__(self):
        rospy.init_node('autonomous_monitor', anonymous=True)
        
        # Subscribers for autonomous driving
        rospy.Subscriber('/autonomous/lane_info', String, self.lane_info_callback)
        rospy.Subscriber('/autonomous/obstacle_detected', Bool, self.obstacle_callback)
        rospy.Subscriber('/autonomous/stop_line_detected', Bool, self.stop_line_callback)
        rospy.Subscriber('/autonomous/driving_state', String, self.driving_state_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        # State tracking
        self.current_state = "unknown"
        self.lane_info = None
        self.obstacle_detected = False
        self.stop_line_detected = False
        self.last_cmd_vel = None
        
        # Performance metrics
        self.lane_following_time = 0.0
        self.obstacle_encounters = 0
        self.stop_line_encounters = 0
        self.start_time = rospy.Time.now()
        
        rospy.loginfo("Autonomous Driving Monitor started")
        
    def lane_info_callback(self, msg):
        try:
            self.lane_info = json.loads(msg.data)
        except:
            self.lane_info = None
            
    def obstacle_callback(self, msg):
        if msg.data and not self.obstacle_detected:
            self.obstacle_encounters += 1
            rospy.loginfo("🚨 OBSTACLE DETECTED! Starting avoidance maneuver...")
        self.obstacle_detected = msg.data
        
    def stop_line_callback(self, msg):
        if msg.data and not self.stop_line_detected:
            self.stop_line_encounters += 1
            rospy.loginfo("🛑 STOP LINE DETECTED! Stopping robot...")
        self.stop_line_detected = msg.data
        
    def driving_state_callback(self, msg):
        if msg.data != self.current_state:
            rospy.loginfo(f"🚗 Driving state changed: {self.current_state} → {msg.data}")
            self.current_state = msg.data
            
        # Track lane following time
        if msg.data == "lane_following":
            self.lane_following_time += 0.1  # Approximate
            
    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg
        
        # Log detailed status periodically
        if self.lane_info:
            lane_offset = self.lane_info.get('lane_center_offset', 0.0)
            confidence = self.lane_info.get('confidence', 0.0)
            
            rospy.loginfo_throttle(3, 
                f"🤖 Autonomous Status: State={self.current_state}, "
                f"Lane Offset={lane_offset:.3f}, Confidence={confidence:.2f}, "
                f"Speed={msg.linear.x:.2f}, Steering={msg.angular.z:.2f}")
                
        # Safety warnings
        if abs(msg.angular.z) > 1.5:
            rospy.logwarn_throttle(1, "⚠️  Sharp steering detected - check lane detection!")
            
        if msg.linear.x > 0.5:
            rospy.logwarn_throttle(1, "⚠️  High speed detected - safety check!")
            
    def print_summary(self):
        """Print driving session summary"""
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        
        rospy.loginfo("\n" + "="*50)
        rospy.loginfo("🚗 AUTONOMOUS DRIVING SESSION SUMMARY")
        rospy.loginfo("="*50)
        rospy.loginfo(f"Total time: {elapsed_time:.1f} seconds")
        rospy.loginfo(f"Lane following time: {self.lane_following_time:.1f} seconds")
        rospy.loginfo(f"Obstacles encountered: {self.obstacle_encounters}")
        rospy.loginfo(f"Stop lines encountered: {self.stop_line_encounters}")
        rospy.loginfo(f"Current state: {self.current_state}")
        
        if self.lane_info:
            confidence = self.lane_info.get('confidence', 0.0)
            rospy.loginfo(f"Final lane confidence: {confidence:.2f}")
            
        rospy.loginfo("="*50)

if __name__ == '__main__':
    try:
        monitor = AutonomousMonitor()
        rospy.on_shutdown(monitor.print_summary)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 