#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Bool, Float32

class SystemMonitor:
    def __init__(self):
        rospy.init_node('system_monitor', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/object_follower/target_position', Point, self.target_callback)
        rospy.Subscriber('/object_follower/target_found', Bool, self.target_found_callback)
        rospy.Subscriber('/object_follower/target_distance', Float32, self.distance_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        
        self.target_found = False
        self.target_position = None
        self.distance = 0.0
        
        rospy.loginfo("System Monitor started - watching all topics")
        
    def target_callback(self, msg):
        self.target_position = msg
        
    def target_found_callback(self, msg):
        if msg.data != self.target_found:
            self.target_found = msg.data
            if self.target_found:
                rospy.loginfo("TARGET DETECTED!")
            else:
                rospy.loginfo("Target lost")
                
    def distance_callback(self, msg):
        self.distance = msg.data
        
    def cmd_vel_callback(self, msg):
        if self.target_found and self.target_position:
            rospy.loginfo(f"Following: pos=({self.target_position.x:.2f},{self.target_position.y:.2f}) "
                         f"dist={self.distance:.2f}m → vel=({msg.linear.x:.2f},{msg.angular.z:.2f})")

if __name__ == '__main__':
    try:
        monitor = SystemMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass