#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from geometry_msgs.msg import Point

class DetectorTester:
    def __init__(self):
        rospy.init_node('detector_tester', anonymous=True)
        
        self.bridge = CvBridge()
        
        # Subscribers
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        rospy.Subscriber('/object_follower/target_found', Bool, self.target_found_callback)
        rospy.Subscriber('/object_follower/target_position', Point, self.target_position_callback)
        
        self.image_count = 0
        self.target_detections = 0
        
        rospy.loginfo("Detector Tester started")
        
        # Print stats every 5 seconds
        rospy.Timer(rospy.Duration(5.0), self.print_stats)
        
    def image_callback(self, msg):
        self.image_count += 1
        
        # Show images for debugging
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Create a simple red detection test
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Red color range
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on image
            result_image = cv_image.copy()
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 500:
                    cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 3)
                    
                    # Calculate center
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(result_image, (cx, cy), 10, (255, 0, 0), -1)
                        
                        rospy.loginfo_throttle(2, f"Manual detection: Red object at ({cx}, {cy}), area: {cv2.contourArea(largest_contour):.0f}")
            
            cv2.imshow('Detector Test', result_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logwarn(f"Error processing image: {e}")
    
    def target_found_callback(self, msg):
        if msg.data:
            self.target_detections += 1
            rospy.loginfo("Enhanced detector found target!")
    
    def target_position_callback(self, msg):
        rospy.loginfo(f"Target position: ({msg.x:.2f}, {msg.y:.2f})")
    
    def print_stats(self, event):
        rospy.loginfo("=" * 40)
        rospy.loginfo(f"DETECTOR TEST STATS")
        rospy.loginfo(f"Images received: {self.image_count}")
        rospy.loginfo(f"Target detections: {self.target_detections}")
        rospy.loginfo(f"Detection rate: {self.target_detections/max(1,self.image_count/10)*100:.1f}%")
        rospy.loginfo("=" * 40)

if __name__ == '__main__':
    try:
        tester = DetectorTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()