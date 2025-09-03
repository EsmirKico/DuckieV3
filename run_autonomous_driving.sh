#!/bin/bash

echo "🚗 Starting DuckieBot Autonomous Driving System..."

# Compile the workspace
echo "Building ROS workspace..."
catkin_make

# Source the setup file
source devel/setup.bash

# Launch the autonomous driving system
echo "Launching autonomous driving with Qwen2.5-VL..."
roslaunch object_follower autonomous_driving.launch robot_name:=ducky 