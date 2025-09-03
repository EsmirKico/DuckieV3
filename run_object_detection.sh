#!/bin/bash

# Compile the workspace
catkin_make

# Source the setup file
source devel/setup.bash

# Launch the object detection model
roslaunch object_follower duckiebot_following_mode.launch robot_name:=blueduckie