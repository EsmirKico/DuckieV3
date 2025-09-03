cmake_minimum_required(VERSION 3.0.2)
project(duckietown_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
  sensor_msgs
)

# Add all message files
add_message_files(
  DIRECTORY msg
  FILES
    Twist2DStamped.msg
    WheelsCmdStamped.msg
    BoolStamped.msg
    Pose2DStamped.msg
    FSMState.msg
    LanePose.msg
    CarControl.msg
    # Add other essential messages as needed
)

# Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    message_runtime 
    std_msgs 
    geometry_msgs 
    sensor_msgs
)