cmake_minimum_required(VERSION 3.0.2)
project(object_follower)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    std_msgs 
    sensor_msgs 
    geometry_msgs 
    cv_bridge 
    image_transport 
    tf2 
    tf2_ros 
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/enhanced_object_detector.py
  scripts/enhanced_motor_controller.py
  scripts/system_monitor.py
  scripts/performance_monitor.py
  scripts/debug_system.py
  scripts/detector_tester.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)