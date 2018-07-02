cmake_minimum_required(VERSION 2.8.12)

PROJECT(week3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    std_msgs
    cv_bridge
    image_transport
)


# ###  Linux and MacOS uncomment the 5 lines below ###
include(~/Desktop/everything/cv4faces/dlib-19.4/dlib/cmake)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib)
ENDMACRO()

add_example(MouthPose)

target_link_libraries(MouthPose
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
