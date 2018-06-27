cmake_minimum_required(VERSION 2.8.12)

SET(CMAKE_BUILD_TYPE "Release")

PROJECT(week3)

find_package( OpenCV REQUIRED )

include(~/Desktop/everything/cv4faces/dlib-19.4/dlib/cmake) // absolute pah for dlib 19.4 library folder

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib)
ENDMACRO()

add_example(mouth_pose)
