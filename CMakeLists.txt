cmake_minimum_required(VERSION 2.8.3)
project(face_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -mavx -O3")

find_package( OpenCV REQUIRED )

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

find_package( dlib REQUIRED )

message(STATUS "dlib_INCLUDE_DIRS = ${dlib_INCLUDE_DIRS}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3_INCLUDE_DIRS = ${Eigen3_INCLUDE_DIRS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  std_msgs
  visualization_msgs
  sensor_msgs
  cv_bridge
  image_transport
  image_geometry
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  LIBRARIES face_detection
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs cv_bridge image_transport tf
  #DEPENDS eigen
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
##include(../../../../dlib/dlib-19.4/dlib/cmake)
##include(/usr/lib/cmake/eigen3)
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${dlib_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  /usr/include/eigen3
)


# MACRO(add_example name)
#  ADD_EXECUTABLE(${name} ${name}.cpp)
#  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib)
# ENDMACRO()

# add_example(MouthPose)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/face_detection.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
file(GLOB_RECURSE face_srcs "${CMAKE_SOURCE_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${face_srcs})
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${dlib_LIBRARIES}
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#==============================================================================
# Automatic code formatting setup
#
define_property(GLOBAL PROPERTY CODE_FORMATTING_SOURCES
  BRIEF_DOCS "Source files that are automatically formatted by clang-format."
  FULL_DOCS "Source files that are automatically formatted by clang-format."
)

function(format_add_sources)
  foreach(source ${ARGN})
    if(IS_ABSOLUTE "${source}")
      set(source_abs "${source}")
    else()
      get_filename_component(source_abs
        "${CMAKE_CURRENT_LIST_DIR}/${source}" ABSOLUTE)
    endif()
    if(EXISTS "${source_abs}")
      set_property(GLOBAL APPEND PROPERTY CODE_FORMATTING_SOURCES "${source_abs}")
    else()
      message(FATAL_ERROR
        "Source file '${source}' does not exist at absolute path"
        " '${source_abs}'. This should never happen. Did you recently delete"
        " this file or modify 'CMAKE_CURRENT_LIST_DIR'")
    endif()
  endforeach()
endfunction()

file(GLOB_RECURSE face_headers "${CMAKE_SOURCE_DIR}/include/*.hpp")
format_add_sources(${face_headers})
format_add_sources(${face_srcs})

#===============================================================================
# Automatic code formatting using clang-format.
#
find_program(
  CLANG_FORMAT_EXECUTABLE
  NAMES clang-format-3.8
)
# We only support one specific ClangFormat version because different versions
# result in different code formatting with the same configuration. ClangFormat
# 3.8 is chosen since it's the latest version supported by Ubuntu Trusty.

if (CLANG_FORMAT_EXECUTABLE)
  get_property(formatting_files GLOBAL PROPERTY CODE_FORMATTING_SOURCES)
  list(LENGTH formatting_files formatting_files_length)

  if (formatting_files)
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E echo "Formatting code style of"
            "${formatting_files_length} files... "
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -style=file -i ${formatting_files}
        COMMAND ${CMAKE_COMMAND} -E echo "Done."
        DEPENDS ${CLANG_FORMAT_EXECUTABLE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_custom_target(check-format
        COMMAND ${CMAKE_COMMAND} -E echo "Checking code style of"
            "${formatting_files_length} files... "
        COMMAND ${CMAKE_SOURCE_DIR}/tools/check_format.sh
            ${CLANG_FORMAT_EXECUTABLE} ${formatting_files}
        COMMAND ${CMAKE_COMMAND} -E echo "Done."
        DEPENDS ${CLANG_FORMAT_EXECUTABLE}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
  else()
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E echo "No file to format code style."
    )
    add_custom_target(check-format
        COMMAND ${CMAKE_COMMAND} -E echo "No file to check code style."
    )
  endif()
else()
  message(STATUS "Looking for clang-format-3.8 - NOT found, please install clang-format-3.8 to enable automatic code formatting")
endif()

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
 install(TARGETS ${PROJECT_NAME}
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## Mark cpp header files for installation
 install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.hpp"
#   PATTERN ".svn" EXCLUDE
 )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_face_detection.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
