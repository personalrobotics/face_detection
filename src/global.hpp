#pragma once

#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include "std_msgs/String.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <image_transport/image_transport.h>
#include <mutex>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <sstream>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <string>
#include <vector>
#include <stdlib.h>


// ModelPoints.cpp
// 3D Model Points of selected landmarks in an arbitrary frame of reference
std::vector<cv::Point3d> get3dModelPoints();
// 2D landmark points from all landmarks
std::vector<cv::Point2d> get2dImagePoints(dlib::full_object_detection &d);

// MouthPose.cpp
static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
static dlib::rectangle openCVRectToDlib(cv::Rect r) ;
void imageCallback(const sensor_msgs::ImageConstPtr &msg);

void publishMarker(float tx, float ty, float tz);
void DepthCallBack(const sensor_msgs::ImageConstPtr depth_img_ros);
void cameraInfo(const sensor_msgs::CameraInfoConstPtr &msg);
int main(int argc, char **argv);

// MouthStatusEstimation.cpp
std::vector<cv::Point2d> get2dmouthPoints(dlib::full_object_detection &d);
bool checkMouth(dlib::full_object_detection &shape);

// RenderFace.cpp
// Draw an open or closed polygon between
// start and end indices of full_object_detection
void drawPolyline(cv::Mat &img, const dlib::full_object_detection &landmarks,
                  const int start, const int end, bool isClosed = false);
// Draw face for the 68-point model.
void renderFace(cv::Mat &img, const dlib::full_object_detection &landmarks);
// Draw points on an image.
// Works for any number of points.
void renderFace(cv::Mat &img,                            // Image to draw the points on
                const std::vector<cv::Point2d> &points, // Vector of points
                cv::Scalar color,                       // color points
                int radius = 3);                        // Radius of points.

#endif  // _GLOBAL_H_
