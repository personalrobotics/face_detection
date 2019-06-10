#include "global.hpp"

using namespace dlib;

// 3D Model Points of selected landmarks in an arbitrary frame of reference
std::vector<cv::Point3d> get3dModelPoints() {
  std::vector<cv::Point3d> modelPoints;

  // Stommion origin
  // X direction points forward projecting out of the person's stomion

  modelPoints.push_back(cv::Point3d(0., 0., 0.));          // Stommion
  modelPoints.push_back(cv::Point3d(-30.0, -65.5, 70.0));  // Right Eye
  modelPoints.push_back(cv::Point3d(-30.0, 65.5, 70.));    // Left Eye
  modelPoints.push_back(cv::Point3d(11.0, 0., 27.0));      // Nose
  modelPoints.push_back(cv::Point3d(-10.0, 0.0, 75.0));    // Sellion
  modelPoints.push_back(cv::Point3d(-10.0, 0., -58.0));    // Menton
  modelPoints.push_back(cv::Point3d(-10.0, -3.4, 75.0));   // Right Eye Lid
  modelPoints.push_back(cv::Point3d(-10.0, 3.4, 75.0));    // Left Eye Lid
  modelPoints.push_back(cv::Point3d(-5.0, -2.5, 0.0));     // Right Lip corner
  modelPoints.push_back(cv::Point3d(-5.0, 2.5, 0.0));      // Left Lip corner
  modelPoints.push_back(cv::Point3d(-115.0, -77.5, 69.0)); // Right side
  modelPoints.push_back(cv::Point3d(-115.0, 77.5, 69.0));  // Left side

  return modelPoints;
}

// 2D landmark points from all landmarks
std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d) {
  std::vector<cv::Point2d> imagePoints;

  imagePoints.push_back(
      cv::Point2d((d.part(62).x() + d.part(66).x()) * 0.5,
                  (d.part(62).y() + d.part(66).y()) * 0.5)); // Stommion
  // imagePoints.push_back( cv::Point2d( d.part(66).x(),d.part(66).y()  ) );
  // // Stommion
  imagePoints.push_back(
      cv::Point2d(d.part(36).x(), d.part(36).y())); // Right Eye
  imagePoints.push_back(
      cv::Point2d(d.part(45).x(), d.part(45).y())); // Left Eye
  imagePoints.push_back(cv::Point2d(d.part(30).x(), d.part(30).y())); // Nose
  imagePoints.push_back(cv::Point2d(d.part(27).x(), d.part(27).y())); // Sellion
  imagePoints.push_back(cv::Point2d(d.part(8).x(), d.part(8).y()));   // Menton
  imagePoints.push_back(
      cv::Point2d(d.part(38).x(), d.part(38).y())); // Right Eye Lid
  imagePoints.push_back(
      cv::Point2d(d.part(43).x(), d.part(43).y())); // Left Eye Lid
  imagePoints.push_back(
      cv::Point2d(d.part(48).x(), d.part(48).y())); // Right Lip Corner
  imagePoints.push_back(
      cv::Point2d(d.part(54).x(), d.part(54).y())); // Left Lip Corner
  imagePoints.push_back(
      cv::Point2d(d.part(0).x(), d.part(0).y())); // Right Side
  imagePoints.push_back(
      cv::Point2d(d.part(16).x(), d.part(16).y())); // Left Side

  return imagePoints;
}
