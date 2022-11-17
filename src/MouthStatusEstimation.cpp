#include "global.hpp"

std::vector<cv::Point2d> get2dmouthPoints(dlib::full_object_detection &d) {
  std::vector<cv::Point2d> mouthPoints;
  mouthPoints.push_back(
      cv::Point2d(d.part(50).x(), d.part(50).y())); // 0  Top of upper lip, left
  mouthPoints.push_back(
      cv::Point2d(d.part(51).x(), d.part(51).y())); // 1  Top of upper lip, mid
  mouthPoints.push_back(cv::Point2d(
      d.part(52).x(), d.part(52).y())); // 2  Top of upper lip, right
  mouthPoints.push_back(cv::Point2d(
      d.part(56).x(), d.part(56).y())); // 3  Bottom of bottom lip, right
  mouthPoints.push_back(cv::Point2d(
      d.part(57).x(), d.part(57).y())); // 4  Bottom of bottom lip, mid
  mouthPoints.push_back(cv::Point2d(
      d.part(58).x(), d.part(58).y())); // 5  Bottom of bottom lip, left
  mouthPoints.push_back(cv::Point2d(
      d.part(61).x(), d.part(61).y())); // 6  Bottom of upper lip, left
  mouthPoints.push_back(cv::Point2d(
      d.part(62).x(), d.part(62).y())); // 7  Bottom of upper lip, mid
  mouthPoints.push_back(cv::Point2d(
      d.part(63).x(), d.part(63).y())); // 8  Bottom of upper lip, right
  mouthPoints.push_back(cv::Point2d(
      d.part(65).x(), d.part(65).y())); // 9  Top of bottom lip, right
  mouthPoints.push_back(
      cv::Point2d(d.part(66).x(), d.part(66).y())); // 10 Top of bottom lip, mid
  mouthPoints.push_back(cv::Point2d(
      d.part(67).x(), d.part(67).y())); // 11 Top of bottom lip, left
  return mouthPoints;
}

bool checkMouth(dlib::full_object_detection &shape) {
  bool mouthOpen;

  auto mouthPoints = get2dmouthPoints(shape);
  float lipDist = (float)sqrt(pow((mouthPoints[10].x - mouthPoints[7].x), 2) +
                              pow((mouthPoints[10].y - mouthPoints[7].y), 2));
  float lipThickness =
      (float)sqrt(pow((mouthPoints[1].x - mouthPoints[7].x), 2) +
                  pow((mouthPoints[1].y - mouthPoints[7].y), 2)) /
          2 +
      sqrt(pow((mouthPoints[4].x - mouthPoints[10].x), 2) +
           pow((mouthPoints[4].y - mouthPoints[10].y), 2)) /
          2;

  if (lipDist >= 1.5 * lipThickness) {
    mouthOpen = true;
  } else {
    mouthOpen = false;
  }
  return (mouthOpen);
  // Print bool : mouth open, closed
}
