#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "renderFace.hpp"
#include "mouth_status_estimation.hpp"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 5
#define OPENCV_FACE_RENDER

// global declarations

bool mouthOpen; // store status of mouth being open or closed
cv::Mat im; // matrix to store the image
int counter=0;
std::vector<rectangle> faces; // variable to store face rectangles
cv::Mat imSmall, imDisplay; // matrices to store the resized image to oprate on and display
// Load face detection and pose estimation models.
frontal_face_detector detector = get_frontal_face_detector(); // get the frontal face
shape_predictor predictor;


// 3D Model Points of selected landmarks in an arbitrary frame of reference
std::vector<cv::Point3d> get3dModelPoints()
{
  std::vector<cv::Point3d> modelPoints;

  modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); // Nose tip //The first must be (0,0,0) while using POSIT
  modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f)); // Chin
  modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f)); // Left eye left corner
  modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f)); // Right eye right corner
  modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f)); // Left Mouth corner
  modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f)); // Right Mouth corner
  modelPoints.push_back(cv::Point3d(-236.0f, 340.0f, -125.0f)); // Left temple
  modelPoints.push_back(cv::Point3d(236.0f, 340.0f, -125.0f)); // Right temple

  return modelPoints;

}

// 2D landmark points from all landmarks
std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d)
{

  std::vector<cv::Point2d> imagePoints;
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    // Nose tip
  imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );      // Chin
  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    // Left eye left corner
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );    // Right eye right corner
  imagePoints.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );    // Left Mouth corner
  imagePoints.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );    // Right mouth corner
  imagePoints.push_back( cv::Point2d( d.part(17).x(), d.part(17).y() ) );    // Left temple
  imagePoints.push_back( cv::Point2d( d.part(26).x(), d.part(26).y() ) );    // Right temple

  return imagePoints;

}

// Camera Matrix from focal length and focal center
cv::Mat getCameraMatrix(float focal_length, cv::Point2d center)
{
  cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
  return cameraMatrix;
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
      im = cv_bridge::toCvShare(msg, "bgr8")->image;

      // Create imSmall by resizing image for face detection
      cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

      // Change to dlib's image format. No memory is copied.
      cv_image<bgr_pixel> cimgSmall(imSmall);
      cv_image<bgr_pixel> cimg(im);

      // Process frames at an interval of SKIP_FRAMES.
      // This value should be set depending on your system hardware
      // and camera fps.
      // To reduce computations, this value should be increased
      if ( counter % SKIP_FRAMES == 0 )
      {
        // Detect faces
        faces = detector(cimgSmall);
      }

      // Pose estimation
      std::vector<cv::Point3d> modelPoints = get3dModelPoints();

      // Iterate over faces
      std::vector<full_object_detection> shapes;
      for (unsigned long i = 0; i < faces.size(); ++i)
      {
        // Since we ran face detection on a resized image,
        // we will scale up coordinates of face rectangle
        rectangle r(
              (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
              );

        // Find face landmarks by providing reactangle for each face
        full_object_detection shape = predictor(cimg, r);
        shapes.push_back(shape);

        // Draw landmarks over face
        renderFace(im, shape);

        // get 2D landmarks from Dlib's shape object
        std::vector<cv::Point2d> imagePoints = get2dImagePoints(shape);

        // Camera parameters
        double focal_length = im.cols;
        cv::Mat cameraMatrix = getCameraMatrix(focal_length, cv::Point2d(im.cols/2,im.rows/2));

        // Assume no lens distortion
        cv::Mat distCoeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);

        // calculate rotation and translation vector using solvePnP
        cv::Mat rotationVector;
        cv::Mat translationVector;

        cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector,
        translationVector);

        // Project a 3D point (0, 0, 1000.0) onto the image plane.
        // We use this to draw a line sticking out of the nose
        std::vector<cv::Point3d> noseEndPoint3D;
        std::vector<cv::Point2d> noseEndPoint2D;
        noseEndPoint3D.push_back(cv::Point3d(0,0,1000.0));
        cv::projectPoints(noseEndPoint3D, rotationVector, translationVector, cameraMatrix, distCoeffs, noseEndPoint2D);

        // draw line between nose points in image and 3D nose points
        // projected to image plane
        cv::line(im,imagePoints[0], noseEndPoint2D[0], cv::Scalar(255,0,0), 2);

        // mouth status display
        mouthOpen = checkMouth(shape);
        if (mouthOpen == true){
            cv::putText(im, cv::format("OPEN"), cv::Point(450, 50),
                cv::FONT_HERSHEY_COMPLEX, 1.5,cv::Scalar(0, 0, 255), 3);
            } else {
            cv::putText(im, cv::format("CLOSED"), cv::Point(450, 50),
                cv::FONT_HERSHEY_COMPLEX, 1.5,cv::Scalar(0, 0, 255), 3);
            }

      }

      // Resize image for display
      imDisplay = im;
      cv::resize(im, imDisplay, cv::Size(), 1, 1);
      cv::imshow("Face Pose Detector", imDisplay);
      cv::waitKey(1);

  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  try
  {

   ros::init(argc, argv, "image_listener");
   ros::NodeHandle nh;

   image_transport::ImageTransport it(nh);

   std::string mMarkerTopic = "/camera/color/image_raw";

   deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

   image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, imageCallback);
   ros::spin();

  }
  catch(serialization_error& e)
  {
    cout << "Shape predictor model file not found" << endl;
    cout << "Put shape_predictor_68_face_landmarks in models directory" << endl;
    cout << endl << e.what() << endl;
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}

