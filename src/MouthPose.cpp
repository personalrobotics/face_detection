#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "face_detection/renderFace.hpp"
#include "face_detection/mouth_status_estimation.hpp"
#include "std_msgs/String.h"
#include <sstream>

using namespace dlib;
using namespace std;
using namespace sensor_msgs;

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

ros::Publisher marker_array_pub;

cv::Mat_<double> distCoeffs(5,1);
cv::Mat_<double> cameraMatrix(3,3);


// 3D Model Points of selected landmarks in an arbitrary frame of reference
std::vector<cv::Point3d> get3dModelPoints()
{
  std::vector<cv::Point3d> modelPoints;

  modelPoints.push_back(cv::Point3d(0.0, 0.0, 0.0));    // Sellion
  modelPoints.push_back(cv::Point3d(-20., -65.5,-5.));  // Right Eye
  modelPoints.push_back(cv::Point3d(-20., 65.5,-5.));   // Left Eye
  modelPoints.push_back(cv::Point3d(-100., -77.5,-6.)); // Right Ear
  modelPoints.push_back(cv::Point3d(-100., 77.5,-6.));  // Left Ear
  modelPoints.push_back(cv::Point3d(21.0, 0., -48.0));  // Nose
  modelPoints.push_back(cv::Point3d(10.0, 0., -75.0));  // Stommion
  modelPoints.push_back(cv::Point3d(0., 0.,-133.0));    // Menton

  return modelPoints;

}

// 2D landmark points from all landmarks
std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d)
{

  std::vector<cv::Point2d> imagePoints;

  imagePoints.push_back( cv::Point2d( d.part(27).x(), d.part(27).y() ) );                             // Sellion
  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );                             // Right Eye
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );                             // Left Eye
  imagePoints.push_back( cv::Point2d( d.part(0).x(), d.part(0).y() ) );                               // Right Ear
  imagePoints.push_back( cv::Point2d( d.part(16).x(), d.part(16).y() ) );                             // Left Ear
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );                             // Nose
  imagePoints.push_back( cv::Point2d( (d.part(62).x()+
  				       d.part(66).x())*0.5, (d.part(62).y()+d.part(66).y())*0.5 ) );  // Stommion
  imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );                               // Menton

  return imagePoints;

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
      // To reduce computations, this value should be increased
      if ( counter % SKIP_FRAMES == 0 )
      {
        // Detect faces
        faces = detector(cimgSmall);
      }

      // Pose estimation
      std::vector<cv::Point3d> modelPoints = get3dModelPoints();

      // Marker Array begin
      visualization_msgs::MarkerArray marker_arr;

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

        // calculate rotation and translation vector using solvePnP
        cv::Mat rotationVector;
        cv::Mat translationVector;

        translationVector = (cv::Mat_<double>(3,1) << 0., 0., -530.);
        rotationVector = (cv::Mat_<double>(3,3) << 0.0, 0.0, 0.0);

        cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector);

        cv::Mat R;
        cv::Rodrigues(rotationVector, R); // R is 3x3

        R = R.t();  // rotation of inverse
        translationVector = -R * translationVector; // translation of inverse

        // fill up a Marker
        visualization_msgs::Marker new_marker;
        // Grab the position
	new_marker.pose.position.x =translationVector.at<double>(0) ;
        new_marker.pose.position.y =translationVector.at<double>(1) ;
        new_marker.pose.position.z =translationVector.at<double>(2);
        // Grab the orientation

        double theta = cv::norm(rotationVector, CV_L2);

        new_marker.pose.orientation.x = sin(theta / 2)*rotationVector.at<double>(0, 2) / theta;
        new_marker.pose.orientation.y = sin(theta / 2)*rotationVector.at<double>(0, 1) / theta;
        new_marker.pose.orientation.z = sin(theta / 2)*rotationVector.at<double>(0, 0) / theta;
        new_marker.pose.orientation.w = cos(theta / 2);

        // mouth status display
        mouthOpen = checkMouth(shape);
        if (mouthOpen == true){
            cv::putText(im, cv::format("OPEN"), cv::Point(450, 50),
            cv::FONT_HERSHEY_COMPLEX, 1.5,cv::Scalar(0, 0, 255), 3);
            // Grab the mouth status when the mouth is open
            new_marker.text="O";
            } 
	else {
            cv::putText(im, cv::format("CLOSED"), cv::Point(450, 50),
            cv::FONT_HERSHEY_COMPLEX, 1.5,cv::Scalar(0, 0, 255), 3);
            // Grab the mouth status when the mouth is closed
            new_marker.text="C";
            }

            marker_arr.markers.push_back(new_marker);

      }

      // publish the marker array
      marker_array_pub.publish(marker_arr);

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

void cameraInfo(const sensor_msgs::CameraInfoConstPtr& msg)
   {
    int i,j;
    int count=0;
    // Obtain camera parameters from the relevant rostopic
    for(i=0;i<=2;i++) {
        for(j=0;j<=2;j++) {
            cameraMatrix.at<double>(i,j)=msg->K[count];
            count++;
            }
        }

    // Obtain lens distortion from the relevant rostopic
    for(i=0;i<5;i++) {
        distCoeffs.at<double>(i)=msg->D[i];
        }
   }


int main(int argc, char **argv)
{
  try
  {
   ros::init(argc, argv, "image_listener");
   ros::NodeHandle nh;
   image_transport::ImageTransport it(nh);

   std::string MarkerTopic = "/camera/color/image_raw";
   deserialize("../../../src/face_detection/model/shape_predictor_68_face_landmarks.dat") >> predictor;
   image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, imageCallback);
   ros::Subscriber sub_info = nh.subscribe("/camera/color/camera_info", 1, cameraInfo);
   marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("face_pose", 1);

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

