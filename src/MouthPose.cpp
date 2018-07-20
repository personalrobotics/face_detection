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

  // sellion origin
  // X direction points forward projecting out of the person's stomion
  /*
  modelPoints.push_back(cv::Point3d(0.0, 0.0, 0.0));    // Sellion
  modelPoints.push_back(cv::Point3d(-20., -65.5,-5.));  // Right Eye
  modelPoints.push_back(cv::Point3d(-20., 65.5,-5.));   // Left Eye
  modelPoints.push_back(cv::Point3d(-100., -77.5,-6.)); // Right Ear
  modelPoints.push_back(cv::Point3d(-100., 77.5,-6.));  // Left Ear
  modelPoints.push_back(cv::Point3d(21.0, 0., -48.0));  // Nose
  modelPoints.push_back(cv::Point3d(10.0, 0., -75.0));  // Stommion
  modelPoints.push_back(cv::Point3d(0., 0.,-133.0));    // Menton
  */
  // Stomion Origin
  // X direction points forward projecting out of the person's stomion

  modelPoints.push_back(cv::Point3d(0., 0., 0.));  // Stommion
  modelPoints.push_back(cv::Point3d(-30., -65.5,70.0));  // Right Eye
  modelPoints.push_back(cv::Point3d(-30., 65.5,70.));   // Left Eye
  // modelPoints.push_back(cv::Point3d(-110., -77.5,69.)); // Right Ear
  // modelPoints.push_back(cv::Point3d(-110., 77.5,69.));  // Left Ear
  modelPoints.push_back(cv::Point3d(11.0, 0., 27.0));  // Nose
  modelPoints.push_back(cv::Point3d(-10.0, 0.0, 75.0));    // Sellion
 modelPoints.push_back(cv::Point3d(-10., 0.,-58.0));    // Menton


  return modelPoints;

}

// 2D landmark points from all landmarks
std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d)
{

  std::vector<cv::Point2d> imagePoints;

  // Sellion Origin
  /*
  imagePoints.push_back( cv::Point2d( d.part(27).x(), d.part(27).y() ) );   // Sellion
  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );   // Right Eye
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );   // Left Eye
  imagePoints.push_back( cv::Point2d( d.part(0).x(), d.part(0).y() ) );     // Right Ear
  imagePoints.push_back( cv::Point2d( d.part(16).x(), d.part(16).y() ) );   // Left Ear
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );   // Nose
  imagePoints.push_back( cv::Point2d( (d.part(62).x()+
  d.part(66).x())*0.5, (d.part(62).y()+d.part(66).y())*0.5 ) );             // Stommion
  imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );     // Menton
  */
  // Stomion Origin
  imagePoints.push_back( cv::Point2d( (d.part(62).x()+
  d.part(66).x())*0.5, (d.part(62).y()+d.part(66).y())*0.5 ) );             // Stommion
  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );   // Right Eye
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );   // Left Eye
  // imagePoints.push_back( cv::Point2d( d.part(0).x(), d.part(0).y() ) );     // Right Ear
  // imagePoints.push_back( cv::Point2d( d.part(16).x(), d.part(16).y() ) );   // Left Ear
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );   // Nose
  imagePoints.push_back( cv::Point2d( d.part(27).x(), d.part(27).y() ) );   // Sellion
  imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );     // Menton


  return imagePoints;

}

void getQuaternion(cv::Mat R, double Q[])
{
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
 
    if (trace > 0.0) 
    {
        double s = sqrt(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<double>(2,1) - R.at<double>(1,2)) * s);
        Q[1] = ((R.at<double>(0,2) - R.at<double>(2,0)) * s);
        Q[2] = ((R.at<double>(1,0) - R.at<double>(0,1)) * s);
    } 
    
    else 
    {
        int i = R.at<double>(0,0) < R.at<double>(1,1) ? (R.at<double>(1,1) < R.at<double>(2,2) ? 2 : 1) : (R.at<double>(0,0) < R.at<double>(2,2) ? 2 : 0); 
        int j = (i + 1) % 3;  
        int k = (i + 2) % 3;

        double s = sqrt(R.at<double>(i, i) - R.at<double>(j,j) - R.at<double>(k,k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<double>(k,j) - R.at<double>(j,k)) * s;
        Q[j] = (R.at<double>(j,i) + R.at<double>(i,j)) * s;
        Q[k] = (R.at<double>(k,i) + R.at<double>(i,k)) * s;
    }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
      im = cv_bridge::toCvShare(msg, "bgr8")->image;

      cv::rotate(im, im, cv::ROTATE_90_COUNTERCLOCKWISE);

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

        cv::Mat R;

        cv::Mat R_z = (cv::Mat_<double>(3,3) <<
               0.0,    -1.0,      0.0,
               1.0, 0.0, 0.0,
               0.0,0.0,1.0);

        cv::Mat R_y = (cv::Mat_<double>(3,3) <<
               0.0,    0.0,      1.0,
               0.0, 1.0, 0.0,
               -1.0,0.0,0.0);

       cv::Mat R_x = (cv::Mat_<double>(3,3) <<
               1.0,    0.0,      0.0,
               0.0, 0.0, -1.0,
               0.0,1.0,0.0);

        std::vector<int> inliers;
        //std::vector<int>* inliers =NULL;

         float reprojection_error=8.0;
         int num_iters=100;
         bool use_extrinsic_guess=false;
         double confidence=0.99;

        cv::solvePnPRansac(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector,translationVector,
         use_extrinsic_guess,num_iters, reprojection_error,confidence,inliers); 

        //cout << rotationVector << "\n";

        //cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector,translationVector);

        cout << inliers.size() << endl;

        cv::Rodrigues(rotationVector,R);

        R = R_x*R_z*R;

        cv::Rodrigues(R,rotationVector);

        // fill up a Marker
        visualization_msgs::Marker new_marker;

        // Grab the position
        translationVector=R_z*translationVector;

	      new_marker.pose.position.x =(translationVector.at<double>(0)) / 1000;
        new_marker.pose.position.y =(translationVector.at<double>(1)) / 1000;
        new_marker.pose.position.z =(translationVector.at<double>(2)) / 1000;

      /*
       double tr=R.at<double>(0,0)+R.at<double>(1,1)+R.at<double>(2,2);
       double qw,qx,qy,qz;

        if (tr > 0) { 
        float S = sqrt(tr+1.0) * 2; // S=4*qw 
        qw = 0.25 * S;
        qx = (R.at<double>(2,1) - R.at<double>(1,2)) / S;
        qy = (R.at<double>(0,2) - R.at<double>(2,0)) / S; 
        qz = (R.at<double>(1,0) - R.at<double>(0,1)) / S; 
         } 
         else if ((R.at<double>(0,0) > R.at<double>(1,1))& (R.at<double>(0,0) > R.at<double>(2,2))) { 
         float S = sqrt(1.0 + R.at<double>(0,0) - R.at<double>(1,1) - R.at<double>(2,2)) * 2; // S=4*qx 
         qw = (R.at<double>(2,1) - R.at<double>(1,2)) / S;
         qx = 0.25 * S;
         qy = (R.at<double>(0,1) + R.at<double>(1,0)) / S; 
         qz = (R.at<double>(0,2) + R.at<double>(2,0)) / S; 
        } else if (R.at<double>(1,1) > R.at<double>(2,2)) { 
        float S = sqrt(1.0 + R.at<double>(1,1) - R.at<double>(0,0) - R.at<double>(2,2)) * 2; // S=4*qy
        qw = (R.at<double>(0,2) - R.at<double>(2,0)) / S;
        qx = (R.at<double>(0,1) + R.at<double>(1,0)) / S; 
        qy = 0.25 * S;
        qz = (R.at<double>(1,2) + R.at<double>(2,1)) / S; 
       } else { 
       float S = sqrt(1.0 + R.at<double>(2,2) - R.at<double>(0,0) - R.at<double>(1,1)) * 2; // S=4*qz
       qw = (R.at<double>(1,0) - R.at<double>(0,1)) / S;
       qx = (R.at<double>(0,2) + R.at<double>(2,0)) / S;
       qy = (R.at<double>(1,2) + R.at<double>(2,1)) / S;
       qz = 0.25 * S;
       }

       new_marker.pose.orientation.x = qx;
       new_marker.pose.orientation.y = qy;
       new_marker.pose.orientation.z = qz;
       new_marker.pose.orientation.w = qw;
        
        */

/*
        double theta = cv::norm(rotationVector, CV_L2);
      
        // Grab the orientation
        new_marker.pose.orientation.x = sin(theta / 2)*rotationVector.at<double>(2) / theta;
        new_marker.pose.orientation.y = sin(theta / 2)*rotationVector.at<double>(1) / theta;
        new_marker.pose.orientation.z = sin(theta / 2)*rotationVector.at<double>(0) / theta;
        new_marker.pose.orientation.w = cos(theta / 2);   */

        double Q[4];
        getQuaternion(R,Q);

       new_marker.pose.orientation.x = Q[2];
       new_marker.pose.orientation.y = Q[1];
       new_marker.pose.orientation.z = Q[0];
       new_marker.pose.orientation.w = Q[3];

        

        // mouth status display
        mouthOpen = checkMouth(shape);
        if (mouthOpen == true){
            cv::putText(im, cv::format("OPEN"), cv::Point(450, 50),
                cv::FONT_HERSHEY_COMPLEX, 1.5,cv::Scalar(0, 0, 255), 3);
            // Grab the mouth status when the mouth is open
            new_marker.text="{\"id\": \"mouth\", \"mouth-status\": \"open\"}";
            new_marker.ns="mouth";
            } else {
            cv::putText(im, cv::format("CLOSED"), cv::Point(450, 50),
               cv::FONT_HERSHEY_COMPLEX, 1.5,cv::Scalar(0, 0, 255), 3);
            // Grab the mouth status when the mouth is closed
            new_marker.text="{\"id\": \"mouth\", \"mouth-status\": \"closed\"}";
            new_marker.ns="mouth";
            }

            new_marker.header.frame_id="/camera_color_optical_frame";

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

