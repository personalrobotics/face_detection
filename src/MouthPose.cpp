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
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <mutex>

using namespace dlib;
using namespace std;
using namespace sensor_msgs;

#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 1
#define OPENCV_FACE_RENDER

// global declarations

bool depthCallbackBool=false;
bool imgCallbackBool=false;
bool recieved=false;

cv::Mat depth_mat;

std::vector<uint32> abscissae;
std::vector<uint32> ordinates;

std::vector<double>WorldFrameApplicates;
std::vector<double>WorldFrameOrdinates;
std::vector<double>WorldFrameAbscissae;

std::vector<cv::Point3d> RealWorld3D;

std::vector<cv::Point3d> modelPoints3D;
std::vector<cv::Point3d> modelPoints3DReal;

std::vector<cv::Point2d> imagePoints;
std::vector<cv::Point2d> imagePoints1;

cv::Mat rotationVector;
cv::Mat translationVector;

cv::Mat rotationVector1;
cv::Mat translationVector1;

cv::Mat temp;

int flags;
bool mouthOpen; // store status of mouth being open or closed
cv::Mat im; // matrix to store the image
int counter=0;
std::vector<rectangle> faces; // variable to store face rectangles
cv::Mat imSmall, imDisplay; // matrices to store the resized image to oprate on and display
// Load face detection and pose estimation models.
frontal_face_detector detector = get_frontal_face_detector(); // get the frontal face
shape_predictor predictor;

ros::Publisher marker_array_pub;
visualization_msgs::MarkerArray store_marker;

cv::Mat_<double> distCoeffs(5,1);
cv::Mat_<double> cameraMatrix(3,3);

// Distance funtion to obtain relative distances/co-ordinates for the 3D model points in the real world

cv::Point3d dist(cv::Point3d Origin ,cv::Point3d Point)
{

cv::Point3d co_ordinates;

co_ordinates.x=Point.x-Origin.x;
co_ordinates.y=Point.y-Origin.y;
co_ordinates.z=Point.z-Origin.z;

return co_ordinates;

}


// 3D Model Points of selected landmarks in an arbitrary frame of reference

std::vector<cv::Point3d> get3dModelPoints()
{
  std::vector<cv::Point3d> modelPoints;

  // Stomion Origin
  // X direction points forward projecting out of the person's stomion

  modelPoints.push_back(cv::Point3d(0., 0., 0.));  // Stommion
  modelPoints.push_back(cv::Point3d(-30.0, -65.5,70.0));  // Right Eye
  modelPoints.push_back(cv::Point3d(-30.0, 65.5,70.));   // Left Eye
  modelPoints.push_back(cv::Point3d(11.0, 0., 27.0));  // Nose
  modelPoints.push_back(cv::Point3d(-10.0, 0.0, 75.0));    // Sellion
  modelPoints.push_back(cv::Point3d(-10.0, 0.,-58.0));    // Menton
  modelPoints.push_back(cv::Point3d(-10.0,-3.4,75.0)); // Right Eye Lid
  modelPoints.push_back(cv::Point3d(-10.0,3.4,75.0)); // Left Eye Lid
  modelPoints.push_back(cv::Point3d(-5.0,-2.5,0.0)); // Right Lip corner
  modelPoints.push_back(cv::Point3d(-5.0,2.5,0.0)); // Left Lip corner
  modelPoints.push_back(cv::Point3d(-115.0,-77.5,69.0)); // Right side
  modelPoints.push_back(cv::Point3d(-115.0,77.5,69.0));  // Left side

  return modelPoints;

}

std::vector<cv::Point3d> get3dRealModelPoints()
{
  std::vector<cv::Point3d> modelPoints;

  modelPoints.clear();

  for(int i=0;i<RealWorld3D.size();i++)
  modelPoints.push_back(cv::Point3d(dist(RealWorld3D[0],RealWorld3D[i]).x, dist(RealWorld3D[0],RealWorld3D[i]).y,dist(RealWorld3D[0],RealWorld3D[i]).z));

  return modelPoints;

}

// 2D landmark points from all landmarks
std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d)
{

  imagePoints.clear();
  // Stomion Origin
  imagePoints.push_back( cv::Point2d( (d.part(62).x()+
  d.part(66).x())*0.5, (d.part(62).y()+d.part(66).y())*0.5 ) );             // Stommion
  //imagePoints.push_back( cv::Point2d( d.part(66).x(),d.part(66).y()  ) );             // Stommion
  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );   // Right Eye
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );   // Left Eye
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );   // Nose
  imagePoints.push_back( cv::Point2d( d.part(27).x(), d.part(27).y() ) );   // Sellion
  imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );     // Menton
  imagePoints.push_back( cv::Point2d( d.part(38).x(), d.part(38).y() ) );     // Right Eye Lid
  imagePoints.push_back( cv::Point2d( d.part(43).x(), d.part(43).y() ) );     // Left Eye Lid
  imagePoints.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );     // Right Lip Corner
  imagePoints.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );     // Left Lip Corner
  imagePoints.push_back( cv::Point2d( d.part(0).x(), d.part(0).y() ) );     // Right Side
  imagePoints.push_back( cv::Point2d( d.part(16).x(), d.part(16).y() ) );     // Left Side


  return imagePoints;

}

std::vector<cv::Point2d> get2dImagePoints1(full_object_detection &d)
{

  imagePoints1.clear();
  // Stomion Origin
  //imagePoints.push_back( cv::Point2d( (d.part(62).x()+
  //d.part(66).x())*0.5, (d.part(62).y()+d.part(66).y())*0.5 ) );             // Stommion
  imagePoints1.push_back( cv::Point2d( d.part(66).x(),d.part(66).y()  ) );             // Stommion
  imagePoints1.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );   // Right Eye
  imagePoints1.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );   // Left Eye
  imagePoints1.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );   // Nose
  imagePoints1.push_back( cv::Point2d( d.part(27).x(), d.part(27).y() ) );   // Sellion
  //imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );     // Menton
  imagePoints1.push_back( cv::Point2d( d.part(38).x(), d.part(38).y() ) );     // Right Eye Lid
  imagePoints1.push_back( cv::Point2d( d.part(43).x(), d.part(43).y() ) );     // Left Eye Lid
  imagePoints1.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );     // Right Lip Corner
  imagePoints1.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );     // Left Lip Corner

  imagePoints1.push_back( cv::Point2d( d.part(28).x(), d.part(28).y() ) ); // Nose Point 1
  imagePoints1.push_back( cv::Point2d( d.part(29).x(), d.part(29).y() ) ); // Nose Point 2
  imagePoints1.push_back( cv::Point2d( d.part(17).x(), d.part(17).y() ) ); // Outer Eyebrow Tip Right
  imagePoints1.push_back( cv::Point2d( d.part(26).x(), d.part(26).y() ) ); // Outer Eyebrow Tip Left
  imagePoints1.push_back( cv::Point2d( d.part(21).x(), d.part(21).y() ) ); // Inner Eyebrow Tip Right
  imagePoints1.push_back( cv::Point2d( d.part(22).x(), d.part(22).y() ) ); // Inner Eyebrow Tip Left */


  return imagePoints1;

}


void method()
{

    if((depthCallbackBool && imgCallbackBool))
    {

      // Create imSmall by resizing image for face detection
      cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

      // Change to dlib's image format. No memory is copied.
      cv_image<bgr_pixel> cimgSmall(imSmall);
      cv_image<bgr_pixel> cimg(im);

      // from image callback

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

      // Marker Array begin
      visualization_msgs::MarkerArray marker_arr;

      // Iterate over faces
      std::vector<full_object_detection> shapes;
      for (unsigned long i = 0; i < faces.size(); ++i)
      {
       abscissae.clear();
       ordinates.clear();
       WorldFrameApplicates.clear();
       WorldFrameOrdinates.clear();
       WorldFrameAbscissae.clear();
       RealWorld3D.clear();
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

       //modelPoints3D.clear();

       // get 2D landmarks from Dlib's shape object
       std::vector<cv::Point2d> imagePoints = get2dImagePoints(shape);
       std::vector<cv::Point2d> imagePoints1 = get2dImagePoints1(shape);

       for(int i=0;i<imagePoints1.size();i++)
       {
       abscissae.push_back(imagePoints1[i].x);
       ordinates.push_back(imagePoints1[i].y);
       }

       // from depth callback

       double cam_fx = cameraMatrix.at<double>(0, 0);
       double cam_fy = cameraMatrix.at<double>(1, 1);
       double cam_cx = cameraMatrix.at<double>(0, 2);
       double cam_cy = cameraMatrix.at<double>(1, 2);

       // Obtain depth values of chosen facial landmark points, these are the applicates in the real world frame


       for(int i=0;i<abscissae.size();i++)
       {
       WorldFrameApplicates.push_back(depth_mat.at<float>(abscissae[i], ordinates[i]));
       cout<<WorldFrameApplicates[i]<<endl;
       }

       // Obtain the abscissae and ordinates of the real world co-ordinates in the world frame

       for(int j=0;j<abscissae.size();j++)
       {
       WorldFrameAbscissae.push_back((WorldFrameApplicates[j] / cam_fx) * (abscissae[j] - cam_cx));
       WorldFrameOrdinates.push_back((WorldFrameApplicates[j] / cam_fx) * (ordinates[j] - cam_cy));
       }

       // store the abscissae, ordinates and applicates of the real world co-ordinates in the world frame

       for(int k=0;k<abscissae.size();k++)
       RealWorld3D.push_back(cv::Point3d(WorldFrameAbscissae[k],WorldFrameOrdinates[k],WorldFrameApplicates[k]));


       modelPoints3D = get3dModelPoints();
       modelPoints3DReal=get3dRealModelPoints();

       // calculate rotation and translation vector using solvePnP

       cv::Mat R;

       cv::solvePnP(modelPoints3DReal, imagePoints1, cameraMatrix, distCoeffs, rotationVector,translationVector, cv::SOLVEPNP_ITERATIVE);
       cv::solvePnP(modelPoints3D, imagePoints, cameraMatrix, distCoeffs, rotationVector1,translationVector1, cv::SOLVEPNP_ITERATIVE);
       //cv::solvePnPRansac(modelPoints3D, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector,flags=cv::SOLVEPNP_P3P);

       Eigen::Vector3d Translate;
       Eigen::Quaterniond quats;

       cv::Rodrigues(rotationVector1,R);
       //R=R*R_z;
       Eigen::Matrix3d mat;
       cv::cv2eigen(R, mat);
       Eigen::Quaterniond EigenQuat(mat);
       quats = EigenQuat;


       // fill up a Marker
       visualization_msgs::Marker new_marker;

       // Grab the position

       new_marker.pose.position.x =(translationVector.at<double>(0));
       new_marker.pose.position.y =(translationVector.at<double>(1));
       new_marker.pose.position.z =(translationVector.at<double>(2));

       new_marker.pose.orientation.x = quats.vec()[0];
       new_marker.pose.orientation.y = quats.vec()[1];
       new_marker.pose.orientation.z = quats.vec()[2];
       new_marker.pose.orientation.w = quats.w();

       new_marker.scale.x = 1;
       new_marker.scale.y = 1;
       new_marker.scale.z = 1;

       new_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
       new_marker.action = visualization_msgs::Marker::ADD;
       new_marker.id = 78234;
       new_marker.header.stamp = ros::Time();
       new_marker.color.a = 1.0;
       new_marker.color.r = 1.0;
       new_marker.color.g = 1.0;
       new_marker.color.b = 1.0;
       new_marker.mesh_resource = "package://pr_ordata/data/objects/tom.dae";


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

      // Project a 3D point (50.0, 0.0, 100.0) onto the image plane.
      // We use this to draw a line sticking out of the stommion
       std::vector<cv::Point3d> StomionPoint3D;
       std::vector<cv::Point2d> StomionPoint2D;
       StomionPoint3D.push_back(cv::Point3d(50.0,0.0,100.0));
       cv::projectPoints(StomionPoint3D, rotationVector1, translationVector1, cameraMatrix, distCoeffs, StomionPoint2D);

      // draw line between stomion points in image and 3D stomion points
      // projected to image plane
       cv::line(im, imagePoints[0],StomionPoint2D[0] , cv::Scalar(255,0,0), 2);


       std::vector<cv::Point2d> reprojectedPoints;
       cv::projectPoints(modelPoints3D, rotationVector1, translationVector1, cameraMatrix, distCoeffs, reprojectedPoints);
       //cv::projectPoints(modelPoints3DReal, rotationVector, translationVector, cameraMatrix, distCoeffs, reprojectedPoints);
       cout << "reprojectedPoints size: "<< reprojectedPoints.size()<<endl;
       for (auto point : reprojectedPoints) {
           cv::circle(im, point, 1, cv::Scalar(50, 255, 70, 255), 3);
           }

       store_marker=marker_arr;
       recieved=true;
      }

      // publish the marker array and retain previous value if no face is detected
      if(recieved){
      marker_array_pub.publish(marker_arr);
      recieved=false;
      }
      else
      marker_array_pub.publish(store_marker);

      // Resize image for display
      imDisplay = im;
      cv::resize(im, imDisplay, cv::Size(), 1, 1);
      cv::imshow("Face Pose Detector", imDisplay);
      cv::waitKey(1);

      }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
      im = cv_bridge::toCvShare(msg, "bgr8")->image;
      imgCallbackBool=true;
      method();

  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

void DepthCallBack(const sensor_msgs::ImageConstPtr depth_img_ros){

  cv_bridge::CvImageConstPtr depth_img_cv;
  depth_img_cv = cv_bridge::toCvShare (depth_img_ros, sensor_msgs::image_encodings::TYPE_16UC1); // Get the ROS image to openCV
  depth_img_cv->image.convertTo(depth_mat, CV_32F, 0.001);  // Convert the uints to floats
  depthCallbackBool=true;

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
   ros::Subscriber sub_info = nh.subscribe("/camera/color/camera_info", 1, cameraInfo);
   ros::Subscriber sub_depth = nh.subscribe("/camera/aligned_depth_to_color/image_raw", 1, DepthCallBack );
   image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 1, imageCallback, image_transport::TransportHints("compressed"));
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

