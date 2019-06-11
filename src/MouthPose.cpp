#include "global.hpp"

using namespace dlib;
using namespace std;
using namespace sensor_msgs;

#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 1
#define OPENCV_FACE_RENDER

static float rotateFace();

// global declarations
int counter = 0;
uint32 stomionPointX, stomionPointY;
Eigen::Quaterniond quats;
uint32 betweenEyesPointX, betweenEyesPointY;
int indexStomion, indexLeftEyeLid, indexRightEyeLid;
cv::Mat rotationVector;
cv::Mat translationVector;
std::unique_ptr<ros::NodeHandle> nh;

bool mouthOpen;               // store status of mouth being open or closed
cv::Mat im;                   // matrix to store the image
std::vector<rectangle> faces; // variable to store face rectangles
cv::Mat imSmall, imDisplay;   // matrices to store the resized image to oprate on and display

// Load face detection and pose estimation models.
frontal_face_detector detector = get_frontal_face_detector(); // get the frontal face
shape_predictor predictor;

ros::Publisher marker_array_pub;

cv::Mat_<double> distCoeffs(5, 1);
cv::Mat_<double> cameraMatrix(3, 3);

double oldX, oldY, oldZ;
bool firstTimeDepth = true;
bool firstTimeImage = true;

static cv::Rect dlibRectangleToOpenCV(rectangle r) {
  return cv::Rect(cv::Point2i(r.left(), r.top()),
                  cv::Point2i(r.right() + 1, r.bottom() + 1));
}

static rectangle openCVRectToDlib(cv::Rect r) {
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1,
                         (long)r.br().y - 1);
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
  bool facePerceptionOn = true;
  if (!nh || !nh->getParam("/feeding/facePerceptionOn", facePerceptionOn)) {
    facePerceptionOn = true;
  }
  if (!facePerceptionOn) {
    cv::destroyAllWindows();
    return;
  }
  try {
    im = cv_bridge::toCvShare(msg, "bgr8")->image;

    // cv::rotate(im, im, cv::ROTATE_90_COUNTERCLOCKWISE);

    // Create imSmall by resizing image for face detection
    cv::resize(im, imSmall, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO,
               1.0 / FACE_DOWNSAMPLE_RATIO);

    // Change to dlib's image format. No memory is copied.
    cv_image<bgr_pixel> cimgSmall(imSmall);
    cv_image<bgr_pixel> cimg(im);

    float rotAngle = 0.0f;

    // Process frames at an interval of SKIP_FRAMES.
    // This value should be set depending on your system hardware
    // and camera fps.
    // To reduce computations, this value should be increased
    if (counter % SKIP_FRAMES == 0) {
      // Detect faces
      faces = detector(cimgSmall);
      if (faces.size() == 0) {
        // if no faces detected, rotate frame to find faces
        // and rotate image back to original
        rotAngle = rotateFace();
      }
    }
    if (faces.size() == 0) {
      // No faces detected
      stomionPointX = 0;
      stomionPointY = 0;
    }

    // Pose estimation
    std::vector<cv::Point3d> modelPoints = get3dModelPoints();

    // Iterate over faces
    // cout << "Detected Faces: " << faces.size() << endl;
    // cout << "Final rotation: " << rotAngle << endl;
    for (unsigned long i = 0; i < faces.size(); ++i) {
      // Since we ran face detection on a resized image,
      // we will scale up coordinates of face rectangle
      rectangle r((long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
                  (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
                  (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
                  (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO));

      // rotate big image pre-detection (if needed)
      cv::Mat imRot;
      cv::Point2f center(im.cols / 2.0, im.rows / 2.0);
      cv::Mat rot = cv::getRotationMatrix2D(center, rotAngle, 1.0);
      cv::warpAffine(im, imRot, rot, im.size());
      cv_image<bgr_pixel> cimgRot(imRot);

      // Find face landmarks by providing rectangle for each face
      full_object_detection shape = predictor(cimgRot, r);

      // get 2D landmarks from Dlib's shape object
      std::vector<cv::Point2d> imagePoints = get2dImagePoints(shape);

      // get corners of face rectangle
      cv::Rect cvRect = dlibRectangleToOpenCV(r);
      std::vector<cv::Point2d> rectCorners{cvRect.tl(), cvRect.br()};

      // Rotate image points back
      rot = cv::getRotationMatrix2D(center, -rotAngle, 1.0);
      cv::transform(imagePoints, imagePoints, rot);
      cv::transform(rectCorners, rectCorners, rot);
      cv::Rect cvRectRotated(rectCorners[0], rectCorners[1]);
      // Draw landmarks over face (circles and rectangle)
      // renderFace(im, shape);
      renderFace(im, imagePoints, cv::Scalar(255, 200, 0));
      cv::rectangle(im, cvRectRotated, cv::Scalar( 255, 0, 0 ));
      // cv::rectangle(im, dlibRectangleToOpenCV(r), cv::Scalar(0, 255, 0));

      stomionPointX = imagePoints[0].x;
      stomionPointY = imagePoints[0].y;
      betweenEyesPointX =
          (imagePoints[indexLeftEyeLid].x + imagePoints[indexRightEyeLid].x) /
          2;
      betweenEyesPointY =
          (imagePoints[indexLeftEyeLid].y + imagePoints[indexRightEyeLid].y) /
          2;

      // calculate rotation and translation vector using solvePnP
      cv::solvePnPRansac(modelPoints, imagePoints, cameraMatrix, distCoeffs,
                         rotationVector, translationVector);

      // Convert rotation vector to rotation matrix
      cv::Mat R; // ouput rotation matrix
      cv::Rodrigues(rotationVector, R);

      Eigen::AngleAxisd rollAngle(3.14159, Eigen::Vector3d::UnitZ());
      Eigen::AngleAxisd yawAngle(0, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd pitchAngle(0, Eigen::Vector3d::UnitX());
      Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
      Eigen::Matrix3d zRot = q.matrix();

      Eigen::Matrix3d mat;
      cv::cv2eigen(R, mat);
      mat = zRot * mat;
      Eigen::Quaterniond EigenQuat(mat);

      quats = EigenQuat;

      auto euler = quats.toRotationMatrix().eulerAngles(0, 1, 2);
      std::cout << "Euler from quaternion in roll, pitch, yaw" << std::endl
                << euler << std::endl;

      // mouth status display
      mouthOpen = checkMouth(shape);
    }

    firstTimeImage = false;

    imDisplay = im;
    cv::resize(im, imDisplay, cv::Size(), 1, 1);
    cv::imshow("Face Pose Detector", imDisplay);
    cv::waitKey(30);

  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

static float rotateFace() {
  // cout << "[Rotation] No faces detected, attempting rotations." <<
  // endl;
  cv::Mat imRot;
  cv::Point2f center(imSmall.cols / 2.0, imSmall.rows / 2.0);
  static float angles[2] = {-55.0f, 55.0f};
  for (int i = 0; i < 2; i++) {
    // cout << "[Rotation] Rotating by (degrees): " << angles[i] << endl;
    cv::Mat rot = cv::getRotationMatrix2D(center, angles[i], 1.0);
    cv::warpAffine(imSmall, imRot, rot, imSmall.size());
    cv_image<bgr_pixel> cimgRot(imRot);
    faces = detector(cimgRot);
    if (faces.size() > 0) {
      // Detected a face! Rotate bounding rectangle back
      // cout << "[Rotation] Detected face at (degrees): " << angles[i] <<
      // endl;
      return angles[i]; // No need to rotate other direction
    }
  }
  return 0.0f;
}

void publishMarker(float tx, float ty, float tz) {
  visualization_msgs::Marker new_marker;
  new_marker.pose.position.x = tx;
  new_marker.pose.position.y = ty;
  new_marker.pose.position.z = tz;

  // Accounting for apparent facial rotation
  Eigen::Vector3d forwardZ(0, 0, 1);
  Eigen::Vector3d cameraTranslation(tx, ty, tz);
  // cout << translationVector << "  " << cameraTranslation;
  Eigen::Quaterniond cameraRot =
      Eigen::Quaterniond().setFromTwoVectors(forwardZ, cameraTranslation);
  Eigen::Quaterniond newRot = cameraRot * quats;

  new_marker.pose.orientation.x = newRot.vec()[0];
  new_marker.pose.orientation.y = newRot.vec()[1];
  new_marker.pose.orientation.z = newRot.vec()[2];
  new_marker.pose.orientation.w = newRot.w();

  // Additional Visualization
  new_marker.scale.x = 1;
  new_marker.scale.y = 1;
  new_marker.scale.z = 1;

  new_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  new_marker.mesh_resource = "package://pr_assets/data/objects/tom.dae";
  new_marker.action = visualization_msgs::Marker::ADD;
  new_marker.id = 1;
  new_marker.header.stamp = ros::Time();
  new_marker.color.a = 1.0;
  new_marker.color.r = 0.5;
  new_marker.color.g = 0.5;
  new_marker.color.b = 0.5;

  // mouth status display
  if (mouthOpen == true) {
    cv::putText(im, cv::format("OPEN"), cv::Point(450, 50),
                cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    // Grab the mouth status when the mouth is open
    new_marker.text = "{\"db_key\": \"mouth\", \"mouth-status\": \"open\"}";
    new_marker.ns = "mouth";
  } else {
    cv::putText(im, cv::format("CLOSED"), cv::Point(450, 50),
                cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    // Grab the mouth status when the mouth is closed
    new_marker.text = "{\"db_key\": \"mouth\", \"mouth-status\": \"closed\"}";
    new_marker.ns = "mouth";
  }

  new_marker.header.frame_id = "/camera_color_optical_frame";

  visualization_msgs::MarkerArray marker_arr;

  if (tx != 0 || ty != 0 || tz != 0) {
    marker_arr.markers.push_back(new_marker);
  } // else no perception

  marker_array_pub.publish(marker_arr);
}

void DepthCallBack(const sensor_msgs::ImageConstPtr depth_img_ros) {
  bool facePerceptionOn = true;
  if (!nh || !nh->getParam("/feeding/facePerceptionOn", facePerceptionOn)) {
    facePerceptionOn = true;
  }
  if (!facePerceptionOn) {
    cv::destroyAllWindows();
    return;
  }

  cv_bridge::CvImageConstPtr depth_img_cv;
  cv::Mat depth_mat;
  // Get the ROS image to openCV
  depth_img_cv = cv_bridge::toCvShare(depth_img_ros,
                                      sensor_msgs::image_encodings::TYPE_16UC1);
  // Convert the uints to floats
  depth_img_cv->image.convertTo(depth_mat, CV_32F, 0.001);

  if (betweenEyesPointX >= depth_mat.cols ||
      betweenEyesPointY >= depth_mat.rows) {
    std::cout << "invalid points or depth mat. points: (" << betweenEyesPointX
              << ", " << betweenEyesPointY << ") mat: (" << depth_mat.cols
              << ", " << depth_mat.rows << ")" << std::endl;
    return;
  }

  cout << "depth: " << depth_mat.at<float>(betweenEyesPointX, betweenEyesPointY)
       << "  point between eyes at: (" << betweenEyesPointX << ", "
       << betweenEyesPointY << ") mat: (" << depth_mat.cols << ", "
       << depth_mat.rows << ")" << std::endl;

  float averageDepth = 0;
  int depthCounts = 0;

  for (int x = std::max(0, (int)betweenEyesPointX - 4);
       x < std::min(depth_mat.cols, (int)betweenEyesPointX + 5); x++) {
    for (int y = std::max(0, (int)betweenEyesPointY - 4);
         y < std::min(depth_mat.rows, (int)betweenEyesPointY + 5); y++) {
      float depth = depth_mat.at<float>(y, x);
      if (depth > 0.0001) {
        averageDepth += depth;
        depthCounts++;
      }
      //    std::cout << depth << ", ";
    }
    //  std::cout << std::endl;
  }
  averageDepth /= depthCounts;

  std::cout << "average depth: " << averageDepth << std::endl;

  if (depthCounts == 0) {
    std::cout << "depth between eyes is zero! Skipping..." << std::endl;
    return;
  }

  double cam_fx = cameraMatrix.at<double>(0, 0);
  double cam_fy = cameraMatrix.at<double>(1, 1);
  double cam_cx = cameraMatrix.at<double>(0, 2);
  double cam_cy = cameraMatrix.at<double>(1, 2);
  double tz = averageDepth;
  double tx = (tz / cam_fx) * (stomionPointX - cam_cx);
  double ty = (tz / cam_fy) * (stomionPointY - cam_cy);
  // tvec = np.array([tx, ty, tz])
  // cout << "position: (" << tx << ", " << ty << ", " << tz << ")" << endl;

  if (firstTimeImage) {
    std::cout << "skipping because image not yet received" << std::endl;
    return;
  }

  if (stomionPointX == 0) {
    std::cout << "No face detected in image callback. Skipping." << std::endl;
    publishMarker(0, 0, 0);
    return;
  }

  double squareDist = (tx - oldX) * (tx - oldX) + (ty - oldY) * (ty - oldY) +
                      (tz - oldZ) * (tz - oldZ);

  if (tz < 0.3) {
    std::cout << "calculated depth too close. Skipping frame" << std::endl;
    publishMarker(oldX, oldY, oldZ);
    return;
  }

  if (tz > 1.0) {
    std::cout << "calculated depth too far. Skipping frame" << std::endl;
    publishMarker(oldX, oldY, oldZ);
    return;
  }

  if (squareDist > 0.2 * 0.2 && !firstTimeDepth) {
    std::cout << "calculated pose would be too far" << std::endl;
  }

  firstTimeDepth = false;
  oldX = tx;
  oldY = ty;
  oldZ = tz;

  publishMarker(tx, ty, tz);
}

void cameraInfo(const sensor_msgs::CameraInfoConstPtr &msg) {
  int i, j;
  int count = 0;
  // Obtain camera parameters from the relevant rostopic
  for (i = 0; i <= 2; i++) {
    for (j = 0; j <= 2; j++) {
      cameraMatrix.at<double>(i, j) = msg->K[count];
      count++;
    }
  }

  // Obtain lens distortion from the relevant rostopic
  for (i = 0; i < 5; i++) {
    distCoeffs.at<double>(i) = msg->D[i];
  }
}

int main(int argc, char **argv) {
  try {
    ros::init(argc, argv, "image_listener");
    nh = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle);
    image_transport::ImageTransport it(*nh);

    std::string MarkerTopic = "/camera/color/image_raw";
    std::string path = ros::package::getPath("face_detection");
    deserialize(path + "/model/shape_predictor_68_face_landmarks.dat") >>
        predictor;
    ros::Subscriber sub_info =
        nh->subscribe("/camera/color/camera_info", 1, cameraInfo);
    image_transport::Subscriber sub =
        it.subscribe("/camera/color/image_raw", 1, imageCallback,
                     image_transport::TransportHints("compressed"));

    ros::Subscriber sub_depth = nh->subscribe(
        "/camera/aligned_depth_to_color/image_raw", 1, DepthCallBack);
    marker_array_pub =
        nh->advertise<visualization_msgs::MarkerArray>("face_pose", 1);

    ros::spin();
  } catch (serialization_error &e) {
    cout << "Shape predictor model file not found" << endl;
    cout << "Put shape_predictor_68_face_landmarks in models directory" << endl;
    cout << endl << e.what() << endl;
  } catch (exception &e) {
    cout << e.what() << endl;
  }
}
