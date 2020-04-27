#include "global.hpp"

using namespace dlib;
using namespace cv;
using namespace std;
using namespace sensor_msgs;

#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 30
#define OPENCV_FACE_RENDER

static float rotateFace();

// global declarations
static int counter = 0;
static uint32 stomionPointX, stomionPointY;
static Eigen::Quaterniond quats;
static uint32 betweenEyesPointX, betweenEyesPointY;
static int indexStomion, indexLeftEyeLid, indexRightEyeLid;
static cv::Mat rotationVector;
static cv::Mat translationVector;
static std::unique_ptr<ros::NodeHandle> nh;

static bool mouthOpen;               // store status of mouth being open or closed
static cv::Mat im;                   // matrix to store the image
static std::vector<dlib::rectangle> faces; // variable to store face rectangles
static cv::Mat imSmall, imDisplay;   // matrices to store the resized image to oprate on and display

// Load face detection and pose estimation models.
static frontal_face_detector detector = get_frontal_face_detector(); // get the frontal face
static shape_predictor predictor;

static ros::Publisher marker_array_pub;

// Initialize KCF tracker
static Ptr<Tracker> tracker1;
static Ptr<Tracker> tracker2;
static Ptr<Tracker> tracker3;

static cv::Rect2d mouth_bbox; // Initial region of interest. Will be set later after detetcing the mouth
static cv::Rect2d left_eye;
static cv::Rect2d right_eye;

static cv::Mat_<double> distCoeffs(5, 1);
static cv::Mat_<double> cameraMatrix(3, 3);

static double oldX, oldY, oldZ;
static bool firstTimeDepth = true;
static bool firstTimeImage = true;
static bool tracking = false;

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r) {
  return cv::Rect(cv::Point2i(r.left(), r.top()),
                  cv::Point2i(r.right() + 1, r.bottom() + 1));
}

static dlib::rectangle openCVRectToDlib(cv::Rect r) {
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1,
                         (long)r.br().y - 1);
}

void DepthCallBack();

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
    //cv_image<bgr_pixel> cimgSmall(imSmall);
    //cv_image<bgr_pixel> cimg(im);

    //matrix<rgb_pixel> matrix;
    //assign_image(matrix, cimgSmall);

    float rotAngle = 0.0f;


    // Process frames at an interval of SKIP_FRAMES.
    // This value should be set depending on your system hardware
    // and camera fps.
    // To reduce computations, this value should be increased
    if (!tracking && counter % SKIP_FRAMES == 0) {
      // Detect faces
      bool detected = RunFaceDetection(imSmall, 0.0);
      
      if (!detected) {
        // if no faces detected, rotate frame to find faces
        // and rotate image back to original
        
        rotAngle = rotateFace();
        if(rotAngle != 0.0) {
          detected = true;
        }
      
      if (!detected) {
       // No faces detected
        stomionPointX = 0;
        stomionPointY = 0;
      } else {
      	tracker1 = cv::TrackerKCF::create();
      	tracker2 = cv::TrackerKCF::create();
      	tracker3 = cv::TrackerKCF::create();

        tracker1->init(im, mouth_bbox);
        tracker2->init(im, left_eye);
        tracker3->init(im, right_eye);

        stomionPointX = mouth_bbox.x + (mouth_bbox.width / 2);
        stomionPointY = mouth_bbox.y + (mouth_bbox.height / 2);
        tracking = true;
      } 
      // initialize tracking 
      // find the stomion point from mouth bbox
    }

    if(tracking) {
      // Update trackers 
      // find stomion point
      int leftX = left_eye.x;
      int leftY = left_eye.y;
      int rightX = right_eye.x;
      int rightY = right_eye.y;

      // Update the tracking result
      // Run eye trackers
      // Results are better if we keep running the eye trackers togther
      // instead of just running them when mouth tracking 
      bool success = tracker1->update(im, mouth_bbox);
      bool left = tracker2->update(im, left_eye);
      bool right = tracker3->update(im, right_eye);

      if (success) {
        // Mouth Tracking success : Draw the bounding box around the tracked mouth
        cv::rectangle(im, left_eye, Scalar(255, 0, 0), 2, 1);
        cv::rectangle(im, right_eye, Scalar(255, 0, 0), 2, 1);
        cv::rectangle(im, mouth_bbox, Scalar(255, 0, 0), 2, 1);
        putText(im, "Mouth tracking success", Point(100, 80),
        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        stomionPointX = mouth_bbox.x + (mouth_bbox.width / 2);
        stomionPointY = mouth_bbox.y + (mouth_bbox.height / 2);
      } else if (left && right) {
        // find stomion point from eyes
        putText(im, "Estimating mouth from the eyes", Point(100, 120),
                FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        findStomionPoint(leftX, leftY, rightX, rightY);
        cv::rectangle(im, left_eye, Scalar(255, 0, 0), 2, 1);
        cv::rectangle(im, right_eye, Scalar(255, 0, 0), 2, 1);
        cv::circle(im, Point(stomionPointX, stomionPointY), 5, Scalar(255, 0, 0), -1, 8, 0);
      } else {
        // fall back to detection
        putText(im, "Running Face detection again", Point(100, 120),
                FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        tracker1->clear();
        tracker2->clear();
        tracker3->clear();

        bool detected = RunFaceDetection(imSmall, 0.0);
      
      	if (!detected) {
	        // if no faces detected, rotate frame to find faces
	        // and rotate image back to original
        
	        rotAngle = rotateFace();

	        if(rotAngle != 0.0) {
	          detected = true;
	        }
      	}

      	if (!detected) {
       		// No faces detected
        	stomionPointX = 0;
        	stomionPointY = 0;
        	putText(im, "Failed Face Detection", Point(100, 160),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        	tracking = false;
      	} else {
	      	tracker1 = cv::TrackerKCF::create();
	      	tracker2 = cv::TrackerKCF::create();
	      	tracker3 = cv::TrackerKCF::create();

	        tracker1->init(im, mouth_bbox);
	        tracker2->init(im, left_eye);
	        tracker3->init(im, right_eye);

	        stomionPointX = mouth_bbox.x + (mouth_bbox.width / 2);
	        stomionPointY = mouth_bbox.y + (mouth_bbox.height / 2);
	        cv::rectangle(im, mouth_bbox, Scalar(255, 0, 0), 2, 1);

	        tracking = true;
      	} 
        
      }
  }
      











      /*

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

        stomionPointX = imagePoints[0].x;
        stomionPointY = imagePoints[0].y;uint32
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

        /*
        auto euler = quats.toRotationMatrix().eulerAngles(0, 1, 2);
        std::cout << "Euler from quaternion in roll, pitch, yaw" << std::endl
                  << euler << std::endl;
         ////

        // mouth status display
        mouthOpen = checkMouth(shape);

      }FileStorage

      firstTimeImage = false;
      */

      
      imDisplay = im;
      cv::resize(im, imDisplay, cv::Size(), 1, 1);
      cv::imshow("Face Pose Detector", imDisplay);
      cv::waitKey(30);

      // DepthCallBack();
    }

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
     if (RunFaceDetection(imRot, angles[i])) {
      return angles[i];
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
    // Grab the mouth status when the mouth is open
    new_marker.text = "{\"db_key\": \"mouth\", \"mouth-status\": \"open\"}";
    new_marker.ns = "mouth";
    std::cout << "OPEN" << std::endl;
  } else {
    // Grab the mouth status when the mouth is closed
    new_marker.text = "{\"db_key\": \"mouth\", \"mouth-status\": \"closed\"}";
    new_marker.ns = "mouth";
    std::cout << "CLOSED" << std::endl;
  }

  new_marker.header.frame_id = "/camera_color_optical_frame";

  visualization_msgs::MarkerArray marker_arr;

  if (tx != 0 || ty != 0 || tz != 0) {
    marker_arr.markers.push_back(new_marker);
  } // else no perception

  std::cout << "Publishing Marker!" << std::endl;
  marker_array_pub.publish(marker_arr);
}

void DepthCallBack() {
  bool facePerceptionOn = true;
  if (!nh || !nh->getParam("/feeding/facePerceptionOn", facePerceptionOn)) {
    facePerceptionOn = true;
  }
  if (!facePerceptionOn) {
    cv::destroyAllWindows();
    return;
  }

  /*

  cv::Mat depth_mat;
  std::cout << "Got Depth Image... ";
  // Get the ROS image to openCV
  auto depth_img_cv = cv_bridge::toCvShare(depth_img_ros, sensor_msgs::image_encodings::TYPE_16UC1);
  std::cout << "done 1...";
  // Convert the uints to floats
  depth_img_cv->image.convertTo(depth_mat, CV_32F, 0.001);
  std::cout << "done 2! ";

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
  */

  float averageDepth = 0.45;
  /*
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
  */

  std::cout << "average depth: " << averageDepth << std::endl;
/*
  if (depthCounts == 0) {
    std::cout << "depth between eyes is zero! Skipping..." << std::endl;
    return;
  }
*/

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

bool RunFaceDetection(const cv::Mat &frame, float rotAngle) {
  try {
    // convert OpenCV's MAT to dlib's array2d
    dlib::array2d<dlib::bgr_pixel> img;
    dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(cvIplImage(frame)));

    // Dectect the faces
    std::vector<dlib::rectangle> dets = detector(img);
    cout << "Number of faces detected: " << dets.size() << endl;
    if (dets.size() == 0) {
      cerr << "No faces detected" << endl; 
      mouth_bbox = cv::Rect2d(-1, -1, -1, -1);
      return false;
    } else if (dets.size() != 1) {
      cerr << "Mutiple faces detected. Cannot track multiple faces" << endl;
      mouth_bbox = cv::Rect2d(-1, -1, -1, -1);
      return false;
    }

    std::vector<dlib::full_object_detection> shapes;
    // Add the facial features for all the detected faces into a vector
    for (unsigned long j = 0; j < dets.size(); ++j) {
      dlib::full_object_detection shape = predictor(img, dets[j]);
      shapes.push_back(shape);
    }
    // Convert the facial feature points into lines
    std::vector<image_window::overlay_line> lines =
        render_face_detections(shapes);

    // The last 20 lines are for the mouth. Remove all the other lines from the
    // vector
    //lines.erase(lines.begin() + 44, lines.end());
    //lines.erase(lines.begin(), lines.linesbegin() + 33);
    // lines.erase(lines.begin(), lines.end() - 20);  // mouth

    // Set the bounding box of the mouth.
    // This will be our initial region of interest
    mouth_bbox = GetBoundingBox(lines, frame, lines.size() - 20, lines.size(), rotAngle);
    left_eye = GetBoundingBox(lines, frame, 33, 39, rotAngle);
    right_eye = GetBoundingBox(lines, frame, 39, 45, rotAngle);
    return true;
    //bbox = Rect2d(x1, y1, width, height); //- height / 2, width, height * 2);
  } catch (exception &e) {
    cout << "\nException while detecting face" << endl;
    cout << e.what() << endl;
    mouth_bbox = cv::Rect2d(-1, -1, -1, -1); // failed face detection
  }
}

cv::Point2f operator*(cv::Mat M, const cv::Point2f& p)
{ 
    cv::Mat_<double> src(3/*rows*/,1 /* cols */); 

    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=1.0; 

    cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA 
    return cv::Point2f(dst(0,0),dst(1,0)); 
} 

cv::Rect2d GetBoundingBox(std::vector<image_window::overlay_line> &lines, cv::Mat frame, int start,
                      int end, float rotAngle) {
  // Find the bounding box of the mouth by looping over the lines
  // and computing the min and max x and y pixel coordinates
  int x1 = frame.cols, y1 = frame.rows, x2 = -1, y2 = -1;
  for (auto it = lines.begin() + start; it != lines.begin() + end; it++) {
    int max_x =
        (*it).p1.x() > (*it).p2.x() ? (*it).p1.x() : (*it).p2.x();
    int max_y =
        (*it).p1.y() > (*it).p2.y() ? (*it).p1.y() : (*it).p2.y();
    int min_x =
        (*it).p1.x() < (*it).p2.x() ? (*it).p1.x() : (*it).p2.x();
    int min_y =
        (*it).p1.y() < (*it).p2.y() ? (*it).p1.y() : (*it).p2.y();
    if (max_x > x2)
      x2 = max_x;
    if (min_x < x1)
      x1 = min_x;

    if (max_y > y2)
      y2 = max_y;
    if (min_y < y1)
      y1 = min_y;
  }

  cv::Point2f topLeft(x1 * FACE_DOWNSAMPLE_RATIO, y1 * FACE_DOWNSAMPLE_RATIO);
  cv::Point2f botRight(x2 * FACE_DOWNSAMPLE_RATIO, y2 * FACE_DOWNSAMPLE_RATIO);

  cv::Point2f center(imSmall.cols / 2.0, imSmall.rows / 2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, -1.0 * rotAngle, 1.0);

  cv::Point2f rotTopLeft = rot * topLeft;
  cv::Point2f rotBotRight = rot * botRight;

  // multiply 

  x1 = rotTopLeft.x;
  y1 = rotTopLeft.y;
  x2 = rotBotRight.x;
  y2 = rotBotRight.y;

  return cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
}

// (x1,y1) old position of left eye. (x2,y2) old position of right eye.
void findStomionPoint(int x1, int y1, int x2, int y2) {
  // old distance between eyes
  float d1 = hypot(x2 - x1, y2 - y1);
  // new distance between eyes
  float d2 = hypot(right_eye.x - left_eye.x, right_eye.y - left_eye.y);
  // old distance betwwen left eye and stomion
  float d3 = hypot(stomionPointX - x1, stomionPointY - y1);
  // old distance between right eye and stomion
  float d4 = hypot(stomionPointX - x2, stomionPointY - y2);
  // new distance between left eye and stomion
  float d5 = d3 * d2 / d1;

  // angle between right eye, left eye and stomion with left eye as center
  // change angle value to negative at appropriate times
  float angle = acos((d1 * d1 + d3 * d3 - d4 * d4) / (2.0 * d1 * d3));
  // roation of the eyes from horizontal
  float eyeRot = atan((right_eye.x - left_eye.x) / (right_eye.y - left_eye.y));

  stomionPointX = (int) (left_eye.x + d5 * cos(angle + eyeRot));
  stomionPointY = (int) (left_eye.y + d5 * sin(angle + eyeRot));

}

int main(int argc, char **argv) {
  try {
    ros::init(argc, argv, "face_detector");
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

    //image_transport::Subscriber sub_depth = it.subscribe("/camera/aligned_depth_to_color/image_raw", 1, DepthCallBack);
    marker_array_pub =
        nh->advertise<visualization_msgs::MarkerArray>("/face_detector/marker_array", 1);

    ros::spin();
  } catch (serialization_error &e) {
    cout << "Shape predictor model file not found" << endl;
    cout << "Put shape_predictor_68_face_landmarks in models directory" << endl;
    cout << endl << e.what() << endl;
  } catch (exception &e) {
    cout << e.what() << endl;
  }
}
