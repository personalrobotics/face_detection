#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <time.h>

#define THRESHOLD 50

using namespace dlib;
using namespace cv;
using namespace std;

Rect2d GetBoundingBox(std::vector<image_window::overlay_line> &lines, Mat frame, int start,
                      int end);
void RunFaceDetection(const Mat &frame, frontal_face_detector &detector, const shape_predictor &sp, 
                       Rect2d &mouth_bbox, Rect2d &left_eye, Rect2d &right_eye);

// This program takes in a facial landmark file and a video file to
// process and displays the video file that tracks the mouth of a person.
int main(int argc, char **argv) {

  if (argc != 3) {
    cerr << "Usage: ";
    cerr << argv[0] << " shape_predictor_68_face_landmarks.dat video_file"
         << endl;
    return EXIT_FAILURE;
  }

  // Initialize KCF tracker
  Ptr<Tracker> tracker1 = TrackerKCF::create();
  Ptr<Tracker> tracker2 = TrackerKCF::create();
  Ptr<Tracker> tracker3 = TrackerKCF::create();

  // Read video from the specified video file
  VideoCapture video;
  if (argv[2][0] == '0')
    video = VideoCapture(0);
  else
    video = VideoCapture(argv[2]);

  // Exit if video can't be opened
  if (!video.isOpened()) {
    cerr << "Could not read video file" << endl;
    return EXIT_FAILURE;
  }

  // Read first frame
  Mat frame;
  // loop until the first frame is read successfully
  // or until the program has attempted to read for
  // a certain THRESHOLD amount of times
  int counter = 0;
  while (!video.read(frame)) {
    counter++;
    if (counter > THRESHOLD) {
      cerr << "Could not read the first frame" << endl;
      return EXIT_FAILURE;
    }
  }

  Rect2d mouth_bbox; // Initial region of interest. Will be set later after detetcing the mouth
  Rect2d left_eye;
  Rect2d right_eye;

  // Get dlib's face detector
  frontal_face_detector detector = get_frontal_face_detector();
  // Initialize shape predictor with the provided facial landmarks file.
  shape_predictor sp;
  deserialize(argv[1]) >> sp;

  RunFaceDetection(frame, detector, sp, mouth_bbox, left_eye, right_eye);

  imshow("Tracking", frame);
  tracker1->init(frame, mouth_bbox);
  tracker2->init(frame, left_eye);
  tracker3->init(frame, right_eye);

  int dx_left = mouth_bbox.x - left_eye.x;
  int dy_left = mouth_bbox.y - left_eye.y;

  int dx_right = mouth_bbox.x - right_eye.x;
  int dy_right = mouth_bbox.y - right_eye.y;

  // Keep reading until the end of video
  while (video.read(frame)) {
    // Update the tracking result
    bool success = tracker1->update(frame, mouth_bbox);
    bool left = tracker2->update(frame, left_eye);
    bool right = tracker3->update(frame, right_eye);

    if (success) {
      // Mouth Tracking success : Draw the bounding box around the tracked mouth
      cv::rectangle(frame, mouth_bbox, Scalar(255, 0, 0), 2, 1);
      putText(frame, "Mouth tracking success", Point(100, 80),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
    } else {
      // Mouth Tracking failure
      putText(frame, "Failed tracking the mouth", Point(100, 80),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
      if(left && right)
      {
        putText(frame, "Estimating mouth from the eyes", Point(100, 120),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        cv::rectangle(frame, left_eye, Scalar(255, 0, 0), 2, 1);
        cv::rectangle(frame, right_eye, Scalar(255, 0, 0), 2, 1);
        cv::rectangle(frame, Rect2d(left_eye.x + dx_left, left_eye.y + dy_left, mouth_bbox.width, mouth_bbox.height), Scalar(255, 0, 0), 2, 1);
      }
      else if (!left) {
        putText(frame, "Failed tracking the left eye", Point(100, 120),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        putText(frame, "Estimating mouth from the right eye", Point(100, 160),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        cv::rectangle(frame, right_eye, Scalar(255, 0, 0), 2, 1);
        cv::rectangle(frame, Rect2d(right_eye.x + dx_right, right_eye.y + dy_right, mouth_bbox.width, mouth_bbox.height), Scalar(255, 0, 0), 2, 1);
      }
      else if (!right) {
        putText(frame, "Failed tracking the right eye", Point(100, 120),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        putText(frame, "Estimating mouth from the left eye", Point(100, 160),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        cv::rectangle(frame, left_eye, Scalar(255, 0, 0), 2, 1);
        cv::rectangle(frame, Rect2d(left_eye.x + dx_left, left_eye.y + dy_left, mouth_bbox.width, mouth_bbox.height), Scalar(255, 0, 0), 2, 1);
      } else {
        // Both eye tracking failed and mouth tracking also failed
        // Fall back to detection
        putText(frame, "Running Face detection again", Point(100, 120),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        RunFaceDetection(frame, detector, sp, mouth_bbox, left_eye, right_eye);
        if (mouth_bbox.x == -1)
          putText(frame, "Failed Face Detection", Point(100, 160),
              FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        else {
          cv::rectangle(frame, mouth_bbox, Scalar(255, 0, 0), 2, 1);
        }
      }
      
      

      // Estimate position of the mouth
      //double eyeDistance = (right_eye.x + right_eye.width) - left_eye.x;
      //double distanceToPixels = eyeDistance / 131.0;
      //int y = (int)((left_eye.y + left_eye.height / 2) + 70 * distanceToPixels);
      //int x = (int)(left_eye.x + 63 * distanceToPixels);
      //cv::rectangle(frame, Rect2d(x - mouth_bbox.width / 2, y, mouth_bbox.width, mouth_bbox.height), Scalar(255, 0, 0), 2, 1);
    }

    // Display frame.
    imshow("Tracking", frame);

    // Exit if ESC pressed.
    int k = waitKey(1);
    if (k == 27) {
      break;
    }
  }
}

void RunFaceDetection(const Mat &frame, frontal_face_detector &detector, const shape_predictor &sp, 
                       Rect2d &mouth_bbox, Rect2d &left_eye, Rect2d &right_eye)
{
  try {
    // convert OpenCV's MAT to dlib's array2d
    dlib::array2d<dlib::bgr_pixel> img;
    dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(frame));

    // Dectect the faces
    std::vector<dlib::rectangle> dets = detector(img);
    cout << "Number of faces detected: " << dets.size() << endl;
    if (dets.size() == 0) {
      cerr << "No faces detected" << endl;
      mouth_bbox = Rect2d(-1, -1, -1, -1);
      return;
    } else if (dets.size() != 1) {
      cerr << "Mutiple faces detected. Cannot track multiple faces" << endl;
      mouth_bbox = Rect2d(-1, -1, -1, -1);
      return;
    }

    std::vector<full_object_detection> shapes;
    // Add the facial features for all the detected faces into a vector
    for (unsigned long j = 0; j < dets.size(); ++j) {
      full_object_detection shape = sp(img, dets[j]);
      shapes.push_back(shape);
    }

    // Convert the facial feature points into lines
    std::vector<image_window::overlay_line> lines =
        render_face_detections(shapes);

    // The last 20 lines are for the mouth. Remove all the other lines from the
    // vector
    //lines.erase(lines.begin() + 44, lines.end());
    //lines.erase(lines.begin(), lines.begin() + 33);
    // lines.erase(lines.begin(), lines.end() - 20);  // mouth

    // Set the bounding box of the mouth.
    // This will be our initial region of interest
    mouth_bbox = GetBoundingBox(lines, frame, lines.size() - 20, lines.size());
    left_eye = GetBoundingBox(lines, frame, 33, 39);
    right_eye = GetBoundingBox(lines, frame, 39, 45);
    //bbox = Rect2d(x1, y1, width, height); //- height / 2, width, height * 2);
  } catch (exception &e) {
    cout << "\nException while detecting face" << endl;
    cout << e.what() << endl;
    mouth_bbox = Rect2d(-1, -1, -1, -1);
  }
}

Rect2d GetBoundingBox(std::vector<image_window::overlay_line> &lines, Mat frame, int start,
                      int end) {
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
  int width = x2 - x1;
  int height = y2 - y1;
  return Rect2d(x1, y1, width, height);
}
