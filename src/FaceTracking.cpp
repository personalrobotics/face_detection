
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

#define THRESHOLD 50

using namespace dlib;
using namespace cv;
using namespace std;

// This program takes in a facial landmark file and a video file to
// process and displays the video file that tracks the mouth of a person.
int main(int argc, char** argv) {
    
    if (argc != 3)
    {
        cerr << "Usage: ";
        cerr << argv[0] << " shape_predictor_68_face_landmarks.dat video_file" << endl;
        return EXIT_FAILURE;
    }
    
    // Initialize KCF tracker
    Ptr<Tracker> tracker = TrackerKCF::create();
    
    // Read video from the specified video file
    VideoCapture video(argv[2]);
    
    // Exit if video can't be opened
    if(!video.isOpened())
    {
        cerr << "Could not read video file" << endl;
        return EXIT_FAILURE;
    }
    
    // Read first frame
    Mat frame;
    // loop until the first frame is read successfully
    // or until the program has attempted to read for
    // a certain THRESHOLD amount of times
    int counter = 0;
    while(!video.read(frame)) {
        counter++;
        if (counter > THRESHOLD) {
            cerr << "Could not read the first frame" << endl;
            return EXIT_FAILURE;
        }
    }
    
    Rect2d bbox;  // Initial region of interest. Will be set later after detecing the mouth
    try
    {
        // Get dlib's face detector
        frontal_face_detector detector = get_frontal_face_detector();
        // Initialize shape predictor with the provided facial landmarks file.
        shape_predictor sp;
        deserialize(argv[1]) >> sp;
        
        // convert OpenCV's MAT to dlib's array2d
        dlib::array2d<dlib::bgr_pixel> img;
        dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(frame));
        
        // Dectect the faces
        std::vector<dlib::rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;
        if (dets.size() == 0) {
            cerr << "No faces detected" << endl;
            return EXIT_FAILURE;
        } else if (dets.size() != 1) {
            cerr << "Mutiple faces detected. Cannot track multiple faces" << endl;
            return EXIT_FAILURE;
        }
        
        std::vector<full_object_detection> shapes;
        // Add the facial features for all the detected faces into a vector
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            shapes.push_back(shape);
        }
        
        // Convert the facial feature points into lines
        std::vector<image_window::overlay_line> lines = render_face_detections(shapes);
        
        // The last 20 lines are for the mouth. Remove all the other lines from the vector
        lines.erase(lines.begin(), lines.end() - 20);
        
        // Find the bounding box of the mouth by looping over the lines
        // and computing the min and max x and y pixel coordinates
        int x1 = frame.cols, y1 = frame.rows, x2 = -1, y2 = -1;
        for (int i = 0; i < lines.size(); i++) {
            int max_x = lines[i].p1.x() > lines[i].p2.x() ? lines[i].p1.x() : lines[i].p2.x();
            int max_y = lines[i].p1.y() > lines[i].p2.y() ? lines[i].p1.y() : lines[i].p2.y();
            int min_x = lines[i].p1.x() < lines[i].p2.x() ? lines[i].p1.x() : lines[i].p2.x();
            int min_y = lines[i].p1.y() < lines[i].p2.y() ? lines[i].p1.y() : lines[i].p2.y();
            if (max_x > x2) x2 = max_x;
            if (min_x < x1) x1 = min_x;
            
            if (max_y > y2) y2 = max_y;
            if (min_y < y1) y1 = min_y;
        }
        
        // Set the bounding box of the mouth.
        // This will be our initial region of interest
        int width = x2 - x1;
        int height = y2 - y1;
        bbox = Rect2d(x1, y1 - height / 2, width, height * 2);
    }
    catch (exception& e)
    {
        cout << "\nException while detecting face" << endl;
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    imshow("Tracking", frame);
    tracker->init(frame, bbox);
    
    // Keep reading until the end of video
    while(video.read(frame))
    {
        // Update the tracking result
        bool success = tracker->update(frame, bbox);
        
        if (success)
        {
            // Tracking success : Draw the bounding box around the tracked mouth
            cv::rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure
            putText(frame, "Failed tracking the mouth", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
        
        // Display frame.
        imshow("Tracking", frame);
        
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        
    }
}



