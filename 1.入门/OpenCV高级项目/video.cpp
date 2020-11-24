#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"

int main() {
    putenv((char *) "DISPLAY=windows:0");
    
    cv::VideoCapture cap = cv::VideoCapture();
    cap.open("../vtest.avi");
    if (cap.isOpened()) {
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                break;
            }
            cv::imshow("video", frame);
            int keyCode = cv::waitKey(100);
            if (keyCode == (int)'q') {
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    }

    return 0;
}