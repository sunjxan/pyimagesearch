#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace std;

int main() {
    putenv((char *)"DISPLAY=windows:0");

    cv::Mat image = cv::imread("../jp.png");
    int w = image.cols, h = image.rows, d = image.channels();
    printf("width=%d, height=%d, depth=%d\n", w, h, d);

    cv::Vec3b BGR = image.at<cv::Vec3b>(100, 50);
    printf("R=%d, G=%d, B=%d\n", BGR[2], BGR[1], BGR[0]);

    cv::imshow("Image", image);
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat roi = image(cv::Range(60, 160), cv::Range(320, 420));
    cv::imshow("ROI", roi);
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(200, 200));
    cv::imshow("Resized", resized);
    cv::waitKey();
    cv::destroyAllWindows();

    float ratio = .5;
    cv::resize(image, resized, cv::Size(round(w * ratio), round(h * ratio)));
    cv::imshow("Aspect Ratio Resize", resized);
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat rotated;
    cv::Point center = cv::Point(round(w / 2), round(h / 2));
    cv::Mat M = cv::getRotationMatrix2D(center, -45, 1.0);
    cv::warpAffine(image, rotated, M, cv::Size(w, h));
    cv::imshow("Rotation", rotated);
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat pts = (cv::Mat_<double>(4, 2) << 0, 0, w - 1, 0, 0, h - 1, w - 1, h - 1);
    cv::Mat M_ = M(cv::Range(0, 2), cv::Range(0, 2));
    pts = (M_ * pts.t()).t();
    double minX = pts.at<double>(0, 0),
        maxX = pts.at<double>(0, 0),
        minY = pts.at<double>(0, 1),
        maxY = pts.at<double>(0, 1);
    for (int i = 1; i < pts.rows; ++i) {
        double x = pts.at<double>(i, 0),
            y = pts.at<double>(i, 1);
        if (x < minX) {
            minX = x;
        } else if (x > maxX) {
            maxX = x;
        }
        if (y < minY) {
            minY = y;
        } else if (y > maxY) {
            maxY = y;
        }
    }
    M.at<double>(0, 2) = -minX;
    M.at<double>(1, 2) = -minY;
    int nW = ceil(maxX - minX + 1);
    int nH = ceil(maxY - minY + 1);
    cv::warpAffine(image, rotated, M, cv::Size(nW, nH));
    cv::imshow("Bound Rotation", rotated);
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(11, 11), 0);
    cv::imshow("Blurred", blurred);
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat output = image.clone();
    cv::rectangle(output, cv::Point(320, 60), cv::Point(420, 160), cv::Scalar(0, 0, 255), 2);
    cv::imshow("Rectangle", output);
    cv::waitKey();
    cv::destroyAllWindows();

    output = image.clone();
    cv:circle(output, cv::Point(300, 150), 20, cv::Scalar(255, 0, 0), -1);
    cv::imshow("Circle", output);
    cv::waitKey();
    cv::destroyAllWindows();

    output = image.clone();
    cv::line(output, cv::Point(60, 20), cv::Point(400, 200), cv::Scalar(0, 0, 255), 5);
    cv::imshow("Line", output);
    cv::waitKey();
    cv::destroyAllWindows();

    output = image.clone();
    cv::putText(output, "OpenCV + Jurassic Park!!!", cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX, .7, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Text", output);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}