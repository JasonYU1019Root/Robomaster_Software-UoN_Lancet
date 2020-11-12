// *******************************
// Windmill Detector Realization *
// By JasonYU                    *
// *******************************

#include "opencv2/opencv.hpp"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

//global variables
int flag_color = 0; //0 for blue and 1 for red
cv::Mat frame;

//function headers
cv::Mat Adjust_img(cv::Mat);     //frame preprocessing
void detect(cv::Mat);            //windmill detection
float getDistance(Point, Point); //calc distance between two points

int main()
{
    cv::VideoCapture cap("./windmill.mov");
    if (!cap.isOpened())
    {
        cout << "Error! Video not loaded!\n";
        return -1;
    }

    //cv::Mat frame;
    while (cap.read(frame))
    {
        //img preprocessing
        cv::Mat img_adjusted = Adjust_img(frame);

        //windmill detection
        detect(img_adjusted);

        cv::namedWindow("Windmill Detection", WINDOW_NORMAL);
        cv::imshow("Windmill Detection", frame);
        cv::namedWindow("Adjusted", WINDOW_NORMAL);
        cv::imshow("Adjusted", img_adjusted);
        cv::waitKey(1);
    }

    //release all objects and resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

cv::Mat Adjust_img(cv::Mat srcImg)
{
    //adjust brightness
    Mat dst_br = Mat::zeros(srcImg.size(), srcImg.type());
    Mat brightnessLut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++)
        brightnessLut.at<uchar>(i) = saturate_cast<uchar>(i - 100);
    LUT(srcImg, brightnessLut, dst_br);
    srcImg = dst_br;

    //split color channel and threshold to binary
    /*vector<cv::Mat> imgChannels;
    cv::split(srcImg, imgChannels);
    if (flag_color)
        midImg = imgChannels.at(2) - imgChannels.at(0); //red-blue
    else
    {
        midImg = imgChannels.at(0) - imgChannels.at(2); //blue-red
    }*/
    cv::Mat midImg;
    /*cvtColor(srcImg, midImg, COLOR_BGR2HSV);
    inRange(midImg, Scalar(0, 0, 46), Scalar(180, 43, 220), midImg);*/
    cvtColor(srcImg, midImg, COLOR_BGR2GRAY);
    cv::threshold(midImg, midImg, 80, 255, THRESH_BINARY);

    //dilate to increase the white area
    int structElementSize = 2;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * structElementSize + 1, 2 * structElementSize + 1), Point(structElementSize, structElementSize));
    dilate(midImg, midImg, element);

    //morph_close to reduce holes in white area
    structElementSize = 3;
    element = getStructuringElement(MORPH_RECT, Size(2 * structElementSize + 1, 2 * structElementSize + 1), Point(structElementSize, structElementSize));
    morphologyEx(midImg, midImg, MORPH_CLOSE, element);

    //floodfill
    floodFill(midImg, Point(5, 50), Scalar(255), 0, FLOODFILL_FIXED_RANGE);
    threshold(midImg, midImg, 80, 255, THRESH_BINARY_INV);

    cv::Mat dstImg = midImg;
    return dstImg;
}

void detect(cv::Mat srcImg)
{
    vector<vector<Point>> contours;
    findContours(srcImg, contours, RETR_LIST, CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        vector<Point> points;
        double area = contourArea(contours[i]);
        //screen
        if (area < 1000 || 1e4 < area)
            continue;
        //drawContours(frame, contours, static_cast<int>(i), Scalar(0, 255, 255), 2);

        points = contours[i];
        RotatedRect rrect = fitEllipse(points);
        cv::Point2f *vertices = new cv::Point2f[4];
        rrect.points(vertices);

        float aim = rrect.size.height / rrect.size.width;
        if (aim > 1.7 && aim < 2.6)
        {
            for (int j = 0; j < 4; j++)
            {
                line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 255), 4);
            }

            float middle = 100000;
            for (size_t j = 1; j < contours.size(); j++)
            {
                vector<Point> pointsA;
                double area = contourArea(contours[j]);
                if (area < 1000 || 1e4 < area)
                    continue;
                pointsA = contours[j];
                RotatedRect rrectA = fitEllipse(pointsA);
                float aimA = rrectA.size.height / rrectA.size.width;
                if (aimA > 3.0)
                {
                    float distance = getDistance(rrect.center, rrectA.center);
                    if (middle > distance)
                        middle = distance;
                }
            }
            //test
            //cout << middle << endl;
            //test
            if (middle > 200) //this value was 2 b determind by the size of the img and the distance from the obj
            {
                circle(frame, Point(rrect.center.x, rrect.center.y), 15, Scalar(0, 0, 255), 4);
            }
        }
    }

    return;
}

float getDistance(Point pt1, Point pt2)
{
    float distance;
    distance = powf((pt1.x - pt2.x), 2) + powf((pt1.y - pt2.y), 2);
    distance = sqrtf(distance);
    return distance;
}
