#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

//for storing all parameters for armor detector
struct ArmorParam
{
	float Image_bright;
	bool Armor_color;
	float Light_area_min;
	float Light_angle;
	float Light_aspect_ratio;
	float Light_crown;
	float Light_contour_angle;
	float Light_contour_len;
	float Armor_angle_min;
	float Armor_ratio_min;
	float Armor_ratio_max;
	float Armor_xDiff;
	float Armor_yDiff;
	//for initializing param
	ArmorParam()
	{
		Image_bright = -100;
		Armor_color = 0; //0 for blue and 1 for red
		Light_area_min = 200;
		Light_angle = 5;
		Light_aspect_ratio = 0.7;
		Light_crown = 0.5;
		Light_contour_angle = 4.2;
		Light_contour_len = 0.5;
		Armor_angle_min = 5;
		Armor_ratio_min = 1.0;
		Armor_ratio_max = 5.0;
		Armor_xDiff = 0.5;
		Armor_yDiff = 2.0;
	}
};

//global variables
ArmorParam _Armor;
cv::Mat frame;

//function headers
//frame adjustment
cv::Mat Adjust_Brightness(cv::Mat);
cv::Mat Adjust_Grayscale(cv::Mat);
cv::Mat Adjust_AvgFilter(cv::Mat);
cv::Mat Adjust_Binary(cv::Mat);
cv::Mat Adjust_Dilate(cv::Mat);
cv::Mat Adjust_img(cv::Mat); //for integrating all image adjustments
//detection
cv::Mat detect(cv::Mat);
cv::RotatedRect& adjustRec(cv::RotatedRect&);
cv::Mat drawArmor(cv::RotatedRect, int);

int main()
{
	frame = cv::imread("./target2.jpg");
	if (frame.empty())
	{
		cout << "Error! Image not loaded.\n";
		return -1;
	}
	cout << "Image loaded.\n";

	/*cv::VideoCapture capture("./TargetVideo.mov");
	if (!capture.isOpened())
	{
		cout << "Error! Video not loaded.\n";
		return -1;
	}
	cout << "Video loaded.\n";*/

	//cv::Mat frame;
	//while (capture.read(frame))
	{
		//frame adjustment
		cv::Mat img_adjusted = Adjust_img(frame);

		//detection
		cv::Mat frame_dst = detect(img_adjusted);

		cv::namedWindow("Armor Detection", WINDOW_NORMAL);
		cv::imshow("Armor Detection", frame_dst);
		cv::waitKey();
	}

	//release all objects and resources
	//capture.release();
	cv::destroyAllWindows();

	return 0;
}

cv::Mat Adjust_Brightness(cv::Mat srcImg)
{
	cv::Mat dstImg = cv::Mat::zeros(srcImg.size(), srcImg.type());
	cv::Mat BrightnessLut(1, 256, CV_8UC1);
	for (int i = 0; i < 256; i++)
		BrightnessLut.at<uchar>(i) = cv::saturate_cast<uchar>(i + _Armor.Image_bright);
	cv::LUT(srcImg, BrightnessLut, dstImg);
	return dstImg;
}

cv::Mat Adjust_Grayscale(cv::Mat srcImg)
{
	cv::Mat dstImg, channels[3];
	cv::split(srcImg, channels); //split color channels
	//preprocessing: delete own armor color
	if (_Armor.Armor_color)
		dstImg = (channels[2] - channels[0]) * 2; //get red-blue img
	else
		dstImg = channels[0] - channels[2]; //get blue-red img
	return dstImg;
}

cv::Mat Adjust_AvgFilter(cv::Mat srcImg)
{
	cv::Mat dstImg;
	cv::blur(srcImg, dstImg, cv::Size(1, 3));
	return dstImg;
}

cv::Mat Adjust_Binary(cv::Mat srcImg)
{
	cv::Mat dstImg;
	if (!_Armor.Armor_color)threshold(srcImg, dstImg, 130, 255, THRESH_BINARY);
	else threshold(srcImg, dstImg, 230, 255, THRESH_BINARY);
	return dstImg;
}

cv::Mat Adjust_Dilate(cv::Mat srcImg)
{
	cv::Mat dstImg;
	//cv::Mat kernel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
	int structElementSize = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * structElementSize + 1, 2 * structElementSize + 1), Point(structElementSize, structElementSize));
	cv::dilate(srcImg, dstImg, element);
	return dstImg;
}

cv::Mat Adjust_img(cv::Mat srcImg) //for integrating all image adjustments
{
	cv::Mat frame_bright = Adjust_Brightness(srcImg);
	cv::Mat frame_gray = Adjust_Grayscale(frame_bright);
	//cv::Mat frame_avg = Adjust_AvgFilter(frame_gray);
	cv::Mat frame_bin = Adjust_Binary(frame_gray);
	cv::Mat frame_dilate = Adjust_Dilate(frame_bin);

	cv::Mat dstImg = frame_dilate;
	cv::namedWindow("test_processed", WINDOW_NORMAL);
	cv::imshow("test_processed", dstImg);
	cv::waitKey();
	return dstImg;
}

cv::Mat detect(cv::Mat srcImg)
{
	//finding contours
	vector<vector<cv::Point>> Light_Contour; //for storing detected contours
	Light_Contour.clear();
	cv::findContours(srcImg, Light_Contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<cv::RotatedRect> vContour; //for storing the screened contours
	vContour.clear();

	//screening detected contours
	for (int i = 0; i < Light_Contour.size(); i++)
	{
		//calc the area of the contour
		float Light_Contour_Area = cv::contourArea(Light_Contour[i]);
		//ignoring relatively too small contours & restriction of fitEllipse
		if (Light_Contour_Area < _Armor.Light_area_min || Light_Contour[i].size() <= 5)
			continue;
		cv::RotatedRect Light_Rec = cv::fitEllipse(Light_Contour[i]);
		Light_Rec = adjustRec(Light_Rec);
		if (Light_Rec.angle > _Armor.Light_angle)
			continue;
		//dimension ratio and area ratio restriction
		if (Light_Rec.size.width / Light_Rec.size.height > _Armor.Light_aspect_ratio || Light_Contour_Area / Light_Rec.size.area() < _Armor.Light_crown)
			continue;
		//resize light contour
		Light_Rec.size.height *= 1.1;
		Light_Rec.size.width *= 1.1;
		vContour.push_back(Light_Rec);
		drawArmor(Light_Rec, 1);
	}

	vector<cv::RotatedRect> vRec; //for storing the screened armors
	vRec.clear();

	//find matching light contours
	for (int i = 0; i < vContour.size(); i++)
	{
		for (int j = i + 1; j < vContour.size(); j++)
		{
			float Contour_angle = abs(vContour[i].angle - vContour[j].angle);
			if (Contour_angle >= _Armor.Light_contour_angle)
				continue;
			//ratio of length diff
			float Contour_len1 = abs(vContour[i].size.height - vContour[j].size.height) / max(vContour[i].size.height, vContour[j].size.height);
			//ratio of width diff
			float Contour_len2 = abs(vContour[i].size.width - vContour[j].size.width) / max(vContour[i].size.width, vContour[j].size.width);
			if (Contour_len1 > _Armor.Light_contour_len || Contour_len2 > _Armor.Light_contour_len)
				continue;

			//screening detected armors
			cv::RotatedRect Rect;
			Rect.center.x = (vContour[i].center.x + vContour[j].center.x) / 2;
			Rect.center.y = (vContour[i].center.y + vContour[j].center.y) / 2;
			Rect.angle = (vContour[i].angle + vContour[j].angle) / 2;
			float nh, nw, xDiff, yDiff;
			nh = (vContour[i].size.height + vContour[j].size.height) / 2; //height
			nw = sqrt((vContour[i].center.x - vContour[j].center.x) * (vContour[i].center.x - vContour[j].center.x) + (vContour[i].center.y - vContour[j].center.y) * (vContour[i].center.y - vContour[j].center.y));
			float ratio = nw / nh; //detected armor dimension ratio
			xDiff = abs(vContour[i].center.x - vContour[j].center.x) / nh;
			yDiff = abs(vContour[i].center.y - vContour[j].center.y) / nh;
			Rect = adjustRec(Rect);
			if (Rect.angle > _Armor.Armor_angle_min || ratio < _Armor.Armor_ratio_min || ratio > _Armor.Armor_ratio_max || xDiff < _Armor.Armor_xDiff || yDiff > _Armor.Armor_yDiff)
				continue;
			Rect.size.height = nh;
			Rect.size.width = nw;
			vRec.push_back(Rect);
		}
	}

	//draw rectangles to highlight detected armor
	cv::Mat dstImg;
	if (vRec.empty())
	{
		dstImg = frame;
		cout << "No Armor Detected.\n";
	}
	else
	{
		for (int i = 0; i < vRec.size(); i++)
			dstImg = drawArmor(vRec[i], 0);
		cout << "Armor Detected!\n";
	}

	return dstImg;
}

cv::RotatedRect& adjustRec(cv::RotatedRect& rec)
{
	using std::swap;

	float& width = rec.size.width;
	float& height = rec.size.height;
	float& angle = rec.angle;

	while (angle >= 90.0)
		angle -= 180.0;
	while (angle < -90.0)
		angle += 180.0;

	if (angle >= 45.0)
	{
		swap(width, height);
		angle -= 90.0;
	}
	else if (angle <= -45.0)
	{
		swap(width, height);
		angle += 90.0;
	}

	return rec;
}

cv::Mat drawArmor(cv::RotatedRect rec, int color)
{
	cv::Mat dstImg = frame;
	cv::Point2f p[4];
	rec.points(p);
	if (!color)
	{
		line(dstImg, p[0], p[1], Scalar(0, 255, 255), 3, LINE_AA, 0);
		line(dstImg, p[1], p[2], Scalar(0, 255, 255), 3, LINE_AA, 0);
		line(dstImg, p[2], p[3], Scalar(0, 255, 255), 3, LINE_AA, 0);
		line(dstImg, p[3], p[0], Scalar(0, 255, 255), 3, LINE_AA, 0);
	}
	else
	{
		line(dstImg, p[0], p[1], Scalar(255, 255, 0), 3, LINE_AA, 0);
		line(dstImg, p[1], p[2], Scalar(255, 255, 0), 3, LINE_AA, 0);
		line(dstImg, p[2], p[3], Scalar(255, 255, 0), 3, LINE_AA, 0);
		line(dstImg, p[3], p[0], Scalar(255, 255, 0), 3, LINE_AA, 0);
	}
	return dstImg;
}
