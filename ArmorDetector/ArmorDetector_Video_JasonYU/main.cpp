#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <string>

#include "./General/opencv_extended.h"
#include "./General/numeric_rm.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

//global variables
Mat frame;

enum ColorChannel
{
	BLUE = 0,
	Green = 1,
	RED = 2
};

enum ObjectType
{
	UNKNOWN_ARMOR = 0,
	SMALL_ARMOR = 1,
	BIG_ARMOR = 2,
	MINI_RUNE = 3,
	GREAT_RUNE = 4
};

enum
{
	WIDTH_GREATER_THAN_HEIGHT,
	ANGLE_TO_UP
};

//for storing all parameters of armor detection
struct ArmorParam
{
	//pre-treatment
	int brightness_threshold;
	int color_threshold;
	float light_color_detect_extend_ratio;

	//filter lights
	float light_min_area;
	float light_max_angle;
	float light_min_size;
	float light_contour_min_solidity;
	float light_max_ratio;

	//filter pairs
	float light_max_angle_diff;
	float light_max_height_diff_ratio;//diff/max(r.length,l.length)
	float light_max_y_diff_ratio;//ydiff/max(r.length,l.length)
	float light_min_x_diff_ratio;

	//filter armor
	float armor_big_armor_ratio;
	float armor_small_armor_ratio;
	float armor_min_aspect_ratio;
	float armor_max_aspect_ratio;

	//other params
	float sight_offset_normalized_base;
	float area_normalized_base;
	int enemy_color;
	int max_track_num = 3000;

	//set default values for all params
	ArmorParam()
	{
		//pre-treatment
		brightness_threshold = 210;
		color_threshold = 40;
		light_color_detect_extend_ratio = 1.1;

		//filter lights
		light_min_area = 10;
		light_max_angle = 45.0;
		light_min_size = 5.0;
		light_contour_min_solidity = 0.5;
		light_max_ratio = 1.0;

		//filter pairs
		light_max_angle_diff = 7.0;//20
		light_max_height_diff_ratio = 0.2;//0.5
		light_max_y_diff_ratio = 2.0;//100
		light_min_x_diff_ratio = 0.5;//100

		//filter armor
		armor_big_armor_ratio = 3.2;
		armor_small_armor_ratio = 2;
		armor_min_aspect_ratio = 1.0;
		armor_max_aspect_ratio = 5.0;

		//other params
		sight_offset_normalized_base = 200;
		area_normalized_base = 100;
		enemy_color = BLUE;
	}
};

//for describing light info including angle level, width, length, score
class LightDescriptor
{
public:
	float width;
	float length;
	cv::Point2f center;
	float angle;
	float area;
public:
	LightDescriptor() {};
	LightDescriptor(const cv::RotatedRect& light)
	{
		width = light.size.width;
		length = light.size.height;
		center = light.center;
		angle = light.angle;
		area = light.size.area();
	}
	const LightDescriptor& operator = (const LightDescriptor& ld)
	{
		this->width = ld.width;
		this->length = ld.length;
		this->center = ld.center;
		this->angle = ld.angle;
		this->area = ld.area;
		return *this;
	}
	//return the light as a cv::RotatedRect object
	cv::RotatedRect rec() const
	{
		return cv::RotatedRect(center, cv::Size2f(width, length), angle);
	}
};

//for describing armor info including max bbox, vertex etc
class ArmorDescriptor
{
public:
	std::array<cv::RotatedRect, 2>lightPairs;//0 left, 1 right
	float sizeScore;//S1 = e^(size)
	float distScore;//S2 = e^(-offset)
	float rotationScore;//S3 = -(ratio^2 + yDiff^2)
	float finalScore;
	std::vector<cv::Point2f>vertex;//four vertex of armor area, light bar area exclued!!
	cv::Mat frontImg;//front img after perspective tf from vertex, 1 channel gray img
	int type;//0 small, 1 big, -1 unknown
public:
	//initialize with all 0
	ArmorDescriptor();
	/*
		brief: calc the rest info (exclude math&final score) of ArmorDescriptor based on: 
			l&r light, part of members in ArmorDetector, and the armortype (save time)
		calls: ArmorDescriptor::getFrontImg()
	*/
	ArmorDescriptor(const LightDescriptor& lLight, const LightDescriptor& rLight, const int armorType, const cv::Mat& srcImg, const float rotationScore, ArmorParam param);
	/*
		brief: empty the object
		called: ArmorDetection._targetArmor
	*/
	void clear()
	{
		rotationScore = 0;
		sizeScore = 0;
		distScore = 0;
		finalScore = 0;
		for (int i = 0; i < 4; i++)
		{
			vertex[i] = cv::Point2f(0, 0);
		}
		type = UNKNOWN_ARMOR;
	}
	/*
		brief: get the front img (perspective tf) of armor (if big, return the middle part)
		inputs: grayImg of roi
		outputs: store the front img to ArmorDescriptor's public
	*/
	void getFrontImg(const cv::Mat& grayImg);
	//return: if the centeral pattern belong to an armor
	bool isArmorPattern() const;
};

//for implementing all functions of armor detector
class ArmorDetector
{
public:
	//flag for the detection result
	enum ArmorFlag
	{
		ARMOR_NO = 0,//not found
		ARMOR_LOST = 1,//lost tracking
		ARMOR_GLOBAL = 2,//armor found globally
		ARMOR_LOCAL = 3//armor found locally (in tracking mode)
	};
public:
	ArmorDetector();
	ArmorDetector(const ArmorParam& armorParam);
	~ArmorDetector() {}
	/*
		brief: initialize armor params
		others: API for client
	*/
	void init(const ArmorParam& armorParam);
	/*
		brief: set the enemy's color
		others: API for client
	*/
	void setEnemyColor(int enemy_color)
	{
		_enemy_color = enemy_color;
		_self_color = enemy_color == BLUE ? RED : BLUE;
	}
	/*
		brief: load image and set tracking roi
		inputs: frame
		others: API for client
	*/
	void loadImg(const cv::Mat& srcImg);
	/*
		brief: core of detection algorithm, include all the main detecion process
		outputs: all the info of detection result
		return: see enum ArmorFlag
		others: API for client
	*/
	int detect();
	/*
		brief: get the vertex of armor
		return: vector of four cv::point2f objects
		notice: order->left-top, right-top, right-bottom, left-bottom
		others: API for client
	*/
	const std::vector<cv::Point2f> getArmorVertex() const;
	/*
		brief: returns the type of the armor
		return: 0 for small armor, 1 for big armor
		others: API for client
	*/
	int getArmorType() const;
private:
	ArmorParam _param;
	int _enemy_color;
	int _self_color;
	cv::Rect _roi;//relative coordinates
	cv::Mat _srcImg;//source img
	cv::Mat _roiImg;//roi from the result of the last frame
	cv::Mat _grayImg;//gray img of roi
	int _trackCnt = 0;
	std::vector<ArmorDescriptor> _armors;
	ArmorDescriptor _targetArmor;//relative coordinates
	int _flag;
	bool _isTracking;
};

/*
	brief: regulate the rotated rect
	inputs: rotated rect, regulation mode
	return: regulated rect
*/
cv::RotatedRect& adjustRec(cv::RotatedRect& rec, const int mode)
{
	using std::swap;

	float& width = rec.size.width;
	float& height = rec.size.height;
	float& angle = rec.angle;

	if (mode == WIDTH_GREATER_THAN_HEIGHT)
	{
		if (width < height)
		{
			swap(width, height);
			angle += 90.0;
		}
	}

	while (angle >= 90.0)angle -= 180.0;
	while (angle < -90.0)angle += 180.0;

	if (mode == ANGLE_TO_UP)
	{
		if (angle >= 45.0)
		{
			swap(width, height);
			angle -= 90.0;
		}
		else if (angle < -45.0)
		{
			swap(width, height);
			angle += 90.0;
		}
	}

	return rec;
}

ArmorDescriptor::ArmorDescriptor()
{
	rotationScore = 0;
	sizeScore = 0;
	vertex.resize(4);
	for (int i = 0; i < 4; i++)
	{
		vertex[i] = cv::Point2f(0, 0);
	}
	type = UNKNOWN_ARMOR;
}

ArmorDescriptor::ArmorDescriptor(const LightDescriptor& lLight, const LightDescriptor& rLight, const int armorType, const cv::Mat& grayImg, float rotaScore, ArmorParam _param)
{
	//handle two lights
	lightPairs[0] = lLight.rec();
	lightPairs[1] = rLight.rec();

	cv::Size exLSize(int(lightPairs[0].size.width), int(lightPairs[0].size.height * 2));
	cv::Size exRSize(int(lightPairs[1].size.width), int(lightPairs[1].size.height * 2));
	cv::RotatedRect exLLight(lightPairs[0].center, exLSize, lightPairs[0].angle);
	cv::RotatedRect exRLight(lightPairs[1].center, exRSize, lightPairs[1].angle);

	cv::Point2f pts_l[4];
	exLLight.points(pts_l);
	cv::Point2f upper_l = pts_l[2];
	cv::Point2f lower_l = pts_l[3];

	cv::Point2f pts_r[4];
	exRLight.points(pts_r);
	cv::Point2f upper_r = pts_r[2];
	cv::Point2f lower_r = pts_r[3];

	vertex.resize(4);
	vertex[0] = upper_l;
	vertex[1] = upper_r;
	vertex[2] = lower_r;
	vertex[3] = lower_l;

	//set armor type
	type = armorType;

	//get front view
	getFrontImg(grayImg);
	rotationScore = rotaScore;

	//calc the size score
	float normalized_area = contourArea(vertex) / _param.area_normalized_base;
	sizeScore = exp(normalized_area);

	//calc teh dist score
	Point2f srcImgCenter(grayImg.cols / 2, grayImg.rows / 2);
	float sightOffset = cvex::distance(srcImgCenter, cvex::crossPointOf(array<Point2f, 2>{vertex[0], vertex[2]}, array<Point2f, 2>{vertex[1], vertex[3]}));
	distScore = exp(-sightOffset / _param.sight_offset_normalized_base);
}

void ArmorDescriptor::getFrontImg(const Mat& grayImg)
{
	using cvex::distance;

	const Point2f&
		tl = vertex[0],
		tr = vertex[1],
		br = vertex[2],
		bl = vertex[3];

	int width, height;
	if (type == BIG_ARMOR)
	{
		width = 92;
		height = 50;
	}
	else
	{
		width = 50;
		height = 50;
	}

	Point2f src[4]{ Vec2f(tl),Vec2f(tr),Vec2f(br),Vec2f(bl) };
	Point2f dst[4]{ Point2f(0.0,0.0),Point2f(width,0.0),Point2f(width,height),Point2f(0.0,height) };
	const Mat perspMat = getPerspectiveTransform(src, dst);
	cv::warpPerspective(grayImg, frontImg, perspMat, Size(width, height));
}

ArmorDetector::ArmorDetector()
{
	_flag = ARMOR_NO;
	_roi = Rect(cv::Point(0, 0), _srcImg.size());
	_isTracking = false;
}

ArmorDetector::ArmorDetector(const ArmorParam& armorParam)
{
	_param = armorParam;
	_flag = ARMOR_NO;
	_roi = Rect(cv::Point(0, 0), _srcImg.size());
	_isTracking = false;
}

void ArmorDetector::init(const ArmorParam& armorParam)
{
	_param = armorParam;
}

void ArmorDetector::loadImg(const cv::Mat& srcImg)
{
	_srcImg = srcImg;

	Rect imgBound = Rect(cv::Point(0, 0), _srcImg.size());

	if (_flag == ARMOR_LOCAL && _trackCnt != _param.max_track_num)
	{
		cv::Rect bRect = boundingRect(_targetArmor.vertex) + _roi.tl();
		bRect = cvex::scaleRect(bRect, Vec2f(3, 2));//use the center as an anchor to zoom in (scale factor 2)
		_roi = bRect & imgBound;
		_roiImg = _srcImg(_roi).clone();
	}
	else
	{
		_roi = imgBound;
		_roiImg = _srcImg.clone();
		_trackCnt = 0;
	}
}

int ArmorDetector::detect()
{
	//detect lights and build light bars' descriptors
	_armors.clear();
	std::vector<LightDescriptor> lightInfos;
	{
		//pre-treatment
		cv::Mat binBrightImg;
		cvtColor(_roiImg, _grayImg, COLOR_BGR2GRAY, 1);
		cv::threshold(_grayImg, binBrightImg, _param.brightness_threshold, 255, cv::THRESH_BINARY);

		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		dilate(binBrightImg, binBrightImg, element);

		//find and filter light bars
		vector<vector<Point>> lightContours;
		cv::findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		for (const auto& contour : lightContours)
		{
			float lightContourArea = contourArea(contour);
			if (contour.size() <= 5 || lightContourArea < _param.light_min_area)continue;

			RotatedRect lightRec = fitEllipse(contour);
			adjustRec(lightRec, ANGLE_TO_UP);

			if (lightRec.size.width / lightRec.size.height > _param.light_max_ratio ||
				lightContourArea / lightRec.size.area() < _param.light_contour_min_solidity)continue;

			lightRec.size.width *= _param.light_color_detect_extend_ratio;
			lightRec.size.height *= _param.light_color_detect_extend_ratio;
			Rect lightRect = lightRec.boundingRect();
			const Rect srcBound(Point(0, 0), _roiImg.size());
			lightRect &= srcBound;
			Mat lightImg = _roiImg(lightRect);
			Mat lightMask = Mat::zeros(lightRect.size(), CV_8UC1);
			Point2f lightVertexArray[4];
			lightRec.points(lightVertexArray);
			std::vector<Point> lightVertex;
			for (int i = 0; i < 4; i++)
			{
				lightVertex.emplace_back(Point(lightVertexArray[i].x - lightRect.tl().x,
											   lightVertexArray[i].y - lightRect.tl().y));
			}
			fillConvexPoly(lightMask, lightVertex, 255);

			if (lightImg.size().area() <= 0 || lightMask.size().area() <= 0)continue;
			cv::dilate(lightMask, lightMask, element);
			const Scalar meanVal = mean(lightImg, lightMask);

			if (((_enemy_color == BLUE) && (meanVal[BLUE] - meanVal[RED] > 20.0)) || (_enemy_color == RED && (meanVal[RED] - meanVal[BLUE] > 20.0)))
			{
				lightInfos.push_back(LightDescriptor(lightRec));
			}
		}

		if (lightInfos.empty())return _flag = ARMOR_NO;
	}
	
	//find and filter light bar pairs
	{
		sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2)
			{
				return ld1.center.x < ld2.center.x;
			});
		vector<int>minRightIndices(lightInfos.size(), -1);
		for (size_t i = 0; i < lightInfos.size(); i++)
			for (size_t j = i + 1; (j < lightInfos.size()); j++)
			{
				const LightDescriptor& leftLight = lightInfos[i];
				const LightDescriptor& rightLight = lightInfos[j];

				/*
					works for 2-3 meters situation
					morphologically similar: //parallel
						//similar height
				*/
				float angleDiff = abs(leftLight.angle - rightLight.angle);
				float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
				if (angleDiff > _param.light_max_angle_diff || LenDiff_ratio > _param.light_max_height_diff_ratio)continue;

				/*
					proper location: //y value of light bar close enough
						//ratio of height and width is proper
				*/
				float dis = cvex::distance(leftLight.center, rightLight.center);
				float meanLen = (leftLight.length + rightLight.length) / 2;
				float yDiff = abs(leftLight.center.y - rightLight.center.y);
				float yDiff_ratio = yDiff / meanLen;
				float xDiff = abs(leftLight.center.x - rightLight.center.x);
				float xDiff_ratio = xDiff / meanLen;
				float ratio = dis / meanLen;
				if (yDiff_ratio > _param.light_max_y_diff_ratio || xDiff_ratio < _param.light_min_x_diff_ratio || ratio > _param.armor_max_aspect_ratio || ratio < _param.armor_min_aspect_ratio)continue;

				//calc pairs' info
				int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR;
				//calc the rotation score
				float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio - ratio, float(0)) : max(_param.armor_small_armor_ratio - ratio, float(0));
				float yOff = yDiff / meanLen;
				float rotationScore = -(ratiOff * ratiOff + yOff * yOff);

				ArmorDescriptor armor(leftLight, rightLight, armorType, _grayImg, rotationScore, _param);
				_armors.emplace_back(armor);
				break;
			}

		if (_armors.empty())return _flag = ARMOR_NO;
	}

	//delete the fake armors
	_armors.erase(remove_if(_armors.begin(), _armors.end(), [](ArmorDescriptor& i)
		{
			return !(i.isArmorPattern());
		}), _armors.end());

	if (_armors.empty())
	{
		_targetArmor.clear();

		if (_flag == ARMOR_LOCAL)
		{
			cout << "Tracking lost" << endl;
			return _flag = ARMOR_LOST;
		}
		else
		{
			cout << "No armor pattern detected." << endl;
			return _flag = ARMOR_NO;
		}
	}

	//calc the final score
	for (auto& armor : _armors)
	{
		armor.finalScore = armor.sizeScore + armor.distScore + armor.rotationScore;
	}

	//choose the one with highest score, store it on _targetArmor
	std::sort(_armors.begin(), _armors.end(), [](const ArmorDescriptor& a, const ArmorDescriptor& b)
		{
			return a.finalScore > b.finalScore;
		});
	_targetArmor = _armors[0];

	//update the flag status
	_trackCnt++;

	return _flag = ARMOR_LOCAL;
}

bool ArmorDescriptor::isArmorPattern() const
{
	return true;
}

const std::vector<cv::Point2f> ArmorDetector::getArmorVertex() const
{
	vector<cv::Point2f> realVertex;
	for (int i = 0; i < 4; i++)
	{
		realVertex.emplace_back(Point2f(_targetArmor.vertex[i].x + _roi.tl().x, 
										_targetArmor.vertex[i].y + _roi.tl().y));
	}
	return realVertex;
}

int ArmorDetector::getArmorType() const
{
	return _targetArmor.type;
}

int main()
{
	/*//import the target image while testing the opencv config
	Mat img_in = imread("./target.png");
	//image input validation
	if (img_in.empty())
	{
		cout << "Error! Target image not loaded.\n";
		return -1;
	}
	//namedWindow("Target", WINDOW_NORMAL);
	//imshow("Target", img_in); waitKey();*/

	//Mat frame;
	VideoCapture capture("./TargetVideo1.mp4");
	if (!capture.isOpened())
	{
		cout << "Error! Video not loaded.\n";
		return -1;
	}cout << "Video loaded.\n";

	while (capture.read(frame))
	{
		Mat img_in = frame;

		//initialization
		ArmorParam armorParam;
		ArmorDetector detector;
		detector.init(armorParam);

		detector.setEnemyColor(0);
		detector.loadImg(img_in);
		detector.detect();
		vector<cv::Point2f>area = detector.getArmorVertex();
		line(img_in, area[0], area[1], Scalar(0, 255, 255), 2);
		line(img_in, area[1], area[2], Scalar(0, 255, 255), 2);
		line(img_in, area[2], area[3], Scalar(0, 255, 255), 2);
		line(img_in, area[3], area[0], Scalar(0, 255, 255), 2);

		/*float centerX, centerY;
		centerX = (area[0].x + area[1].x + area[2].x + area[3].x) / 4;
		centerY = (area[0].y + area[1].y + area[2].y + area[3].y) / 4;

		cout << "centerX: " << centerX << ", centerY: " << centerX << endl;
		int imgCenterX = img_in.cols / 2;
		int imgCenterY = img_in.rows / 2;
		cout << "Center: " << imgCenterX << ", " << imgCenterY << endl;

		vector<int>data = { int(imgCenterX - centerX),int(imgCenterY - centerY) };
		for (int i = 0; i < data.size(); i++)cout << data.at(i) << endl;*/

		namedWindow("Armor Detection", WINDOW_NORMAL);
		imshow("Armor Detection", img_in); waitKey(1);
	}

	//release all objects and resources
	destroyAllWindows();
	return 0;
}
