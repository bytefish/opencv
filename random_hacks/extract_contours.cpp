#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

int main() {
	// read in the apple
	Mat img0 = cv::imread("/home/philipp/apple.jpg", 1);

	Mat img1;
	cvtColor(img0, img1, CV_RGB2GRAY);

	// apply your filter
	Canny(img1, img1, 100, 200);

	// find the contours
	std::vector< std::vector<cv::Point> > contours;
	findContours(img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// you could also reuse img1 here
	Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);

	// draw contours with CV_FILLED fills the connected components found
	drawContours(mask, contours, -1, Scalar(1), CV_FILLED);

	/*
	 Actually before drawing _all_ contours you could also decide
	 to only draw the contour of the largest connected component
	 found. Here's some commented out code how to do that:
	*/

//	vector<double> areas(contours.size());
//	for(int i = 0; i < contours.size(); i++)
//		areas[i] = contourArea(Mat(contour[i]));
//	double max;
//	Point maxPosition;
//	minMaxLoc(Mat(areas),0,&max,0,&maxPosition);
//	drawContours(mask, contours, maxPosition.y, Scalar(1), CV_FILLED);



	// let's create a new image now
	Mat crop(img0.rows, img0.cols, CV_8UC3);

	// set background to green
	crop.setTo(Scalar(0,255,0));

	// and copy the magic apple
	img0.copyTo(crop, mask);

	// show the images
	imshow("original", img0);
	imshow("canny", img1);
	imshow("cropped", crop);

//	imwrite("/home/philipp/apple_canny.jpg", img1);
//	imwrite("/home/philipp/apple_cropped.jpg", crop);

	waitKey();
	return 0;
}

