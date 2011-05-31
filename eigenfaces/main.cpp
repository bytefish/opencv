#include "cv.h"
#include "highgui.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <string>

#include <cmath>
#include <limits>
#include <iterator>

#include "Eigenfaces.h"

using namespace std;
using namespace cv;

Mat toGrayscale(const Mat& mat) {
	Mat gMat(mat.rows, mat.cols, CV_8UC1);
	double min, max;
	minMaxLoc(mat, &min, &max);
	for(int row = 0; row < mat.rows; row++) {
		for(int col = 0; col < mat.cols; col++) {
			gMat.at<uchar>(row, col) = 255 * ((mat.at<float>(row, col) - min) / (max - min));
		}
	}
	return gMat;
}

void read_csv(const string& filename, vector<string>& files, vector<int>& classes) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if(file) {
		std::string line, path, classlabel;
		while (std::getline(file, line)) {
			std::stringstream liness(line);
			std::getline(liness, path, ';');
			std::getline(liness, classlabel);

			files.push_back(path);
			classes.push_back(atoi(classlabel.c_str()));
		}
	} else {
		cerr << "Error: Failed to open file.";
	}
}

int main(int argc, char *argv[]) {
	vector<string> files;
	vector<int> classes;

	read_csv("./at.txt", files, classes);
	int numImages = files.size();

	// get dimension from first image
	Mat img = imread(files[0], 0);
	int total = img.cols * img.rows;

	Mat data(total, numImages, CV_32FC1);
	for(int instanceIdx = 0; instanceIdx < files.size() - 1; instanceIdx++) {
		Mat xi = data.col(instanceIdx);
		Mat tmp_img;
		imread(files[instanceIdx], 0).convertTo(tmp_img, CV_32FC1, 1/255.);
		tmp_img.reshape(1, total).copyTo(xi);
	}

	// learn model
	Eigenfaces eigenfaces(data, classes, 20);

	// test model (with last image...)
	imread(files[numImages - 1],0).convertTo(img, CV_32FC1, 1/255.);
	int predicted = eigenfaces.predict(img.reshape(1, total));

	cout << "predicted class = " << predicted << endl;
	cout << "actual class = " << classes[numImages-1] << endl;

	// show first 10 eigenfaces
	for(int i = 0; i < 19; i++) {
		stringstream ss;
		ss << i;//add number to the stream
		imshow(ss.str(), toGrayscale(eigenfaces.eigenvectors().row(i)).reshape(1, img.rows));
	}

	waitKey(0);
}
