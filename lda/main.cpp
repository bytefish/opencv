/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include <cv.h>
#include <highgui.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "subspace.hpp"
#include "fisherfaces.hpp"
#include "helper.hpp"

using namespace cv;
using namespace std;

Mat linspace(float x0, float x1, int n) {
	Mat pts(n, 1, CV_32FC1);
	double step = (x1-x0)/floor(n-1);
	for(int i = 0; i < n; i++)
		pts.at<float>(i,0) = x0+i*step;
	return pts;
}

Mat lut_red_to_blue() {
	Mat roi, lut;

	// ... red
	Mat red = Mat::zeros(256, 1,CV_32FC1);
	roi = Mat(red, Range(127,192), Range::all());
	roi += linspace(0,1,65);
	roi = Mat(red, Range(192,256), Range::all());
	roi += 1;

	// ... green
	Mat green = Mat::zeros(256,1,CV_32FC1);
	roi = Mat(green, Range(0,64), Range::all());
	roi += linspace(0,1,64);
	roi = Mat(green, Range(64,192), Range::all());
	roi += 1;
	roi = Mat(green, Range(192,256), Range::all());
	roi += linspace(1, 0, 64);

	// ... blue
	Mat blue = Mat::zeros(256,1,CV_32FC1);
	roi = Mat(blue, Range(0,64), Range::all());
	roi += 1;
	roi = Mat(blue, Range(64,128), Range::all());
	roi += linspace(1, 0, 64);
	Mat planes[] = {blue,green,red};

	// ... finally merge them
	merge(planes, 3, lut);
	return lut;
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


int main(int argc, const char *argv[]) {

	// example from: http://www.bytefish.de/wiki/pca_lda_with_gnu_octave
	double d[11][2] = {
			{2, 3},
			{3, 4},
			{4, 5},
			{5, 6},
			{5, 7},
			{2, 1},
			{3, 2},
			{4, 2},
			{4, 3},
			{6, 4},
			{7, 6}};
	int c[11] = {0,0,0,0,0,1,1,1,1,1,1};

	// convert into OpenCV representation
	Mat _data = Mat(11, 2, CV_64FC1, d).clone();
	vector<int> _classes(c, c + sizeof(c) / sizeof(int));
	// perform the lda
	subspace::LinearDiscriminantAnalysis lda(_data, _classes);
	// GNU Octave finds the following eigenvalues
	//octave> d
	//d =
	//	 1.5195e+00
	//   6.5052e-18
	cout << "Eigenvalues:" << endl << lda.eigenvalues() << endl;
	// Eigen outputs:
	// [1.519536390756363; 9.980626757982641e-19]

	cout << "Eigenvectors:" << endl << lda.eigenvectors() << endl;

	// project a data sample onto the subspace identified by LDA
	Mat x = _data.row(0);
	cout << "Projection of " << x << ": " << endl;
	cout << lda.project(x) << endl;

	// ...

	vector<string> files;
	vector<int> classes;

	read_csv("/home/philipp/facerec/data/yaleface.txt", files, classes);
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

	// turn into row-order
	transpose(data,data);

	subspace::Fisherfaces model(data, classes);
	// test model (with last image...)
	imread(files[numImages - 1],0).convertTo(img, CV_32FC1, 1/255.);
	int predicted = model.predict(img.reshape(1, 1));

	cout << "predicted class = " << predicted << endl;
	cout << "actual class = " << classes[numImages-1] << endl;
	Mat lut = lut_red_to_blue();
	// show first 10 eigenfaces
	for(int i = 0; i < max(1, min(10, model.eigenvectors().cols)); i++) {
		stringstream ss;
		ss << "fisherface_" << i;
		Mat v;
		model.eigenvectors().col(i).copyTo(v);
		normalize(v, v, 0, 255, NORM_MINMAX, CV_8UC1);
		cvtColor(v, v, CV_GRAY2BGR);
		LUT(v, lut, v);
		normalize(v, v, 0, 255, NORM_MINMAX, CV_8UC3);
		imshow(ss.str(), v.reshape(3, img.rows));
	}
	waitKey(0);
	return 0;
}
