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

	return 0;
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

	// show first 10 eigenfaces
	for(int i = 0; i < max(1, min(10, model.eigenvectors().cols)); i++) {
		stringstream ss;
		ss << i;//add number to the stream
		Mat normalized;
		normalize(model.eigenvectors().col(i), normalized, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow(ss.str(), normalized.reshape(1, img.rows));
	}
	waitKey(0);
	return 0;
}
