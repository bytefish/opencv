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

/*
 * Read image filenames and corresponding classes from a CSV file. Example CSV file:
 *	/path/to/image0.jpg;0
 *	/path/to/image1.jpg;0
 *	/path/to/image2.jpg;0
 *	/path/to/image3.jpg;1
 *	/path/to/image4.jpg;1
 *	...
 */
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
	// example taken from: http://www.bytefish.de/wiki/pca_lda_with_gnu_octave
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
	// hold the path to the image and corresponding class
	vector<string> files;
	vector<int> classes;
	// Example of a CSV File
	//
	// CSV -- https://github.com/bytefish/opencv/blob/master/lda/at.txt
	// Database -- http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
	//
	// Always make sure classes are given as {0, 1,..., n}!
	read_csv("/path/to/your/database.txt", files, classes);
	int numImages = files.size();
	// get dimension from first image
	Mat img = imread(files[0], 0);
	int total = img.cols * img.rows;
	// read in the data, each row is an image
	Mat data(numImages, total, CV_32FC1);
	for(int instanceIdx = 0; instanceIdx < files.size() - 1; instanceIdx++) {
		Mat xi = data.row(instanceIdx);
		Mat tmp_img;
		imread(files[instanceIdx], 0).convertTo(tmp_img, CV_32FC1, 1/255.);
		tmp_img.reshape(1, 1).copyTo(xi);
	}
	// build the Fisherfaces model (data is in row)
	subspace::Fisherfaces model(data, classes);
	// test model (with last image...)
	imread(files[numImages - 1],0).convertTo(img, CV_32FC1, 1/255.);
	int predicted = model.predict(img.reshape(1, 1));
	cout << "predicted class = " << predicted << endl;
	cout << "actual class = " << classes[numImages-1] << endl;
	stringstream ss;
	// show first 10 Fisherfaces
	for(int i = 0; i < max(1, min(10, model.eigenvectors().cols)); i++) {
		ss << "fisherface_" << i;
		Mat v;
		model.eigenvectors().col(i).copyTo(v);
		normalize(v, v, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow(ss.str(), v.reshape(1, img.rows));
	}
	waitKey(0);
	return 0;
}
