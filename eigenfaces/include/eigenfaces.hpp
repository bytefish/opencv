/*
 * Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
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

#ifndef EIGENFACES_HPP_
#define EIGENFACES_HPP_

#include <cv.h>
#include <limits.h>
#include <vector>

using namespace std;
using namespace cv;

class Eigenfaces {
private:
	bool _dataAsRow;
	int _num_components;
	vector<Mat> _projections;
	vector<int> _labels;
	Mat _eigenvectors;
	Mat _eigenvalues;
	Mat _mean;

public:
	Eigenfaces() :
		_num_components(0),
		_dataAsRow(true) {};
	//! create empty eigenfaces with num_components
	Eigenfaces(int num_components, bool dataAsRow = true) :
		_num_components(num_components),
		_dataAsRow(dataAsRow) {};
	//! compute num_component eigenfaces for given images in src and corresponding classes in labels
	Eigenfaces(const vector<Mat>& src, const vector<int>& labels, int num_components = 0, bool dataAsRow = true);
	//! compute num_component eigenfaces for given images in src and corresponding classes in labels
	//    default is observation by row, pass dataAsRow = false if observations are given by column
	Eigenfaces(const Mat& src, const vector<int>& labels, int num_components = 0, bool dataAsRow = true);
	//! computes a PCA for given data
	void compute(const vector<Mat>& src, const vector<int>& labels);
	//! computes a PCA for given data
	void compute(const Mat& src, const vector<int>& labels);
	//! predicts the label for a given sample
	int predict(const Mat& src);
	//! projects a sample
	Mat project(const Mat& src);
	//! reconstructs a sample
	Mat reconstruct(const Mat& src);
	//! returns the eigenvectors of this PCA
	Mat eigenvectors() { return _eigenvectors; }
	//! returns the eigenvalues of this PCA
	Mat eigenvalues() { return _eigenvalues; }
	//! returns the mean of this PCA
	Mat mean() { return _mean; }
};

#endif /* EIGENFACES_H_ */
