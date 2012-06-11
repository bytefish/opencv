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

#include "opencv2/opencv.hpp"
#include <limits.h>
#include <vector>

using namespace std;
using namespace cv;

class Eigenfaces {
private:
	int _num_components;
	double _threshold;
	vector<Mat> _projections;
	vector<int> _labels;
	Mat _eigenvectors;
	Mat _eigenvalues;
	Mat _mean;

public:
	Eigenfaces() :
		_num_components(0),
		_threshold(DBL_MAX) {};

	//! create empty eigenfaces with num_components
	Eigenfaces(int num_components, double threshold = DBL_MAX) :
		_num_components(num_components),
		_threshold(threshold) {};

	//! compute num_component eigenfaces for given images in src and corresponding classes in labels
	Eigenfaces(const vector<Mat>& src,
			const vector<int>& labels,
			int num_components = 0,
			double threshold = DBL_MAX) :
			    _num_components(num_components),
			    _threshold(threshold)
	{
	 compute(src, labels);
	}

	//! computes a PCA for given data
	void compute(const vector<Mat>& src, const vector<int>& labels);
	//! predicts the label for a given sample
	int predict(const Mat& src);
	//! predicts the label for a given sample and the confidence of this prediction
	void predict(const Mat& src, int &label, double &confidence);
	//! projects a sample
	Mat project(const Mat& src);
	//! reconstructs a sample
	Mat reconstruct(const Mat& src);
	//! returns the eigenvectors of this PCA
	Mat eigenvectors() const { return _eigenvectors; }
	//! returns the eigenvalues of this PCA
	Mat eigenvalues() const { return _eigenvalues; }
	//! returns the mean of this PCA
	Mat mean() const { return _mean; }
};

#endif /* EIGENFACES_H_ */
