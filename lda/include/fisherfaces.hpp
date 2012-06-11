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
#ifndef __FISHERFACES_HPP__
#define __FISHERFACES_HPP__

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace subspace {

/**
 * P. Belhumeur, J. Hespanha, and D. Kriegman,
 * "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection",
 * IEEE Transactions on Pattern Analysis and Machine Intelligence,
 * 19(7):711--720, 1997.
 */
class Fisherfaces {

private:

	int _num_components;
	double _threshold;

	Mat _eigenvectors;
	Mat _eigenvalues;
	Mat _mean;

	vector<Mat> _projections;
	vector<int> _labels;

public:

	Fisherfaces() :
		_num_components(0),
		_threshold(DBL_MAX) {};

	Fisherfaces(int num_components, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {};

	Fisherfaces(const vector<Mat>& src,
			const vector<int>& labels,
			int num_components = 0,
			double threshold = DBL_MAX) :
			    _num_components(num_components),
			    _threshold(threshold)
	{
	    compute(src, labels);
	}


	~Fisherfaces() {}

	// compute the discriminants for data in src and labels
	void compute(const vector<Mat>& src, const vector<int>& labels);
	// returns the nearest neighbor to a query
	int predict(const Mat& src);
	// returns the nearest neighbor to a query and confidence for this prediction
	void predict(const Mat& src, int &label, double &confidence);
	// project samples
	Mat project(const Mat& src);
	// reconstruct samples
	Mat reconstruct(const Mat& src);
	// returns a const reference to the eigenvectors of this LDA
	Mat eigenvectors() const { return _eigenvectors; };
	// returns a const reference to the eigenvalues of this LDA
	Mat eigenvalues() const { return _eigenvalues; }
	// returns a const reference to the eigenvalues of this LDA
	Mat mean() const { return _eigenvalues; }

	void setThreshold(double threshold) { _threshold = threshold; }
	double getThreshold() const { return _threshold; }
};
}
#endif
