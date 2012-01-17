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

#include "cv.h"
#include <eigen3/Eigen/SVD>

using namespace cv;
using namespace std;

namespace subspace {

/**
 * Fisherfaces
 *
 * Implements the Fisherfaces method as described in:
 *
 *  * P. Belhumeur, J. Hespanha, and D. Kriegman,
 *    "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection",
 *    IEEE Transactions on Pattern Analysis and Machine Intelligence,
 *    19(7):711--720,
 *    1997.
 *
 * TODO Use OpenCVs KNearestNeighbor class
 */
class Fisherfaces {

private:
	bool _dataAsRow;
	int _num_components;
	Mat _eigenvectors;
	Mat _eigenvalues;
	vector<Mat> _projections;
	vector<int> _labels;

public:

	Fisherfaces() :
		_num_components(0),
		_dataAsRow(true) {};

	Fisherfaces(const Mat& src,
			const vector<int>& labels,
			int num_components = 0,
			bool dataAsRow = true) :
				_num_components(num_components),
				_dataAsRow(dataAsRow)
	{
		this->compute(src, labels); //! compute eigenvectors and eigenvalues
	}

	~Fisherfaces() {}

	// compute the discriminants for data in src and labels
	void compute(const Mat& src, const vector<int>& labels);

	// project
	void project(const Mat& src, Mat& dst);
	Mat project(const Mat& src);

	// reconstruct
	void reconstruct(const Mat& src, Mat& dst);
	Mat reconstruct(const Mat& src);

	// returns a const reference to the eigenvectors of this LDA
	const Mat& eigenvectors() const { return _eigenvectors; };

	// returns a const reference to the eigenvalues of this LDA
	const Mat& eigenvalues() const { return _eigenvalues; }

	// returns the nearest neighbor to a query
	int predict(const Mat& src);
};
}
#endif
