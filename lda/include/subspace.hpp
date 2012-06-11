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

#ifndef __SUBSPACE_HPP__
#define __SUBSPACE_HPP__

#include "opencv2/opencv.hpp"

using namespace cv;

namespace subspace {

//! project samples into W
Mat project(const Mat& W, const Mat& mean, const Mat& src);
//! reconstruct samples into W
Mat reconstruct(const Mat& W, const Mat& mean, const Mat& src);

using namespace cv;
using namespace std;

//! Performs a Linear Discriminant Analysis
class LinearDiscriminantAnalysis {

private:

	int _num_components;
	Mat _eigenvectors;
	Mat _eigenvalues;

public:

	//! initialize with 0 components and data given in rows
	LinearDiscriminantAnalysis() :
		_num_components(0){};

	//! initialize with num_components and specify how observations are given
	LinearDiscriminantAnalysis(int num_components) :
		_num_components(num_components) {};

	//! initialize and perform a discriminant analysis with given data in src and labels
	LinearDiscriminantAnalysis(const Mat& src,
			const vector<int>& labels,
			int num_components = 0) :
				_num_components(num_components)
	{
		this->compute(src, labels); //! compute eigenvectors and eigenvalues
	}
	//! initialize and perform a discriminant analysis with given data in src and labels
	LinearDiscriminantAnalysis(const vector<Mat>& src,
			const vector<int>& labels,
			int num_components = 0) :
				_num_components(num_components)
	{
		this->compute(src, labels); //! compute eigenvectors and eigenvalues
	}
	//! destructor
	~LinearDiscriminantAnalysis() {}
	//! compute the discriminants for data in src and labels
	void compute(const Mat& src, const vector<int>& labels);
	//! compute the discriminants for data in src and labels
	void compute(const vector<Mat>& src, const vector<int>& labels);
	//! project
	Mat project(const Mat& src);
	//! reconstruct
	Mat reconstruct(const Mat& src);
	//! returns the eigenvectors of this LDA
	Mat eigenvectors() const { return _eigenvectors; };
	//! returns the eigenvalues of this LDA
	Mat eigenvalues() const { return _eigenvalues; }
};

} // namespace
#endif
