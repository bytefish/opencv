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

#include <cv.h>

namespace subspace {

using namespace cv;
using namespace std;

/**
 * \class LinearDiscriminantAnalysis
 * \brief Performs a Multiclass Discriminant Analysis for given data.
 */
class LinearDiscriminantAnalysis {

private:
	bool _dataAsRow;
	int _num_components;
	Mat _eigenvectors;
	Mat _eigenvalues;

public:

	LinearDiscriminantAnalysis() :
		_num_components(0),
		_dataAsRow(true) {};

	LinearDiscriminantAnalysis(const Mat& src,
			const vector<int>& labels,
			int num_components = 0,
			bool dataAsRow = true) :
				_num_components(num_components),
				_dataAsRow(dataAsRow)
	{
		this->compute(src, labels); //! compute eigenvectors and eigenvalues
	}

	~LinearDiscriminantAnalysis() {}

	//! compute the discriminants for data in src and labels
	void compute(const Mat& src, const vector<int>& labels);

	//! project
	void project(const Mat& src, Mat& dst);
	Mat project(const Mat& src);

	//! reconstruct
	void reconstruct(const Mat& src, Mat& dst);
	Mat reconstruct(const Mat& src);

	//! returns a const reference to the eigenvectors of this LDA
	const Mat& eigenvectors() const { return _eigenvectors; };

	//! returns a const reference to the eigenvalues of this LDA
	const Mat& eigenvalues() const { return _eigenvalues; }
};

} // namespace
#endif
