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

#include "fisherfaces.hpp"
#include "cv.h"
#include "subspace.hpp"
#include <limits>
#include <cmath>


void subspace::Fisherfaces::compute(const Mat& src, const vector<int>& labels) {

	// assert type
	if((src.type() != CV_32FC1) && (src.type() != CV_64FC1))
		CV_Error(CV_StsBadArg, "src must be a valid matrix float or double matrix");

	Mat data;
	src.convertTo(data, CV_64FC1); // we want to work with double precision!
	// turn into row vector samples
	if(!_dataAsRow)
		transpose(data,data);

	if(labels.size() != data.rows)
		CV_Error(CV_StsBadArg, "labels array must be a valid 1d integer vector of len(src) elements");

	// store labels
	_labels = labels;

	// compute fisherfaces
	int N = data.rows; //! number of samples
	int D = data.cols; //! dimension of samples
	int C = *max_element(labels.begin(), labels.end()) + 1; //! number of classes
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (N-C));
	LinearDiscriminantAnalysis lda(pca.project(data),labels, C-1, true);
	lda.eigenvalues().copyTo(_eigenvalues);
	gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, CV_GEMM_A_T);
	// store projections
	for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++)
		_projections.push_back(project(data.row(sampleIdx)));
}

void subspace::Fisherfaces::project(const Mat& src, Mat& dst) {
	if(_dataAsRow) {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst, CV_GEMM_A_T + CV_GEMM_B_T);
	} else {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst,  CV_GEMM_A_T);
	}
}

void subspace::Fisherfaces::reconstruct(const Mat& src, Mat& dst) {
	if(_dataAsRow) {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst, CV_GEMM_B_T);
	} else {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst);
	}
}

int subspace::Fisherfaces::predict(const Mat& src) {
	Mat query;
	src.convertTo(query, CV_64FC1);
	query = query.reshape(1,1);
	if(!_dataAsRow)
		transpose(query,query);
	Mat projection = project(query);
	double minDist = numeric_limits<double>::max();
	int minClass = -1;
	for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		double dist = norm(_projections[sampleIdx], projection, NORM_L2);
		if(dist < minDist) {
			minDist = dist;
			minClass = _labels[sampleIdx];
		}
	}
	return minClass;
}

Mat subspace::Fisherfaces::project(const Mat& src) {
	Mat dst;
	project(src, dst);
	return dst;
}

Mat subspace::Fisherfaces::reconstruct(const Mat& src) {
	Mat dst;
	reconstruct(src, dst);
	return dst;
}

