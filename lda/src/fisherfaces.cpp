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
#include "helper.hpp"
#include <limits>
#include <cmath>
#include <eigen3/Eigen/Dense>

subspace::Fisherfaces::Fisherfaces(
		const vector<Mat>& src,
		const vector<int>& labels,
		int num_components,
		bool dataAsRow) {
	_num_components = num_components;
	_dataAsRow = dataAsRow;
	// compute the fisherfaces
	compute(src, labels);
}

void subspace::Fisherfaces::compute(const vector<Mat>& src, const vector<int>& labels) {
	compute(_dataAsRow? asRowMatrix(src) : asColumnMatrix(src), labels);
}

void subspace::Fisherfaces::compute(const Mat& src, const vector<int>& labels) {
	if(src.channels() != 1)
		CV_Error(CV_StsBadArg, "Only single channel matrices allowed.");
	Mat data = _dataAsRow ? src.clone() : transpose(src);
	data.convertTo(data, CV_64FC1);
	if(labels.size() != data.rows)
		CV_Error(CV_StsBadArg, "The number of samples must equal the number of labels.");
	// compute the Fisherfaces
	int N = data.rows; // number of samples
	int D = data.cols; // dimension of samples
	int C = vec_unqiue(labels).size(); // number of unique classes
	// clip number of components to be a valid number
	if((_num_components <= 0) || (_num_components > (C-1)))
		_num_components = (C-1);
	// perform a PCA and keep (N-C) components
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (N-C));
	// project the data and perform a LDA on it
	LinearDiscriminantAnalysis lda(pca.project(data),labels, _num_components);
	// store the total mean vector
	_mean = _dataAsRow ? pca.mean.reshape(1,1) : pca.mean.reshape(1, pca.mean.total());
	// store labels
	_labels = labels;
	// store the eigenvalues of the discriminants
	lda.eigenvalues().copyTo(_eigenvalues);
	// Now calculate the projection matrix as pca.eigenvectors * lda.eigenvectors.
	// Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
	gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, CV_GEMM_A_T);
	// store the projections of the original data
	for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++)
		_projections.push_back(project(_dataAsRow ? src.row(sampleIdx) : src.col(sampleIdx)));

}

Mat subspace::Fisherfaces::project(const Mat& src) {
	return subspace::project(_eigenvectors, _mean, src, _dataAsRow);
}

Mat subspace::Fisherfaces::reconstruct(const Mat& src) {
	return subspace::reconstruct(_eigenvectors, _mean, src, _dataAsRow);
}

int subspace::Fisherfaces::predict(const Mat& src) {
	Mat q = project(_dataAsRow ? src.reshape(1,1) : src.reshape(1, src.total()));
	// find 1-nearest neighbor
	double minDist = numeric_limits<double>::max();
	int minClass = -1;
	for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		double dist = norm(_projections[sampleIdx], q, NORM_L2);
		if(dist < minDist) {
			minDist = dist;
			minClass = _labels[sampleIdx];
		}
	}
	return minClass;
}
