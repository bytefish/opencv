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
#include <eigen3/Eigen/Dense>
#include "subspace.hpp"
#include "helper.hpp"

using namespace Eigen;
using namespace cv;

void subspace::LinearDiscriminantAnalysis::compute(const Mat& src, const vector<int>& labels) {
	// assert type
	if((src.type() != CV_32FC1) && (src.type() != CV_64FC1))
		CV_Error(CV_StsBadArg, "src must be a valid float or double matrix");
	// we want double precision, so convert
	Mat data;
	src.convertTo(data, CV_64FC1);
	// turn into row vector samples
	if(!_dataAsRow)
		transpose(data,data);
	// assert, that labels are valid 32bit signed integer values
	if(labels.size() != data.rows)
		CV_Error(CV_StsBadArg, "labels array must be a valid 1d integer vector of len(src) elements");
	// get information about the data
	int N = data.rows; // number of samples
	int D = data.cols; // dimension of samples
	int C = *max_element(labels.begin(), labels.end()) + 1; // number of classes
	// warn: Within-classes scatter matrix will become singular!
	if(N < D)
		cout << "Less instances than feature dimension! Computation will probably fail." << endl;
	// warn: There are atmost (C-1) non-zero eigenvalues!
	if((_num_components > (C-1)) || (_num_components < 1)) {
		_num_components = C-1;
		cout << "num_components set to: " << _num_components << "!" << endl;
	}
	// the mean over all classes
	Mat meanTotal = Mat::zeros(1, D, data.type());
	// the mean for each class
	Mat meanClass[C];
	int numClass[C];
	// initialize
	for (int i = 0; i < C; i++) {
		numClass[i] = 0;
		meanClass[i] = Mat::zeros(1, D, data.type()); //! Dx1 image vector
	}
	// calculate sums
	for (int i = 0; i < N; i++) {
		Mat instance = data.row(i);
		int classIdx = labels[i];
		add(meanTotal, instance, meanTotal);
		add(meanClass[classIdx], instance, meanClass[classIdx]);
		numClass[classIdx]++;
	}
	// calculate means
	meanTotal.convertTo(meanTotal, meanTotal.type(), 1.0/static_cast<double>(N));
	for (int i = 0; i < C; i++)
		meanClass[i].convertTo(meanClass[i], meanClass[i].type(), 1.0/static_cast<double>(numClass[i]));
	// subtract class means
	for (int i = 0; i < N; i++) {
		int classIdx = labels[i];
		Mat instance = data.row(i);
		subtract(instance, meanClass[classIdx], instance);
	}
	// calculate within-classes scatter
	Mat Sw = Mat::zeros(D, D, data.type());
	mulTransposed(data, Sw, true);
	// calculate between-classes scatter
	Mat Sb = Mat::zeros(D, D, data.type());
	for (int i = 0; i < C; i++) {
		Mat tmp;
		subtract(meanClass[i], meanTotal, tmp);
		mulTransposed(tmp, tmp, true);
		add(Sb, tmp, Sb);
	}
	// invert Sw
	Mat Swi = Sw.inv();
	// M = inv(Sw)*Sb
	Mat M;
	gemm(Swi, Sb, 1.0, Mat(), 0.0, M);
	// now switch to eigen (cv2eigen defined in helper.hpp)
	MatrixXd Me;
	cv2eigen(M, Me);
	// solve eigenvalue problem for the general matrix $M = Sw^{-1} Sb$
	Eigen::EigenSolver<MatrixXd> es(Me);
	// copy real values over to opencv
	eigen2cv(MatrixXd(es.eigenvectors().real()), _eigenvectors);
	eigen2cv(MatrixXd(es.eigenvalues().real()), _eigenvalues);
	// get sorted indices descending by eigenvalue
	vector<int> sorted_indices = argsort(_eigenvalues, false);
	// now sort eigenvalues and eigenvectors accordingly
	_eigenvalues = sortMatrixByRow(_eigenvalues, sorted_indices);
	_eigenvectors = sortMatrixByColumn(_eigenvectors, sorted_indices);
	// and now take only the num_components and we're out!
	_eigenvalues = Mat(_eigenvalues, Range(0,_num_components), Range::all());
	_eigenvectors = Mat(_eigenvectors, Range::all(), Range(0, _num_components));
}

void subspace::LinearDiscriminantAnalysis::project(const Mat& src, Mat& dst) {
	if(_dataAsRow) {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst, CV_GEMM_A_T + CV_GEMM_B_T);
	} else {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst,  CV_GEMM_A_T);
	}
}

void subspace::LinearDiscriminantAnalysis::reconstruct(const Mat& src, Mat& dst) {
	if(_dataAsRow) {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst, CV_GEMM_B_T);
	} else {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst);
	}
}

Mat subspace::LinearDiscriminantAnalysis::project(const Mat& src) {
	Mat dst;
	project(src, dst);
	return dst;
}

Mat subspace::LinearDiscriminantAnalysis::reconstruct(const Mat& src) {
	Mat dst;
	reconstruct(src, dst);
	return dst;
}
