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
#include "subspace.hpp"
#include "helper.hpp"
#include <limits>
#include <cmath>


void subspace::Fisherfaces::compute(const vector<Mat>& src, const vector<int>& labels) {
    if(src.size() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(CV_StsUnsupportedFormat, error_message);
    }
    // wrap asRowMatrix in a try/catch, as people tend to pass wrong data here
    Mat data = asRowMatrix(src, CV_64FC1);
    // number of samples (N) and dimensions (D)
    int N = data.rows;
    int D = data.cols;
    // assert data is correctly given
    if(labels.size() != N) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", N, labels.size());
        CV_Error(CV_StsBadArg, error_message);
    }
    // the following equals len(unique(C))
    int C = remove_dups(labels).size();
    // clip number of components to be a valid number
    if((_num_components <= 0) || (_num_components > (C-1)))
        _num_components = (C-1);
    // perform a PCA and keep (N-C) components
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (N-C));
    // project the data and perform a LDA on it
    subspace::LDA lda(pca.project(data), labels, _num_components);
    // store the total mean vector
    _mean = pca.mean.reshape(1,1);
    // store labels
    _labels = labels;
    // store the eigenvalues of the discriminants (and make sure they are doubles!)
    lda.eigenvalues().convertTo(_eigenvalues, CV_64FC1);
    // Now calculate the projection matrix as pca.eigenvectors * lda.eigenvectors.
    // Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
    gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);
    // store the projections of the original data
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = subspace::project(_eigenvectors, _mean, data.row(sampleIdx).clone());
        _projections.push_back(p);
    }
}

Mat subspace::Fisherfaces::project(const Mat& src) {
	return subspace::project(_eigenvectors, _mean, src);
}

Mat subspace::Fisherfaces::reconstruct(const Mat& src) {
	return subspace::reconstruct(_eigenvectors, _mean, src);
}

void subspace::Fisherfaces::predict(const Mat& src, int &minClass, double &minDist) {
    // check data alignment just for clearer exception messages
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This cv::Fisherfaces model is not computed yet. Did you call cv::Fisherfaces::train?";
        CV_Error(CV_StsError, error_message);
    } else if(_eigenvectors.rows != src.total()) {
        string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(CV_StsError, error_message);
    }
    // project into LDA subspace
    Mat q = subspace::project(_eigenvectors, _mean, src.reshape(1,1));
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels[sampleIdx];
        }
    }
}

int subspace::Fisherfaces::predict(const Mat& src) {
    int label;
    double dummy;
    predict(src, label, dummy);
    return label;
}
