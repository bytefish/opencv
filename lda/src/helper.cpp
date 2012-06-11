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
#include <opencv2/opencv.hpp>
#include "helper.hpp"

using namespace cv;

//------------------------------------------------------------------------------
// cv::isSymmetric
//------------------------------------------------------------------------------
namespace cv {

template<typename _Tp> static bool
isSymmetric_(const Mat& src) {
    if(src.cols != src.rows)
        return false;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            _Tp a = src.at<_Tp> (i, j);
            _Tp b = src.at<_Tp> (j, i);
            if (a != b) {
                return false;
            }
        }
    }
    return true;
}

template<typename _Tp> static bool
isSymmetric_(const Mat& src, double eps) {
    if(src.cols != src.rows)
        return false;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            _Tp a = src.at<_Tp> (i, j);
            _Tp b = src.at<_Tp> (j, i);
            if (std::abs(a - b) > eps) {
                return false;
            }
        }
    }
    return true;
}

}

bool cv::isSymmetric(const Mat& m, double eps) {
    switch (m.type()) {
    case CV_8SC1: return isSymmetric_<char>(m); break;
    case CV_8UC1:
        return isSymmetric_<unsigned char>(m); break;
    case CV_16SC1:
        return isSymmetric_<short>(m); break;
    case CV_16UC1:
        return isSymmetric_<unsigned short>(m); break;
    case CV_32SC1:
        return isSymmetric_<int>(m); break;
    case CV_32FC1:
        return isSymmetric_<float>(m, eps); break;
    case CV_64FC1:
        return isSymmetric_<double>(m, eps); break;
    default:
        break;
    }
    return false;
}

//------------------------------------------------------------------------------
// cv::argsort
//------------------------------------------------------------------------------
vector<int> cv::argsort(const Mat& src, bool ascending) {
    if (src.rows != 1 && src.cols != 1) {
        CV_Error(CV_StsBadArg, "cv::argsort only sorts 1D matrices.");
    }
    int flags = CV_SORT_EVERY_ROW+(ascending ? CV_SORT_ASCENDING : CV_SORT_DESCENDING);
    vector<int> sorted_indices;
    cv::sortIdx(src.reshape(1,1), sorted_indices, flags);
    return sorted_indices;
}

//------------------------------------------------------------------------------
// cv::sortMatrixColumnsByIndices
//------------------------------------------------------------------------------

void cv::sortMatrixColumnsByIndices(const Mat& src, const vector<int>& indices, Mat& dst) {
    dst.create(src.rows, src.cols, src.type());
    for(int idx = 0; idx < indices.size(); idx++) {
        Mat originalCol = src.col(indices[idx]);
        Mat sortedCol = dst.col(idx);
        originalCol.copyTo(sortedCol);
    }
}

Mat cv::sortMatrixColumnsByIndices(const Mat&  src, const vector<int>& indices) {
    Mat dst;
    sortMatrixColumnsByIndices(src, indices, dst);
    return dst;
}

//------------------------------------------------------------------------------
// cv::sortMatrixRowsByIndices
//------------------------------------------------------------------------------
void cv::sortMatrixRowsByIndices(const Mat& src, const vector<int>& indices, Mat& dst) {
    dst.create(src.rows, src.cols, src.type());
    for(int idx = 0; idx < indices.size(); idx++) {
        Mat originalRow = src.row(indices[idx]);
        Mat sortedRow = dst.row(idx);
        originalRow.copyTo(sortedRow);
    }
}

Mat cv::sortMatrixRowsByIndices(const Mat& src, const vector<int>& indices) {
   Mat dst;
   sortMatrixRowsByIndices(src, indices, dst);
   return dst;
}

//------------------------------------------------------------------------------
// cv::asRowMatrix
//------------------------------------------------------------------------------
Mat cv::asRowMatrix(const vector<Mat>& src, int rtype, double alpha, double beta) {
    // number of samples
    size_t n = src.size();
    // return empty matrix if no matrices given
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // create data matrix
    Mat data(n, d, rtype);
    // now copy data
    for(int i = 0; i < n; i++) {
        // make sure data can be reshaped, throw exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // get a hold of the current row
        Mat xi = data.row(i);
        // make reshape happy by cloning for non-continuous matrices
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

//------------------------------------------------------------------------------
// cv::toGrayscale
//------------------------------------------------------------------------------
Mat cv::toGrayscale(const Mat& src, int dtype) {
    // only allow one channel
    if(src.channels() != 1) {
        string error_message = format("Only Matrices with one channel are supported. Expected 1, but was %d.", src.channels());
        CV_Error(CV_StsBadArg, error_message);
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

//------------------------------------------------------------------------------
// cv::transpose
//------------------------------------------------------------------------------
Mat cv::transpose(const Mat& src) {
    Mat dst;
    transpose(src, dst);
    return dst;
}
