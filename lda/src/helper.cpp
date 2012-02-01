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

#include "helper.hpp"
#include <iostream>
using namespace cv;

//! sort order for shuffle
template<typename _Tp>
class cv::SortByFirstAscending_ {
public:
	bool operator()(const std::pair<_Tp,int>& left, const std::pair<_Tp,int>& right) {
		return left.first < right.first;
	}
};

//! descending sort operator
template<typename _Tp>
class cv::SortByFirstDescending_ {
public:
	bool operator()(const std::pair<_Tp,int>& left, const std::pair<_Tp,int>& right) {
		return left.first > right.first;
	}
};


void cv::sortMatrixByColumn(const Mat& src, Mat& dst, vector<int> sorted_indices) {
	dst.create(src.rows, src.cols, src.type());
	for(int idx = 0; idx < sorted_indices.size(); idx++) {
		Mat originalCol = src.col(sorted_indices[idx]);
		Mat sortedCol = dst.col(idx);
		originalCol.copyTo(sortedCol);
	}
}

Mat cv::sortMatrixByColumn(const Mat& src, vector<int> sorted_indices) {
	Mat dst;
	sortMatrixByColumn(src, dst, sorted_indices);
	return dst;
}

void cv::sortMatrixByRow(const Mat& src, Mat& dst, vector<int> sorted_indices) {
	dst.create(src.rows, src.cols, src.type());
	for(int idx = 0; idx < sorted_indices.size(); idx++) {
		Mat originalRow = src.row(sorted_indices[idx]);
		Mat sortedRow = dst.row(idx);
		originalRow.copyTo(sortedRow);
	}
}

Mat cv::sortMatrixByRow(const Mat& src, vector<int> sorted_indices) {
	Mat dst;
	sortMatrixByRow(src, dst, sorted_indices);
	return dst;
}

vector<int> cv::vec_unqiue(vector<int> src) {
	src.erase(unique(src.begin(),src.end()), src.end());
	return src;
}

template<typename _Tp>
vector<int> cv::argsort_(const Mat& src, bool asc) {
	if(src.rows != 1 && src.cols != 1)
		CV_Error(CV_StsBadArg, "Argsort only sorts 1D Vectors");
	// <value>,<index>
	vector< pair<_Tp,int> > val_indices;
	for(int i = 0; i < src.rows; i++)
		for(int j = 0; j < src.cols; j++)
			val_indices.push_back(make_pair(src.at<_Tp>(i,j),val_indices.size()));
	if(asc) {
		std::sort(val_indices.begin(), val_indices.end(), SortByFirstAscending_<_Tp>());
	} else {
		std::sort(val_indices.begin(), val_indices.end(), SortByFirstDescending_<_Tp>());
	}

	vector<int> indices;
	for(int i=0; i < val_indices.size(); i++)
		indices.push_back(val_indices[i].second);
	return indices;
}

//! get
vector<int> cv::argsort(const Mat& src, bool asc) {
	switch(src.type()) {
		case CV_8SC1: return argsort_<char>(src,asc); break;
		case CV_8UC1: return argsort_<unsigned char>(src,asc); break;
		case CV_16SC1: return argsort_<short>(src,asc); break;
		case CV_16UC1: return argsort_<unsigned short>(src,asc); break;
		case CV_32SC1: return argsort_<int>(src,asc); break;
		case CV_32FC1: return argsort_<float>(src,asc); break;
		case CV_64FC1: return argsort_<double>(src,asc); break;
	}
}

Mat cv::asColumnMatrix(const vector<Mat>& src) {
	int n = src.size();
	int d = src[0].total();
	Mat data(d, n, CV_32FC1);
	for(int i = 0; i < src.size(); i++) {
		Mat tmp,
			xi = data.col(i);
		src[i].convertTo(tmp, CV_32FC1);
		tmp.reshape(1, d).copyTo(xi);
	}
	return data;
}

Mat cv::asRowMatrix(const vector<Mat>& src) {
	int n = src.size();
	int d = src[0].total();
	Mat data(n, d, CV_32FC1);
	for(int i = 0; i < src.size(); i++) {
		Mat tmp,
			xi = data.row(i);
		src[i].convertTo(tmp, CV_32FC1);
		tmp.reshape(1, 1).copyTo(xi);
	}
	return data;
}

Mat cv::transpose(const Mat& src) {
		Mat dst;
		transpose(src, dst);
		return dst;
}

Mat cv::toGrayscale(const Mat& src) {
	Mat dst;
	cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

string cv::num2str(int i) {
	stringstream ss;
	ss << i;
	return ss.str();
}
