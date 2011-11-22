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
	for(int idx = 0; idx < sorted_indices.size(); idx++) {
		Mat originalCol = src.col(sorted_indices[idx]);
		Mat sortedCol = dst.col(idx);
		originalCol.copyTo(sortedCol);
	}
}

Mat cv::sortMatrixByColumn(const Mat& src, vector<int> sorted_indices) {
	Mat dst = src.clone();
	sortMatrixByColumn(src, dst, sorted_indices);
	return dst;
}

void cv::sortMatrixByRow(const Mat& src, Mat& dst, vector<int> sorted_indices) {
	for(int idx = 0; idx < sorted_indices.size(); idx++) {
		Mat originalRow = src.row(sorted_indices[idx]);
		Mat sortedRow = dst.row(idx);
		originalRow.copyTo(sortedRow);
	}
}

Mat cv::sortMatrixByRow(const Mat& src, vector<int> sorted_indices) {
	Mat dst = src.clone();
	sortMatrixByRow(src, dst, sorted_indices);
	return dst;
}


template<typename _Tp>
vector<int> cv::argsort_(const Mat& src, bool asc=true) {
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


void cv::swapByCol(Mat& src, int idx0, int idx1) {
		Mat col0 = src.col(idx0);
		Mat col1 = src.col(idx1);
		Mat tmp(col0.rows, 1, src.type());
		col0.copyTo(tmp);
		col1.copyTo(col0);
		tmp.copyTo(col1);
}

void cv::reverseByCol(const Mat& src, Mat& dst) {
	src.copyTo(dst);
	for(int i = 0, j = src.cols-1; i < j; i++, j--)
		swapByCol(dst,i,j);
}

Mat cv::reverseByCol(const Mat& src) {
	Mat dst;
	reverseByCol(src, dst);
	return dst;

}
void cv::swapByRow(Mat& src, int idx0, int idx1) {
		Mat row0 = src.row(idx0);
		Mat row1 = src.row(idx1);
		Mat tmp(1, src.cols, src.type());
		row0.copyTo(tmp);
		row1.copyTo(row0);
		tmp.copyTo(row1);
}

void cv::reverseByRow(const Mat& src, Mat& dst) {
	src.copyTo(dst);
	for(int i = 0, j = src.rows-1; i < j; i++, j--)
		swapByRow(dst,i,j);
}

Mat cv::reverseByRow(const Mat& src) {
	Mat dst;
	reverseByRow(src, dst);
	return dst;
}

