#include "helper.hpp"
#include <limits>
using namespace std;
using namespace cv;

Mat cv::linspace(float x0, float x1, int n) {
	Mat pts(n, 1, CV_32FC1);
	double step = (x1-x0)/floor(n-1);
	for(int i = 0; i < n; i++)
		pts.at<float>(i,0) = x0+i*step;
	return pts;
}

template<typename _Tp>
class cv::SortByFirstAscending_ {
public:
	bool operator()(const pair<_Tp,int>& left, const pair<_Tp,int>& right) {
		return left.first < right.first;
	}
};

template<typename _Tp>
class cv::SortByFirstDescending_ {
public:
	bool operator()(const pair<_Tp,int>& left, const pair<_Tp,int>& right) {
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

void cv::diff(const Mat& src, Mat& dst) {
	if(src.rows != 1 && src.cols != 1)
		CV_Error(CV_StsBadArg,"Only 1-dimensional matrices supported.");
	if(src.rows == 1)
		return subtract(Mat(src,Range::all(),Range(1,src.cols)), Mat(src,Range::all(),Range(0,src.cols-1)), dst);
	else
		return subtract(Mat(src,Range(1,src.rows),Range::all()), Mat(src,Range(0,src.rows-1),Range::all()), dst);

}

Mat cv::diff(const Mat& src) {
	Mat dst;
	diff(src, dst);
	return dst;
}

template <typename _Tp>
int nearest_bin_(const Mat& src, _Tp value)
{
  int low = 0;
  int high = src.rows-1;
  // first find the left-most insertion bin
  while (low <= high) {
	int mid = (low + high) / 2;
	if (src.at<_Tp>(mid,0) > value) {
	  high = mid - 1;
	} else if(src.at<_Tp>(mid,0) < value) {
	  low = mid + 1;
	} else {
		return mid;
	}
  }
  // left-most point found
  if(low == 0)
	  return 0;
  // left boundary
  if(src.at<_Tp>(low-1) < src.at<_Tp>(low))
	  return low-1;
  // right boundary is closer
  return low;
}

template <typename _Tp>
int cv::nearest_bin(const Mat& src, _Tp value)
{
  if(src.cols!=1)
		  CV_Error(CV_StsBadArg, "Only row vectors allowed.");
	switch(src.type()) {
		case CV_8SC1: return nearest_bin_<char>(src,value); break;
		case CV_8UC1: return nearest_bin_<unsigned char>(src,value); break;
		case CV_16SC1: return nearest_bin_<short>(src,value); break;
		case CV_16UC1: return nearest_bin_<unsigned short>(src,value); break;
		case CV_32SC1: return nearest_bin_<int>(src,value); break;
		case CV_32FC1: return nearest_bin_<float>(src,value); break;
		case CV_64FC1: return nearest_bin_<double>(src,value); break;
	}
}

template <typename _Tp>
Mat interp1_(const Mat& X, const Mat& Y, const Mat& xi) {
	// Xs,Ys for sorted interpolation tables
	Mat Xs, Ys;
	int n = xi.rows;
	// sort both vectors
	vector<int> sort_indices = argsort(X);
	Xs = sortMatrixByRow(X,sort_indices);
	Ys = sortMatrixByRow(Y,sort_indices);
	// calc interpolation weights
	Mat dy = diff(Ys)/diff(Xs);
	// then interpolate values (lineary), probably enhance it here for cubic interpolation...
	Mat yi = Mat::zeros(xi.size(), xi.type());
	for(int i = 0; i < n; i++) {
		int valIdx = nearest_bin(Xs,xi.at<_Tp>(i,0));
		yi.at<_Tp>(i,0) += Ys.at<_Tp>(valIdx,0) + dy.at<_Tp>(valIdx, 0) * (xi.at<_Tp>(i,0) - Xs.at<_Tp>(valIdx,0));
	}
	return yi;
}

Mat cv::interp1(const Mat& X, const Mat& Y, const Mat& xi) {
	if((X.type() != Y.type()) || (Y.type() != xi.type()))
		CV_Error(CV_StsBadArg, "X and Y and xi must be of same type.");
	if(X.cols != 1)
		CV_Error(CV_StsBadSize, "Only row-vectors allowed.");
	if(X.rows != Y.rows || X.cols != Y.cols)
		CV_Error(CV_StsBadSize, "X and Y must be well-aligned.");
	switch(X.type()) {
		case CV_8SC1: return interp1_<char>(X,Y,xi); break;
		case CV_8UC1: return interp1_<unsigned char>(X,Y,xi); break;
		case CV_16SC1: return interp1_<short>(X,Y,xi); break;
		case CV_16UC1: return interp1_<unsigned short>(X,Y,xi); break;
		case CV_32SC1: return interp1_<int>(X,Y,xi); break;
		case CV_32FC1: return interp1_<float>(X,Y,xi); break;
		case CV_64FC1: return interp1_<double>(X,Y,xi); break;
	}
}
