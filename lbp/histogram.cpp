#include "histogram.hpp"
#include <vector>

template <typename _Tp>
void lbp::histogram_(const Mat& src, Mat& hist, int numPatterns) {
	hist = Mat::zeros(1, numPatterns, CV_32SC1);
	for(int i = 0; i < src.rows; i++) {
		for(int j = 0; j < src.cols; j++) {
			int bin = src.at<_Tp>(i,j);
			hist.at<int>(0,bin) += 1;
		}
	}
}

template <typename _Tp>
double lbp::chi_square_(const Mat& histogram0, const Mat& histogram1) {
	if(histogram0.type() != histogram1.type())
			CV_Error(CV_StsBadArg, "Histograms must be of equal type.");
	if(histogram0.rows != 1 || histogram0.rows != histogram1.rows || histogram0.cols != histogram1.cols)
			CV_Error(CV_StsBadArg, "Histograms must be of equal dimension.");
	double result = 0.0;
	for(int i=0; i < histogram0.cols; i++) {
		double a = histogram0.at<_Tp>(0,i) - histogram1.at<_Tp>(0,i);
		double b = histogram0.at<_Tp>(0,i) + histogram1.at<_Tp>(0,i);
		if(abs(b) > numeric_limits<double>::epsilon()) {
			result+=(a*a)/b;
		}
	}
	return result;
}


void lbp::spatial_histogram(const Mat& src, Mat& hist, int numPatterns, const Size& window, int overlap) {
	int width = src.cols;
	int height = src.rows;
	vector<Mat> histograms;
	for(int x=0; x < width - window.width; x+=(window.width-overlap)) {
		for(int y=0; y < height-window.height; y+=(window.height-overlap)) {
			Mat cell = Mat(src, Rect(x,y,window.width, window.height));
			histograms.push_back(histogram(cell, numPatterns));
		}
	}
	hist.create(1, histograms.size()*numPatterns, CV_32SC1);
	// i know this is a bit lame now... feel free to make this a bit more efficient...
	for(int histIdx=0; histIdx < histograms.size(); histIdx++) {
		for(int valIdx = 0; valIdx < numPatterns; valIdx++) {
			int y = histIdx*numPatterns+valIdx;
			hist.at<int>(0,y) = histograms[histIdx].at<int>(valIdx);
		}
	}
}

// wrappers
void lbp::histogram(const Mat& src, Mat& hist, int numPatterns) {
	switch(src.type()) {
		case CV_8SC1: histogram_<char>(src, hist, numPatterns); break;
		case CV_8UC1: histogram_<unsigned char>(src, hist, numPatterns); break;
		case CV_16SC1: histogram_<short>(src, hist, numPatterns); break;
		case CV_16UC1: histogram_<unsigned short>(src, hist, numPatterns); break;
		case CV_32SC1: histogram_<int>(src, hist, numPatterns); break;
	}
}

double lbp::chi_square(const Mat& histogram0, const Mat& histogram1) {
	switch(histogram0.type()) {
		case CV_8SC1: return chi_square_<char>(histogram0,histogram1); break;
		case CV_8UC1: return chi_square_<unsigned char>(histogram0,histogram1); break;
		case CV_16SC1: return chi_square_<short>(histogram0, histogram1); break;
		case CV_16UC1: return chi_square_<unsigned short>(histogram0,histogram1); break;
		case CV_32SC1: return chi_square_<int>(histogram0,histogram1); break;
	}
}

void lbp::spatial_histogram(const Mat& src, Mat& dst, int numPatterns, int gridx, int gridy, int overlap) {
	int width = static_cast<int>(floor(src.cols/gridx));
	int height = static_cast<int>(floor(src.rows / gridy));
	spatial_histogram(src, dst, numPatterns, Size_<int>(width, height), overlap);
}

// Mat return type functions
Mat lbp::histogram(const Mat& src, int numPatterns) {
	Mat hist;
	histogram(src, hist, numPatterns);
	return hist;
}


Mat lbp::spatial_histogram(const Mat& src, int numPatterns, const Size& window, int overlap) {
	Mat hist;
	spatial_histogram(src, hist, numPatterns, window, overlap);
	return hist;
}


Mat lbp::spatial_histogram(const Mat& src, int numPatterns, int gridx, int gridy, int overlap) {
	Mat hist;
	spatial_histogram(src, hist, numPatterns, gridx, gridy);
	return hist;
}
