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
#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <vector>
#include <cv.h>

namespace cv {

//! MATLAB-equivalent: linspace
Mat linspace(float x0, float x1, int n);

//! ascending sort operator
template<typename _Tp>
class SortByFirstAscending_;

//! descending sort operator
template<typename _Tp>
class SortByFirstDescending_;

//! works like MATLAB/NumPy-equivalent argsort (only 1D sorting supported)
template<typename _Tp>
vector<int> argsort_(const Mat& src, bool asc=true);
vector<int> argsort(const Mat& src, bool asc=true);

//! sorts src by column for given indices
void sortMatrixByColumn(const Mat& src, Mat& dst, vector<int> sorted_indices);
Mat sortMatrixByColumn(const Mat& src, vector<int> sorted_indices);

//! sorts src by row for given indices
void sortMatrixByRow(const Mat& src, Mat& dst, vector<int> sorted_indices);
Mat sortMatrixByRow(const Mat& src, vector<int> sorted_indices);

//! get difference matrix
void diff(const Mat& src, Mat& dst);
Mat diff(const Mat& src);

//! normalize each channel of a multi-channel matrix
void normalize_m(const Mat& src, const Mat& dst, double alpha, double beta, int norm_type, int dtype, const Mat& mask);

//! perform linear interpolation on two interpolation table columns
Mat interp1(const Mat& X, const Mat& Y, const Mat& xi);


}
#endif
