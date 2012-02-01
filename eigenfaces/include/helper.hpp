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

#ifndef HELPER_HPP_
#define HELPER_HPP_

#include "cv.h"
#include <vector>

#include <string>
#include <sstream>

using namespace std;

namespace cv
{
//! matlab equivalent num2str
string num2str(int i);
//! templated sort operator
template<typename _Tp>
class SortByFirstAscending_;
//! descending sort operator
template<typename _Tp>
class SortByFirstDescending_;
//! sorts a matrix by column for given indices
void sortMatrixByColumn(const Mat& src, Mat& dst, vector<int> sorted_indices);
//! sorts a matrix by column for given indices
Mat sortMatrixByColumn(const Mat& src, vector<int> sorted_indices);
//! sorts a matrix by row for given indices
void sortMatrixByRow(const Mat& src, Mat& dst, vector<int> sorted_indices);
//! sorts a matrix by row for given indices
Mat sortMatrixByRow(const Mat& src, vector<int> sorted_indices);
//! turns a vector of matrices into a row matrix
Mat asRowMatrix(const vector<Mat>& src);
//! turns a vector of matrices into a column matrix
Mat asColumnMatrix(const vector<Mat>& src);
//! turns a one-channel matrix into a grayscale representation
Mat toGrayscale(const Mat& src);
//! transposes a matrix
Mat transpose(const Mat& src);
//! performs a matlab/numpy equivalent argsort (only 1 channel matrices supported)
template<typename _Tp>
vector<int> argsort_(const Mat& src, bool sortAscending=true);
//! performs a matlab/numpy equivalent argsort (only 1 channel matrices supported)
vector<int> argsort(const Mat& src, bool sortAscending=true);
}

#endif
