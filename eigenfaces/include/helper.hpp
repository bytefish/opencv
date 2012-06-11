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

#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include "opencv2/opencv.hpp"
#include <vector>
#include <set>

using namespace std;

// Removes duplicate elements in a given vector.
template<typename _Tp>
inline vector<_Tp> remove_dups(const vector<_Tp>& src) {
    typedef typename set<_Tp>::const_iterator constSetIterator;
    typedef typename vector<_Tp>::const_iterator constVecIterator;
    set<_Tp> set_elems;
    for (constVecIterator it = src.begin(); it != src.end(); ++it)
        set_elems.insert(*it);
    vector<_Tp> elems;
    for (constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
        elems.push_back(*it);
    return elems;
}

// The namespace cv provides opencv related helper functions.
namespace cv {

// Checks if a given matrix is symmetric, with an epsilon for floating point
// matrices (1E-16 by default).
//
//      Mat mSymmetric = (Mat_<double>(2,2) << 1, 2, 2, 1);
//      Mat mNonSymmetric = (Mat_<double>(2,2) << 1, 2, 3, 4);
//      bool symmetric = isSymmetric(mSymmetric); // true
//      bool not_symmetric = isSymmetric(mNonSymmetric); // false
//
bool isSymmetric(const Mat& src, double eps = 1E-16);

// Sorts a 1D Matrix by given sort order and returns the sorted indices.
// This is just a wrapper to simplify cv::sortIdx:
//
//      Mat mNotSorted = (Mat_<double>(1,4) << 1.0, 0.0, 3.0, -1.0);
//      // to sort the vector use
//      Mat sorted_indices = cv::argsort(mNotSorted, true);
//      // make a conversion to vector<int>
//      vector<int> sorted_indices = cv::argsort(mNotSorted, true);
//
vector<int> argsort(const Mat& src, bool ascending = true);

// Sorts a given matrix src by column for given indices.
//
// Note: create is called on dst.
void sortMatrixColumnsByIndices(const Mat& src, const vector<int>& indices, Mat& dst);

// Sorts a given matrix src by row for given indices.
Mat sortMatrixColumnsByIndices(const Mat& src, const vector<int>& indices);

// Sorts a given matrix src by row for given indices.
//
// Note: create is called on dst.
void sortMatrixRowsByIndices(const Mat& src, const vector<int>& indices, Mat& dst);

// Sorts a given matrix src by row for given indices.
Mat sortMatrixRowsByIndices(const Mat& src, const vector<int>& indices);

// Turns a vector of matrices into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha=1, double beta=0);

// Turns a given matrix into its grayscale representation.
Mat toGrayscale(const Mat& src, int dtype = CV_8UC1);

// Transposes a matrix.
Mat transpose(const Mat& src);

}

#endif
