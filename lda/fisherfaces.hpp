#ifndef __FISHERFACES_HPP__
#define __FISHERFACES_HPP__

#include "cv.h"
#include <eigen3/Eigen/SVD>

using namespace cv;
using namespace std;

namespace subspace {

/**
 * \class LinearDiscriminantAnalysis
 * \brief Performs a Multiclass Discriminant Analysis for given data.
 */
class Fisherfaces {

private:
	bool _dataAsRow;
	int _num_components;
	Mat _eigenvectors;
	Mat _eigenvalues;
	vector<Mat> _projections;
	vector<int> _labels;

public:

	Fisherfaces() :
		_num_components(0),
		_dataAsRow(true) {};

	Fisherfaces(const Mat& src,
			const vector<int>& labels,
			int num_components = 0,
			bool dataAsRow = true) :
				_num_components(num_components),
				_dataAsRow(dataAsRow)
	{
		this->compute(src, labels); //! compute eigenvectors and eigenvalues
	}

	~Fisherfaces() {}

	//! compute the discriminants for data in src and labels
	void compute(const Mat& src, const vector<int>& labels);

	//! project
	void project(const Mat& src, Mat& dst);
	Mat project(const Mat& src);

	//! reconstruct
	void reconstruct(const Mat& src, Mat& dst);
	Mat reconstruct(const Mat& src);

	//! returns a const reference to the eigenvectors of this LDA
	const Mat& eigenvectors() const { return _eigenvectors; };

	//! returns a const reference to the eigenvalues of this LDA
	const Mat& eigenvalues() const { return _eigenvalues; }

	//! return nearest neighbor to query
	int predict(const Mat& src);
};
}
#endif
