#include "subspace.hpp"
#include "helper.hpp"

using namespace Eigen;
using namespace cv;

void subspace::LinearDiscriminantAnalysis::compute(const Mat& src, const vector<int>& labels) {
	// assert type
	if((src.type() != CV_32FC1) && (src.type() != CV_64FC1))
		CV_Error(CV_StsBadArg, "src must be a valid matrix float or double matrix");

	Mat data;
	src.convertTo(data, CV_64FC1); // we want to work with double precision!
	// turn into row vector samples
	if(!_dataAsRow)
		transpose(data,data);

	// assert, that labels are valid 32bit signed integer values
	if(labels.size() != data.rows)
		CV_Error(CV_StsBadArg, "labels array must be a valid 1d integer vector of len(src) elements");

	int N = data.rows; //! number of samples
	int D = data.cols; //! dimension of samples
	int C = *max_element(labels.begin(), labels.end()) + 1; //! number of classes

	// within scatter matrix will be singular
	if(N < D)
		cout << "Less instances than feature dimension! Computation will probably fail." << endl;


	// There are atmost (C-1) non-zero eigenvalues!
	if((_num_components > (C-1)) || (_num_components < 1)) {
		_num_components = C-1;
		cout << "num_components set to: " << _num_components << "!" << endl;
	}

	Mat meanTotal = Mat::zeros(1, D, data.type());

	Mat meanClass[C];
	int numClass[C];

	// initialize
	for (int i = 0; i < C; i++) {
		numClass[i] = 0;
		meanClass[i] = Mat::zeros(1, D, data.type()); //! Dx1 image vector
	}

	// calculate class means and total mean
	for (int i = 0; i < N; i++) {
		Mat instance = data.row(i);
		int classIdx = labels[i];
		add(meanTotal, instance, meanTotal);
		add(meanClass[classIdx], instance, meanClass[classIdx]);
		numClass[classIdx]++;
	}

	meanTotal.convertTo(meanTotal, meanTotal.type(), 1.0/N);

	for (int i = 0; i < C; i++)
		meanClass[i].convertTo(meanClass[i], meanClass[i].type(), 1.0/(numClass[i] * 1.0));

	// subtract class means
	for (int i = 0; i < N; i++) {
		int classIdx = labels[i];
		Mat instance = data.row(i);
		subtract(instance, meanClass[classIdx], instance);
	}

	Mat Sw = Mat::zeros(D, D, data.type());
	mulTransposed(data, Sw, true);

	Mat Sb = Mat::zeros(D, D, data.type());
	for (int i = 0; i < C; i++) {
		Mat tmp;
		subtract(meanClass[i], meanTotal, tmp);
		mulTransposed(tmp, tmp, true);
		add(Sb, tmp, Sb);
	}

	// i am only using doubles here, make a templated version to allow MatrixXf
	MatrixXd Swe, Sbe;

	// convert to Eigen Datatypes
	cv2eigen(Sw, Swe);
	cv2eigen(Sb, Sbe);

	// solve generalized eigenvalue problem
	Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(Sbe, Swe);

	// copy and we are done!
	eigen2cv(es.eigenvectors(), _eigenvectors);
	eigen2cv(es.eigenvalues(), _eigenvalues);

	// eigen sorts ascending, so reverse the result
	reverseByRow(_eigenvalues, _eigenvalues);
	reverseByCol(_eigenvectors, _eigenvectors);

	// and now take only the num_components
	_eigenvalues = Mat(_eigenvalues, Range(0,_num_components), Range::all());
	_eigenvectors = Mat(_eigenvectors, Range::all(), Range(0, _num_components));
}

void subspace::LinearDiscriminantAnalysis::project(const Mat& src, Mat& dst) {
	if(_dataAsRow) {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst, CV_GEMM_A_T + CV_GEMM_B_T);
	} else {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst,  CV_GEMM_A_T);
	}
}

void subspace::LinearDiscriminantAnalysis::reconstruct(const Mat& src, Mat& dst) {
	if(_dataAsRow) {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst, CV_GEMM_B_T);
	} else {
		gemm(_eigenvectors, src, 1.0, Mat(), 0, dst);
	}
}

Mat subspace::LinearDiscriminantAnalysis::project(const Mat& src) {
	Mat dst;
	project(src, dst);
	return dst;
}

Mat subspace::LinearDiscriminantAnalysis::reconstruct(const Mat& src) {
	Mat dst;
	reconstruct(src, dst);
	return dst;
}
