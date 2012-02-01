#include "helper.hpp"
#include "eigenfaces.hpp"

Eigenfaces::Eigenfaces(const Mat& src, const vector<int>& labels, int num_components, bool dataAsRow) {
	_num_components = num_components;
	_dataAsRow = dataAsRow;
	// compute the eigenfaces
	compute(src, labels);
}

Eigenfaces::Eigenfaces(const vector<Mat>& src, const vector<int>& labels, int num_components, bool dataAsRow) {
	_num_components = num_components;
	_dataAsRow = dataAsRow;
	// compute the eigenfaces
	compute(src, labels);
}

void Eigenfaces::compute(const Mat& src, const vector<int>& labels) {
	// observations in row
	Mat data = _dataAsRow ? src : transpose(src);
	// number of samples
	int n = data.rows;
	// dimensionality of data
	int d = data.cols;
	// assert there are as much samples as labels
	if(n != labels.size())
		CV_Error(CV_StsBadArg, "The number of samples must equal the number of labels!");
	// clip number of components to be valid
	_num_components = max(1, min(_num_components, n));
	// perform the PCA
	PCA pca(data,
			Mat(),
			CV_PCA_DATA_AS_ROW,
			_num_components);
	// set the data
	_mean = _dataAsRow ? pca.mean.reshape(1,1) : pca.mean.reshape(1, pca.mean.total()); // store the mean vector
	_eigenvalues = pca.eigenvalues.clone(); // store the eigenvectors
	_eigenvectors = transpose(pca.eigenvectors); // OpenCV stores the Eigenvectors by row (??)
	_labels = vector<int>(labels); // store labels for projections
	// projections
	for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
		this->_projections.push_back(project(_dataAsRow ? src.row(sampleIdx) : src.col(sampleIdx)));
	}
}

void Eigenfaces::compute(const vector<Mat>& src, const vector<int>& labels) {
	compute(_dataAsRow ? asRowMatrix(src) : asColumnMatrix(src), labels);
}

int Eigenfaces::predict(const Mat& src) {
	Mat q = project(_dataAsRow ? src.reshape(1,1) :	src.reshape(1, src.total()));
	// find 1-nearest neighbor
	double minDist = numeric_limits<double>::max();
	int minClass = -1;
	for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		double dist = norm(_projections[sampleIdx], q, NORM_L2);
		if(dist < minDist) {
			minDist = dist;
			minClass = _labels[sampleIdx];
		}
	}
	return minClass;
}

Mat Eigenfaces::project(const Mat& src) {
	Mat X,Y;
	int n = _dataAsRow ? src.rows : src.cols;
	// center data
	subtract(_dataAsRow ? src : transpose(src),
			repeat(_mean.reshape(1,1), n, 1),
			X,
			Mat(),
			_eigenvectors.type());
	// Y = (X-mean)*W
	gemm(X, _eigenvectors, 1.0, Mat(), 0.0, Y);
	return _dataAsRow ? Y : transpose(Y);
}

Mat Eigenfaces::reconstruct(const Mat& src) {
	Mat X;
	int n = _dataAsRow ? src.rows : src.cols;
	// X = Y*W'+mean
	gemm(_dataAsRow ? src : transpose(src),
			_eigenvectors,
			1.0,
			repeat(_mean.reshape(1,1), n, 1),
			1.0,
			X,
			GEMM_2_T);
	return _dataAsRow ? X : transpose(X);
}
