#include "Eigenfaces.h"

Eigenfaces::Eigenfaces(const Mat& data, const vector<int>& classes, int components) {
	compute(data, classes, components);
}

void Eigenfaces::compute(const Mat& data, const vector<int>& classes, int components) {
	int numSamples = data.cols;
	PCA pca(data, Mat(),CV_PCA_DATA_AS_COL, components);

	for(int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
		this->db.push_back(pca.project(data.col(sampleIdx)));
	}

	this->pca = pca;
	this->classes = classes;
}

Eigenfaces::~Eigenfaces() {
	// TODO Auto-generated destructor stub
}

int Eigenfaces::predict(const Mat& instance) {
	// project sample
	Mat projection = pca.project(instance);
	// find 1-nearest neighbor
	double minDist = numeric_limits<double>::max();
	int minClass = -1;
	for(int sampleIdx = 0; sampleIdx < db.size(); sampleIdx++) {
		double dist = norm(db[sampleIdx], projection, NORM_L2);
		if(dist < minDist) {
			minDist = dist;
			minClass = this->classes[sampleIdx];
		}
	}
	return minClass;
}
