#ifndef EIGENFACES_H_
#define EIGENFACES_H_

#include <cv.h>
#include <limits.h>
#include <vector>

using namespace std;
using namespace cv;
class Eigenfaces {

public:
	Eigenfaces(const Mat& data, const vector<int>& classes, int components);
	virtual ~Eigenfaces();
	void compute(const Mat& data, const vector<int>& classes, int components);
	int predict(const Mat& instance);
	Mat eigenvectors() { return pca.eigenvectors; }
	Mat eigenvalues() { return pca.eigenvalues; }

private:
	PCA pca;
	vector<Mat> db;
	vector<int> classes;
};

#endif /* EIGENFACES_H_ */
