#include <cv.h>
#include <highgui.h>
#include "subspace.hpp"

using namespace cv;
using namespace std;

int main(int argc, const char *argv[]) {
	// example from: http://www.bytefish.de/wiki/pca_lda_with_gnu_octave
	double d[11][2] = {
			{2, 3},
			{3, 4},
			{4, 5},
			{5, 6},
			{5, 7},
			{2, 1},
			{3, 2},
			{4, 2},
			{4, 3},
			{6, 4},
			{7, 6}};
	int c[11] = {0,0,0,0,0,1,1,1,1,1,1};
	// convert into OpenCV representation
	Mat _data = Mat(11, 2, CV_64FC1, d).clone();
	vector<int> _classes(c, c + sizeof(c) / sizeof(int));
	// perform the lda
	subspace::LinearDiscriminantAnalysis lda(_data, _classes);
	// GNU Octave finds the following eigenvalues
	//octave> d
	//d =
	//	 1.5195e+00
	//   6.5052e-18
	cout << lda.eigenvalues() << endl;
	// Eigen outputs:
	// [1.519536390756363; 9.980626757982641e-19]
	return 0;
}
