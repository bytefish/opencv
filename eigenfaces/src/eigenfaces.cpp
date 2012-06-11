#include "helper.hpp"
#include "eigenfaces.hpp"

void Eigenfaces::compute(const vector<Mat>& src, const vector<int>& labels) {
    if(src.size() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(CV_StsUnsupportedFormat, error_message);
    }
    // observations in row
    Mat data = asRowMatrix(src, CV_64FC1);
    // number of samples
    int n = data.rows;
    // dimensionality of data
    int d = data.cols;
    // assert there are as much samples as labels
    if(n != labels.size()) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", n, labels.size());
        CV_Error(CV_StsBadArg, error_message);
    }
    // clip number of components to be valid
    if((_num_components <= 0) || (_num_components > n))
        _num_components = n;
    // perform the PCA
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, _num_components);
    // copy the PCA results
    _mean = pca.mean.reshape(1,1); // store the mean vector
    _eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
    _eigenvectors = transpose(pca.eigenvectors); // eigenvectors by column
    _labels = labels; // store labels for prediction
    // save projections
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = project(data.row(sampleIdx).clone());
        this->_projections.push_back(p);
    }
}


void Eigenfaces::predict(const Mat& src, int &minClass, double &minDist) {
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This cv::Eigenfaces model is not computed yet. Did you call cv::Eigenfaces::train?";
        CV_Error(CV_StsError, error_message);
    } else if(_eigenvectors.rows != src.total()) {
        // check data alignment just for clearer exception messages
        string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(CV_StsError, error_message);
    }
    // project into PCA subspace
    Mat q = project(src.reshape(1,1));
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels[sampleIdx];
        }
    }
}

int Eigenfaces::predict(const Mat& src) {
    int label;
    double dummy;
    predict(src, label, dummy);
    return label;
}

Mat Eigenfaces::project(const Mat& src) {
    Mat W = _eigenvectors;
    Mat mean = _mean;
    // get number of samples and dimension
    int n = src.rows;
    int d = src.cols;
    // make sure the data has the correct shape
    if(W.rows != d) {
        string error_message = format("Wrong shapes for given matrices. Was size(src) = (%d,%d), size(W) = (%d,%d).", src.rows, src.cols, W.rows, W.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    // make sure mean is correct if not empty
    if(!mean.empty() && (mean.total() != d)) {
        string error_message = format("Wrong mean shape for the given data matrix. Expected %d, but was %d.", d, mean.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    // create temporary matrices
    Mat X, Y;
    // make sure you operate on correct type
    src.convertTo(X, W.type());
    // safe to do, because of above assertion
    // safe to do, because of above assertion
    if(!mean.empty()) {
        for(int i=0; i<n; i++) {
            Mat r_i = X.row(i);
            subtract(r_i, mean.reshape(1,1), r_i);
        }
    }
    // finally calculate projection as Y = (X-mean)*W
    gemm(X, W, 1.0, Mat(), 0.0, Y);
    return Y;
}

Mat Eigenfaces::reconstruct(const Mat& src) {
    Mat W = _eigenvectors;
    Mat mean = _mean;
    // get number of samples and dimension
    int n = src.rows;
    int d = src.cols;
    // make sure the data has the correct shape
    if(W.cols != d) {
        string error_message = format("Wrong shapes for given matrices. Was size(src) = (%d,%d), size(W) = (%d,%d).", src.rows, src.cols, W.rows, W.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    // make sure mean is correct if not empty
    if(!mean.empty() && (mean.total() != W.rows)) {
        string error_message = format("Wrong mean shape for the given eigenvector matrix. Expected %d, but was %d.", W.cols, mean.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    // initalize temporary matrices
    Mat X, Y;
    // copy data & make sure we are using the correct type
    src.convertTo(Y, W.type());
    // calculate the reconstruction
    gemm(Y, W, 1.0, Mat(), 0.0, X, GEMM_2_T);
    // safe to do because of above assertion
    if(!mean.empty()) {
        for(int i=0; i<n; i++) {
            Mat r_i = X.row(i);
            add(r_i, mean.reshape(1,1), r_i);
        }
    }
    return X;
}
