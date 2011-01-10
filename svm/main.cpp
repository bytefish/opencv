#include <iostream>
#include <math.h>
#include <string>
#include "cv.h"
#include "ml.h"
#include "highgui.h"

using namespace cv;
using namespace std;

#define numTrainingPoints 2000
#define numTestPoints 200
#define size 200
#define eq 0

#define plotSupportVectors

// tpr
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for(int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i,0);
		float a = actual.at<float>(i,0);
		if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
			t++;
		} else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}

// plot data and class
void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
	cv::Mat plot(size, size, CV_8UC3);
	plot.setTo(CV_RGB(255,255,255));
	for(int i = 0; i < data.rows; i++) {

		float x = data.at<float>(i,0) * size;
		float y = data.at<float>(i,1) * size;

		if(classes.at<float>(i, 0) > 0) {
			cv::circle(plot, Point(x,y), 2, CV_RGB(255,0,0),1);
		} else {
			cv::circle(plot, Point(x,y), 2, CV_RGB(0,255,0),1);
		}
	}
	cv::imshow(name, plot);
}

// function to learn
int f(float x, float y, int equation) {
	switch(equation) {
	case 0:
		return y > sin(x*10) ? -1 : 1;
		break;
	case 1:
		return y > cos(x * 10) ? -1 : 1;
		break;
	case 2:
		return y > 2*x ? -1 : 1;
		break;
	case 3:
		return y > tan(x*10) ? -1 : 1;
		break;
	default:
		return y > cos(x*10) ? -1 : 1;
	}
}

// label data with equation
cv::Mat labelData(cv::Mat points, int equation) {
	cv::Mat labels(points.rows, 1, CV_32FC1);
	for(int i = 0; i < points.rows; i++) {
			 float x = points.at<float>(i,0);
			 float y = points.at<float>(i,1);
			 labels.at<float>(i, 0) = f(x, y, equation);
		}
	return labels;
}

void svm(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
	CvSVMParams param = CvSVMParams();

	param.svm_type = CvSVM::C_SVC;
	param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
	param.degree = 0; // for poly
	param.gamma = 20; // for poly/rbf/sigmoid
	param.coef0 = 0; // for poly/sigmoid

	param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR

	param.class_weights = NULL; // for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;

	// SVM training (use train auto for OpenCV>=2.0)
	CvSVM svm(trainingData, trainingClasses, cv::Mat(), cv::Mat(), param);

	cv::Mat predicted(testClasses.rows, 1, CV_32F);

	for(int i = 0; i < testData.rows; i++) {
		cv::Mat sample = testData.row(i);

		float x = sample.at<float>(0,0);
		float y = sample.at<float>(0,1);

		predicted.at<float>(i, 0) = svm.predict(sample);
	}

	cout << "TPR_{SVM} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions SVM");

	// plot support vectors
#ifdef plotSupportVectors
	cv::Mat plot_sv(size, size, CV_8UC3);
	plot_sv.setTo(CV_RGB(255,255,255));

	int svec_count = svm.get_support_vector_count();
	for(int vecNum = 0; vecNum < svec_count; vecNum++) {
		const float* vec = svm.get_support_vector(vecNum);
		cv::circle(plot_sv, Point(vec[0]*size, vec[1]*size), 3 , CV_RGB(0, 0, 0));
	}
	cv::imshow("Support Vectors", plot_sv);
#endif

}

void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

	cv::Mat layers = cv::Mat(4, 1, CV_32SC1);

	layers.row(0) = cv::Scalar(2);
	layers.row(1) = cv::Scalar(10);
	layers.row(2) = cv::Scalar(15);
	layers.row(3) = cv::Scalar(1);

	CvANN_MLP mlp;
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	mlp.create(layers);

	// train
	mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);

	cv::Mat response(1, 1, CV_32FC1);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for(int i = 0; i < testData.rows; i++) {
		cv::Mat response(1, 1, CV_32FC1);
		cv::Mat sample = testData.row(i);

		mlp.predict(sample, response);
		predicted.at<float>(i,0) = response.at<float>(0,0);

	}

	cout << "TPR_{MLP} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions Backpropagation");
}

int main() {

	cv::Mat trainingData(numTrainingPoints, 2, CV_32FC1);
	cv::Mat testData(numTestPoints, 2, CV_32FC1);

	cv::randu(trainingData,0,1);
	cv::randu(testData,0,1);

	cv::Mat trainingClasses = labelData(trainingData, eq);
	cv::Mat testClasses = labelData(testData, eq);

	plot_binary(trainingData, trainingClasses, "Training Data");
	plot_binary(testData, testClasses, "Test Data");

	svm(trainingData, trainingClasses, testData, testClasses);
	mlp(trainingData, trainingClasses, testData, testClasses);

	cv::waitKey();

	return 0;
}

