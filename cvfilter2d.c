#include <cv.h>
#include <highgui.h>
 
int main(int argc, char** argv) {
    CvCapture* capture = cvCaptureFromCAM(0);
 
    /*
    // Gaussian kernel
    float kernel[] = {1.0/16, 2.0/16, 1.0/16,
        2.0/16, 4.0/16, 2.0/16,
        1.0/16, 2.0/16, 1.0/16};
    */
 
    float kernel[] = {0, 1, 2,
        -1, 0, 1,
        -2, -1, 0};
 
    CvMat filter = cvMat(
            3,
            3,
            CV_32FC1,
            kernel);
 
    cvNamedWindow("cam", CV_WINDOW_AUTOSIZE);
    cvQueryFrame(capture);
 
    IplImage* frame = 0;
    IplImage* dst = 0;
 
    for(;;) {
        frame = cvQueryFrame(capture);
        dst = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
        cvCvtColor(frame, dst, CV_BGR2GRAY);
 
        cvFilter2D(dst, dst, &filter, cvPoint(-1,-1) );
 
        cvShowImage("cam", dst);
        cvReleaseImage(&dst);
        int c = cvWaitKey(10);
        if((char) c == 27)
            break;
    }
    cvReleaseCapture(&capture);
    cvDestroyWindow("cam");
 
    return 0;
}
