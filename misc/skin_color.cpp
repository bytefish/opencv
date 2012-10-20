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
 *   See <http:www.opensource.org/licenses/bsd-license>
 */
 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

// This snippet implements common Skin Color Thresholding rules taken from:
//
//  Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See. RGB-H-CbCr Skin Colour Model for Human Face Detection.
//  (Online available at http://pesona.mmu.edu.my/~johnsee/research/papers/files/rgbhcbcr_m2usic06.pdf)

using namespace std;
using namespace cv;

bool R1(int R, int G, int B) {
    bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
    return (e1||e2);
}
 
bool R2(float Y, float Cr, float Cb) {
    bool e3 = Cr <= 1.5862*Cb+20;
    bool e4 = Cr >= 0.3448*Cb+76.2069;
    bool e5 = Cr >= -4.5652*Cb+234.5652;
    bool e6 = Cr <= -1.15*Cb+301.75;
    bool e7 = Cr <= -2.2857*Cb+432.85;
    return e3 && e4 && e5 && e6 && e7;
}
 
bool R3(float H, float S, float V) {
    return (H<25) || (H > 230);
}
 
Mat ThresholdSkin(const Mat &src) {
    // Allocate the result matrix
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    // We operate in YCrCb and HSV:     
    Mat src_ycrcb, src_hsv;
    // OpenCV scales the YCrCb components, so that they
    // cover the whole value range of [0,255], so there's
    // no need to scale the values:
    cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision:
    src.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
    // And then scale between [0,255] for the rules in the paper
    // to apply. This uses normalize with CV_32FC3, which may fail
    // on older OpenCV versions. If so, you probably want to split
    // the channels first and call normalize independently on each
    // channel:
    normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);
    // Iterate over the data:
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            // Get the pixel in BGR space: 
            Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // And apply RGB rule:
            bool a = R1(R,G,B);
            // Get the pixel in YCrCB space:
            Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
            int Y = pix_ycrcb.val[0];
            int Cr = pix_ycrcb.val[1];
            int Cb = pix_ycrcb.val[2];
            // And apply the YCrCB rule:
            bool b = R2(Y,Cr,Cb);
            // Get the pixel in HSV space:
            Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            float S = pix_hsv.val[1];
            float V = pix_hsv.val[2];
            // And apply the HSV rule:
            bool c = R3(H,S,V);
            // If not skin, then black 
            if(a && b && c) {
                dst.at<unsigned char>(i,j) = 255;
            }
        }
    }
    return dst;
}
 
 
int main(int argc, const char *argv[]) {
    // Get filename to the source image:
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <image.ext>" << endl;
        exit(1);
    }
    // Load image & get skin proportions:
    Mat image = imread(argv[1]);
    // Put a little Gaussian blur on:
    blur(image, image, Size(5,5));
    // Filter for skin:
    Mat skin = ThresholdSkin(image);
    // And finally perform a little dilation and erosion, I'll just
    // steal from:
    //
    //      http://docs.opencv.org/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
    //

    int dilation_size = 5;
    int erosion_size = 5;

    dilate(skin, skin, getStructuringElement(
            MORPH_RECT,
            Size(2*dilation_size+1, 2*dilation_size+1),
            Point(dilation_size, dilation_size)));

    erode(skin, skin, getStructuringElement(
            MORPH_RECT,
            Size(2*erosion_size+1, 2*erosion_size+1),
            Point(erosion_size, erosion_size)));

    // The Results:
    namedWindow("original");
    namedWindow("skin");
 
    imshow("original", image);
    imshow("skin", skin);

    // Show the images:
    waitKey(0);
    
    // Success!
    return 0;
}
