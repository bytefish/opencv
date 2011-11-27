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
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include <string>
#include <cv.h>
#include <highgui.h>
#include "colormap.hpp"
//#include "helper.hpp"

using namespace cv;
using namespace std;


void save_image(const string filename, const Mat& src, const colormap::ColorMap& cm) {
	Mat img = cm(src);
	normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC3);
	imwrite(filename, img);
}

int main(int argc, const char *argv[]) {
	//	simple example with an image
//	Mat img0 = imread("01.jpg",0);
//	colormap::Hot cm;
//	imshow("cm", cm(img0));
//	waitKey(0);

	Mat img1 = Mat::zeros(30, 256, CV_8UC1);
	for(int i = 0; i < 256; i++) {
		Mat roi = Mat(img1, Range::all(), Range(i,i+1));
		roi += i;
	}
	string prefix("colorscale_");
	save_image(prefix + string("autumn.jpg"), img1, colormap::Autumn());
	save_image(prefix + string("bone.jpg"), img1, colormap::Bone());
	save_image(prefix + string("jet.jpg"), img1, colormap::Jet());
	save_image(prefix + string("winter.jpg"), img1, colormap::Winter());
	save_image(prefix + string("rainbow.jpg"), img1, colormap::Rainbow());
	save_image(prefix + string("ocean.jpg"), img1, colormap::Ocean());
	save_image(prefix + string("summer.jpg"), img1, colormap::Summer());
	save_image(prefix + string("spring.jpg"), img1, colormap::Spring());
	save_image(prefix + string("cool.jpg"), img1, colormap::Cool());
	save_image(prefix + string("hsv.jpg"), img1, colormap::HSV());
	save_image(prefix + string("pink.jpg"), img1, colormap::Pink());
	save_image(prefix + string("hot.jpg"), img1, colormap::Hot());
	save_image(prefix + string("mkpj1.jpg"), img1, colormap::MKPJ1());
	save_image(prefix + string("mkpj2.jpg"), img1, colormap::MKPJ2());

	return 0; // success
}
