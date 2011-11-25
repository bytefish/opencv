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


void save_image(const string filename, const Mat& src, const colormap::ColorMap& colorMap) {
	Mat img = src.clone();
	// make sure it's ok for cv::LUT...
	cvtColor(img, img, CV_GRAY2BGR);
	normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC3);
	// apply the color map
	img = colorMap(img);
	// normalize it to display / save it with highgui tools
	normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC3);
	imwrite(filename, img);
}

int main(int argc, const char *argv[]) {

	//Mat img = imread("image.jpg",0);
	Mat img = Mat::zeros(30, 256, CV_8UC1);
	for(int i = 0; i < 256; i++) {
		Mat roi = Mat(img, Range::all(), Range(i,i+1));
		roi +=i;
	}
	string prefix("colorscale_");
	save_image(prefix + string("jet.jpg"), img, colormap::Jet());
	save_image(prefix + string("blueorange.jpg"), img, colormap::BlueOrange());
	save_image(prefix + string("winter.jpg"), img, colormap::Winter());
	save_image(prefix + string("rainbow.jpg"), img, colormap::Rainbow());
	save_image(prefix + string("ocean.jpg"), img, colormap::Ocean());
	save_image(prefix + string("summer.jpg"), img, colormap::Summer());
	save_image(prefix + string("spring.jpg"), img, colormap::Spring());
	save_image(prefix + string("cool.jpg"), img, colormap::Cool());
	save_image(prefix + string("hsv.jpg"), img, colormap::HSV());
	save_image(prefix + string("pink.jpg"), img, colormap::Pink());
	save_image(prefix + string("hot.jpg"), img, colormap::Hot());
	save_image(prefix + string("mkpj1.jpg"), img, colormap::MKPJ1());
	save_image(prefix + string("mkpj2.jpg"), img, colormap::MKPJ2());

	return 0; // success
}
