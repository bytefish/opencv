#!/usr/bin/python

import cv

kernel = [ [0, 1, 2], [-1, 0, 1], [-2, -1, 0] ]

mat = cv.CreateMat(3,3,cv.CV_32FC1)
for i in range(3):
	for j in range(3):
		mat[i,j] = kernel[i][j]

cv.NamedWindow('filterdemo', cv.CV_WINDOW_AUTOSIZE)
camera = cv.CaptureFromCAM(0)

while 1:
	frame = cv.QueryFrame(camera)
	cv.Filter2D(frame, frame, mat)
	cv.ShowImage('filterdemo', frame)
	key = cv.WaitKey(20)
	if key == 0x1b:
		break

