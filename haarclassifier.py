#!/usr/bin/python
import cv

cv.NamedWindow("FaceDetect", 1)

camera = cv.CaptureFromCAM(0)
cv.SetCaptureProperty(camera, cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cv.SetCaptureProperty(camera, cv.CV_CAP_PROP_FRAME_HEIGHT, 240) 

green = cv.RGB(0,255,0)

storage = cv.CreateMemStorage(0)
haar_cascade = cv.Load("/usr/local/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")

min_size = (20,20)
scale_factor = 1.2
neighbors = 2
flags = cv.CV_HAAR_DO_CANNY_PRUNING

while 1:
	frame = cv.QueryFrame(camera)
	faces = cv.HaarDetectObjects(frame, haar_cascade, storage, scale_factor, neighbors, flags)
	
	for ((x, y, w, h), n) in faces:
		cx = int(x + (w/2))
		cy = int(y + (h/2))
		# draw a fancy crosshair
		cv.Line(frame, (cx-5, cy), (cx+5, cy), green, 1, cv.CV_AA, 0)
		cv.Line(frame, (cx, cy-5), (cx, cy+5), green, 1, cv.CV_AA, 0)
		# or draw a rectangle
		#cv.Rectangle(frame, (x,y), (x + w, y + h), green, 1, cv.CV_AA, 0)

	
	cv.ShowImage("FaceDetect", frame)
	
	key = cv.WaitKey(20)
	if key == 0x1b:
		break
