import cv2
import numpy as np 
from collections import deque
import imutils

def empty(a):
    pass

kernel = np.ones((5,5), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(4,480)
cap.set(10,80)

cv2.namedWindow("Trackbars")
cv2.resizeWindow('Trackbars', (500,400))

cv2.createTrackbar("Hue Min", "Trackbars", 113, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 144, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 142, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 248, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 73, 83, empty)
cv2.createTrackbar("Val Max", "Trackbars", 240, 255, empty)

pts = deque(maxlen = 1024)


while True:
	success, cam = cap.read()
	cam = cv2.flip(cam, 1)
	cv2.putText(cam, "Air-Canvas", (10,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
	imgHSV = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)

	h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
	h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")    
	sat_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
	sat_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
	val_min = cv2.getTrackbarPos("Val Min", "Trackbars")
	val_max = cv2.getTrackbarPos("Val Max", "Trackbars")

	mask = cv2.inRange(imgHSV, np.array([h_min, sat_min, val_min]), np.array([h_max, sat_max, val_max]))
	mask = cv2.erode(cv2.GaussianBlur(mask,(11,11),0), None, iterations = 2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.dilate(mask, kernel, iterations = 2)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None


	if len(cnts) > 0:
		c = max(cnts, key = cv2.contourArea)
		((x,y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M['m10']/M['m00']), int(M['m01']/ M['m00']))

		if radius > 10:

			cv2.circle(cam, (int(x), int(y)), int(radius), (0,255,255), 2)
			cv2.circle(cam, center, 5, (0,0,255), -1)

	pts.appendleft(center)

	for i in range(1,len(pts)):
		if pts[i - 1] is None or pts[i] is None:
			continue

		cv2.line(cam, pts[i - 1], pts[i], (0,0,255), 2)
		print(pts)
	cv2.imshow("Mask", mask)
	cv2.imshow('Contour',cam)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

























