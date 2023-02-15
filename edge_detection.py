import numpy as np
import cv2
import imutils
import sys

img = cv2.imread('./test_images/rail_fiducial_green_red.jpg')
edges = cv2.Canny(img, 50, 100)

# cv2.imshow('original',img)
# cv2.imshow('edge detected', edges)
cv2.waitKey()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2,3,0,0.04)
#dilate?
dst = cv2.dilate(dst, None)

#threshold
img[dst>0.01*dst.max()] = [0,0,255]
cv2.imshow('dst', img)
cv2.waitKey()

