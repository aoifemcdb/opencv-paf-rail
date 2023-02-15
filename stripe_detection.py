import numpy as np
import cv2
import imutils
import sys
import matplotlib.pyplot as plt

# obtain binary image through Otsu thresholding, detect horizontal lines using morphology, draw detected lines onto
# mask them perform additional morphological operations to combine stripes into a single contour

img = cv2.imread('./test_images/rail_fiducial_background.jpg')
mask = np.zeros(img.shape, dtype = np.uint8)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayscale', gray)
# cv2.waitKey()
thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

#threshold
black = (0, 0, 0)
white = (255, 255, 255)

# mask_white = cv2.inRange(gray, white, white)
# mask_black = cv2.inRange(gray, black, black)
# mask_final = mask_white + mask_black

# result = cv2.bitwise_and(img, img, mask=mask_final)
# cv2.imshow('white', mask_white)
# cv2.imshow('black', mask_black)
# cv2.imshow('thresh', mask_final)
# cv2.waitKey()

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(mask, [c], -1, (255,255,255), 5)

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=2)

x,y,w,h = cv2.boundingRect(opening)
ROI = img[y:y+h, x:x+w]



cv2.imshow('thresh', thresh)
cv2.imshow('mask', mask)
cv2.imshow('opening', opening)
cv2.imshow('ROI', ROI)
cv2.waitKey()