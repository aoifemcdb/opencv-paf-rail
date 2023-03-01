from __future__ import print_function
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

def order_points(pts):
	# sort points based on x-coord
	x_sorted = pts[np.argsort(pts[:,0]), :]

	# get right and left-most points from sorted points
	left_most = x_sorted[:2, :]
	right_most = x_sorted[2:, :]

	# sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively

	left_most = left_most[np.argsort(left_most[:,1]),:]
	top_left, bottom_left = left_most

	D = dist.cdist(top_left[np.newaxis], right_most, "euclidean")[0]
	(bottom_right, top_right) = right_most[np.argsort(D)[::-1], :]

	return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

image = cv2.imread('./test_images/contoured_image_samples.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#find contours in edge map (not necessary?)
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the bounding box
# point colors
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

# loop over the contours individually
for (i, c) in enumerate(cnts):
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 1500:
		continue
	# compute the rotated bounding box of the contour, then
	# draw the contours
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	# show the original coordinates
	print("Object #{}:".format(i + 1))
	print(box)


# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	rect = order_points(box)
	# check to see if the new method should be used for
	# ordering the coordinates
	# rect = perspective.order_points(box)
	# show the re-ordered coordinates
	print(rect.astype("int"))
	print("")


	# loop over the original points and draw them
	for ((x, y), color) in zip(rect, colors):
		cv2.circle(image, (int(x), int(y)), 5, color, -1)
	# draw the object num at the top-left corner

	cv2.putText(image, "Object #{}".format(i + 1),
		(int(rect[0][0] - 15), int(rect[0][1] - 15)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
	# show the image
	# image.resize(100)
	# plt.figure()
	# plt.imshow(image)
	# plt.show()
	cv2.imshow("Image", image)
	cv2.waitKey(0)


