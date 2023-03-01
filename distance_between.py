# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

def midpoint(pt_a, pt_b):
	return ((pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5)


image = cv2.imread('./test_images/contoured_image_samples.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#find contours in edge map (not necessary?)
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the bounding box
# point colors
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

ref_obj = None

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 1500:
		continue
	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	# compute the center of the bounding box
	c_x = np.average(box[:, 0])
	c_y = np.average(box[:, 1])

	# if this is the first contour we are examining (i.e.,
	# the left-most contour), we presume this is the
	# reference object


	#TO DO: ADD ref object that's **not** one of the bounding boxes on rail
	if ref_obj is None:
		# unpack the ordered bounding box, then compute the
		# midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-right and
		# bottom-right
		(top_left, top_right, bottom_right, bottom_left) = box
		(tlbl_x, tlbl_y) = midpoint(top_left, bottom_left)
		(trbr_x, trbr_y) = midpoint(top_right, bottom_right)
		# compute the Euclidean distance between the midpoints,
		# then construct the reference object
		D = dist.euclidean((tlbl_x, tlbl_y), (trbr_x, trbr_y))
		ref_obj = (box, (c_x, c_y), D / 100)
		continue

# draw the contours on the image
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [ref_obj[0].astype("int")], -1, (0, 255, 0), 2)
	# stack the reference coordinates and the object coordinates
	# to include the object center
	ref_coords = np.vstack([ref_obj[0], ref_obj[1]])
	obj_coords = np.vstack([box, (c_x, c_y)])

# loop over the original points
	for ((x_a, y_a), (x_b, y_b), color) in zip(ref_coords, obj_coords, colors):
		cv2.circle(orig, (int(x_a), int(y_a)), 5, color, -1)
		cv2.circle(orig, (int(x_b), int(y_b)), 5, color, -1)
		cv2.line(orig, (int(x_a), int(y_a)), (int(x_b), int(y_b)),
			color, 2)
		# compute the Euclidean distance between the coordinates,
		# and then convert the distance in pixels to distance in
		# units
		D = dist.euclidean((x_a, y_a), (x_b, y_b)) / ref_obj[2]
		(m_x, m_y) = midpoint((x_a, y_a), (x_b, y_b))
		cv2.putText(orig, "{:.1f}mm".format(D), (int(m_x), int(m_y - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,5)
		# show the output image
		# cv2.imshow("Image", orig)
		cv2.imwrite('./output_images/distance_between.jpg', orig)

		# cv2.waitKey(0)
		# plt.figure()
		# plt.imshow(orig)
		# plt.show()