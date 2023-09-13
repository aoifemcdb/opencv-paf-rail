# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from Archive.thresholding import threshold_red
import matplotlib.pyplot as plt

ref_obj = None
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
CALIBRATION_MATRIX = 1
image = './input_images/print_samples_long.jpg'

def midpoint(pt_a, pt_b):
	return ((pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5)

def get_contours(image, CALIBRATION_MATRIX, ref_obj): #input: image, ref obj
	thresholded_image = threshold_red(image)
	image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)

	# find contours in edge map
	cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	#sort contours from left to right
	(cnts, _) = contours.sort_contours(cnts)
	#list object_coords stores co-ordinates of each of the detected coloured objects (squares)
	objects_coords = []
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
	# 	orig = image.copy()
	# 	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# 	cv2.drawContours(orig, [ref_obj[0].astype("int")], -1, (0, 255, 0), 2)
		# stack the reference coordinates and the object coordinates
		# to include the object center
		#(5 x 2) matrix of corners of bounding box + centre co ordinate
		ref_coords = np.vstack([ref_obj[0], ref_obj[1]])
		ref_coords = CALIBRATION_MATRIX*ref_coords  ##common to both ref coords and object coords so
		# take out and move to different function
		obj_coords = np.vstack([box, (c_x, c_y)])
		objects_coords.append(obj_coords)
		return ref_coords, objects_coords



# loop over the original points
# 	for ((x_a, y_a), (x_b, y_b), color) in zip(ref_coords, obj_coords, colors):
# 		cv2.circle(orig, (int(x_a), int(y_a)), 5, color, -1)
# 		cv2.circle(orig, (int(x_b), int(y_b)), 5, color, -1)
# 		cv2.line(orig, (int(x_a), int(y_a)), (int(x_b), int(y_b)),
# 			color, 2)
# 		# compute the Euclidean distance between the coordinates,
# 		# and then convert the distance in pixels to distance in
# 		# units
# 		D = dist.euclidean((x_a, y_a), (x_b, y_b)) / ref_obj[2]
# 		(m_x, m_y) = midpoint((x_a, y_a), (x_b, y_b))
# 		cv2.putText(orig, "{:.1f}mm".format(D), (int(m_x), int(m_y - 10)),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,5)
#
# 		# show the output image
# 		# cv2.imwrite('./output_images/distance_between_test.jpg', orig)
# 		# cv2.waitKey(0)
# 		# plt.figure()
# 		# plt.imshow(orig)
# 		# plt.show()
def calibrate(coords): ## CAN BE USED FOR BOTH REF OBJECT COORDS AND OBJECT COORDS
	if type(coords) == list:
		coords_array = np.array(coords)
		coords = CALIBRATION_MATRIX * coords_array
	else:
		coords = CALIBRATION_MATRIX * coords
	return coords

def make_traj_positive(objects_coords_arr):
	#use if need to make co-ordinates +ve
	objects_coords_arr = np.absolute(objects_coords_arr)
	return objects_coords_arr

def normalize_objects(ref_coords, objects):
	##### NEEDS FIXING #####
	### NORMALIZE TO MAKE REF OBJECT THE STARTING POINT ###
	objects[:, :, 0] = objects[:, :, 0] - ref_coords[4, 0]
	objects[:, :, 1] = objects[:, :, 1] - ref_coords[4, 1]
	return objects

def normalize_ref(ref_coords):
	### NORMALIZE TO MAKE REF OBJECT THE STARTING POINT ###
	ref_coords[:,0] = ref_coords[:,0] - ref_coords[4,0]
	ref_coords[:,1] = ref_coords[:,1] - ref_coords[4,1]

def make_ref_positive(ref_coords):
	#make sure all co-ords are positive
	ref_coords = np.absolute(ref_coords)
	return ref_coords

def get_centres(ref_coords, objects_coords_arr):
	ref_centre = ref_coords[4,:]
	obj_centre = objects_coords_arr[:,4,:]
	return ref_centre, obj_centre

def get_trajectory(ref_centre,obj_centre):
	ref_len = len(ref_centre) - 1
	obj_len = len(obj_centre)
	# array_len = ref_len + obj_len

	line_list = []
	for i in range(0,obj_len):
		line_element = obj_centre[i]
		line_list.append(line_element)

	line_array = np.array(line_list)
	line_array = np.vstack((ref_centre, line_array))


	x = []
	y = []
	for i in range(0, len(line_array)):
		element = line_array[i]
		element_x = element[0]
		element_y = element[1]
		x.append(element_x)
		y.append(element_y)
	return x,y


def plot_objects_and_trajectory(ref_coords, objects_coords_arr, x,y):
	#plot
	plt.figure(figsize=(12,4))
	plt.scatter(ref_coords[:,0], ref_coords[:,1])
	plt.plot(x,y, '--')
	plt.scatter(objects_coords_arr[:,:,0], objects_coords_arr[:,:,1], marker='v')

	for x,y in zip(x,y):
		label = f"({x:.1f},{y:.1f})"
		plt.annotate(label, (x,y), textcoords = "offset points", xytext = (0,10), ha = 'center')

	plt.xlim([-300, 1000])
	plt.xlabel('mm')
	plt.ylabel('mm')
	ax=plt.gca()
	# ax.yaxis.set_tick_params(labelleft=False)
	# ax.set_yticks([])
	plt.show()

#get co-ordinates of reference object and detected objects
ref_coords, object_coords = get_contours(image, CALIBRATION_MATRIX, None)
#perform camera calibration on both
ref_coords_calibrated = calibrate(ref_coords)
object_coords_calibrated = calibrate(object_coords)
#make positive
ref_coords_calibrated = make_ref_positive(ref_coords_calibrated)
#normalize (set ref object to be 0)
normalized_object = normalize_objects(ref_coords_calibrated,object_coords_calibrated)
normalized_ref = normalize_ref(ref_coords_calibrated)
#get co-ordinates of centre of objects
ref_centre, obj_centre = get_centres(normalized_ref, normalized_object)
#make line using x,y co-ordinates
x,y = get_trajectory(ref_centre, obj_centre)
#plot
plot_objects_and_trajectory(ref_centre, obj_centre,x,y)





#TO DO: set to run on multiple images - using GUI?