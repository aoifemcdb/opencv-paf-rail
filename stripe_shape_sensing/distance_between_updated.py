# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from Archive.thresholding import threshold_red
import matplotlib.pyplot as plt
from json import JSONEncoder
import pandas as pd

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

ref_obj = None
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

CALIBRATION_MATRIX = 0.23


def midpoint(pt_a, pt_b):
	return ((pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5)

def load_image_threshold(filename):
	thresholded_image = threshold_red(filename)
	# cv2.imshow('thresholded image',thresholded_image)
	# returns thresholded image (objects are green)
	image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)
	# returns full grayscale image (objects are white)
	# cv2.imshow('grayscaled image',image)
	# cv2.waitKey(0)
	return image

#find contours in edge map
def sort_contours(image):
	cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	#sort contours from left to right
	(cnts, _) = contours.sort_contours(cnts)
	return cnts

def get_bounding_box(cnts, ref_obj):
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

		ref_coords = np.vstack([ref_obj[0], ref_obj[1]])
		obj_coords = np.vstack([box, (c_x, c_y)])
		objects_coords.append(obj_coords)
	return ref_coords, objects_coords



def calibrate(coords, CALIBRATION_MATRIX): ## CAN BE USED FOR BOTH REF OBJECT COORDS AND OBJECT COORDS
	if type(coords) == list:
		coords_array = np.array(coords)
		coords = CALIBRATION_MATRIX * coords_array
	else:
		coords = CALIBRATION_MATRIX * coords
	return coords



def normalize_coords(coords, ref_coords):
	for i in range(0,1):
		coords[:,:,i] = coords[:,:,i] - ref_coords[4,i]
	return coords

def set_ref_zero(coords, ref_coords):
	for i in range(0,1):
		coords[:,i] = coords[:,i] - ref_coords[4,i]
	return coords

def get_positive_values(coords):
	coords = np.absolute(coords)
	return coords

def get_centres(ref_coords, object_coords):
	ref_centre = ref_coords[4,:]
	obj_centre = object_coords[:,4,:]
	return ref_centre, obj_centre

def get_line(ref_centre, obj_centre):
	ref_len = len(ref_centre) - 1
	obj_len = len(obj_centre)
	array_len = ref_len + obj_len

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

def get_trajectory(image, CALIBRATION_MATRIX):
	#needed to invert image
	image = cv2.flip(image,0)
	cnts = sort_contours(image)
	ref_coords, object_coords = get_bounding_box(cnts, None)
	ref_coords_calibrated = calibrate(ref_coords, CALIBRATION_MATRIX)
	ref_coords_calibrated = np.absolute(ref_coords_calibrated)
	objects_coords_calibrated = calibrate(object_coords, CALIBRATION_MATRIX)
	objects_coords_calibrated = np.absolute(objects_coords_calibrated)

	objects_coords_calibrated = normalize_coords(objects_coords_calibrated, ref_coords_calibrated)
	ref_coords_calibrated= set_ref_zero(ref_coords_calibrated, ref_coords_calibrated)
	ref_centre, obj_centre = get_centres(ref_coords_calibrated, objects_coords_calibrated)
	x,y  = get_line(ref_centre, obj_centre)
	return x,y,ref_coords_calibrated, objects_coords_calibrated

image = load_image_threshold('../colour_shape_sensing/input_images/print_samples_rotate.jpg')
x,y, ref_coords_calibrated, objects_coords_calibrated = get_trajectory(image, CALIBRATION_MATRIX)

def save_trajectory(x,y, filename: str):
	xy = np.array([x,y])
	xy = xy.T
	df = pd.DataFrame(xy, columns=['x','y'])
	df.to_csv(filename, index=False)
	print("Trajectory Saved")
	return

# save_trajectory(x,y, './output_trajectories/trajectory_2.csv')


# def decode_json(data):
# 	print("Decode JSON serialized NumPy array")
# 	decoded_array = json.loads(data.read())
# 	final_array = np.asarray(decoded_array["array"])
# 	print("NumPy Array")
# 	print(final_array)
# 	return
#
# decode_json(data)

#plot
plt.figure(figsize=(12,4))
plt.scatter(ref_coords_calibrated[:,0], ref_coords_calibrated[:,1])
plt.plot(x,y, '--')
plt.scatter(objects_coords_calibrated[:,:,0], objects_coords_calibrated[:,:,1], marker='v')

for x,y in zip(x,y):
	label = f"({x:.1f},{y:.1f})"
	plt.annotate(label, (x,y), textcoords = "offset points", xytext = (0,10), ha = 'center')

# plt.xlim([-300, 1000])
plt.xlabel('mm')
plt.ylabel('mm')
ax=plt.gca()
# ax.yaxis.set_tick_params(labelleft=False)
# ax.set_yticks([])
plt.show()