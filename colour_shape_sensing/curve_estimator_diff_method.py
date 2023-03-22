import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('./input_images/IMG_4141.jpg')

# Convert to HSV and threshold to isolate red object
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# Find contours of red object
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Fit a curve to the contour
curve_fit = cv2.approxPolyDP(contours[0], 0.005 * cv2.arcLength(contours[0], True), True)

# Create a function for the curve
if curve_fit.shape[0] < 2:
    curve_func = None
else:
    curve_func = interp1d(curve_fit[:,0,0], curve_fit[:,0,1])


# Find 10 (x,y) coordinates along the curve
midpoint = (0, 0)  # Replace with actual midpoint coordinates
start_point = (curve_fit[0,0,0], curve_fit[0,0,1])  # Use actual starting point
curve_points = []
if curve_func is not None:
    curve_x = np.linspace(start_point[0], midpoint[0], num=10)
    curve_x = np.clip(curve_x, curve_fit[:, 0, 0].min(), curve_fit[:, 0, 0].max())
    curve_y = curve_func(curve_x)
    curve_points = list(zip(curve_x, curve_y))
print(curve_points)

# Shift coordinates to make midpoint (0,0)
midpoint_shift = np.array(midpoint)
curve_points_shifted = [tuple(np.array(p) - midpoint_shift) for p in curve_points]
# print(curve_points_shifted)

# plt.figure()
# for point in curve_points:
#     plt.plot(point)
# plt.show()