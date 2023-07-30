import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Load the image in BGR format
img = cv2.imread('./test_images/CAD_models/red_resized/red_110mm_resized.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of red color in HSV color space
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# Combine the two masks
mask = cv2.bitwise_or(mask1, mask2)

# Apply a median filter to remove noise
mask = cv2.medianBlur(mask, 5)
# plt.figure()
# plt.imshow(mask)
# plt.show()

# Find the contours in the binary image
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Find the contour with the largest area
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Extract the x and y coordinates of the bottom contour
x_vals = np.unique(max_contour[:,:,0])
y_values = np.zeros_like(x_vals)
for i, x_val in enumerate(x_vals):
    y_values[i] = np.max(max_contour[max_contour[:,:,0]==x_val][:,1])
x_new = np.linspace(x_vals.min(), x_vals.max(), num=1000)
y_new = np.interp(x_new, x_vals, y_values)

# Fit a new spline curve to the new data
smoothing_factor = 0.1
spline_new = UnivariateSpline(x_new, y_new, k=3, s=smoothing_factor)

# Evaluate the new spline curve at a range of x-values
y_new_smooth = spline_new(x_new)

# Check for discontinuous curves and apply fix
if np.any(np.diff(y_new_smooth) < -500):
    print('Discontinuous curve detected, applying fix...')
    y_new_smooth_fixed = []
    for i in range(len(x_new)):
        if i == 0:
            y_new_smooth_fixed.append(y_new_smooth[i])
        elif y_new_smooth[i] - y_new_smooth[i-1] < -500:
            y_new_smooth_fixed.append(y_new_smooth_fixed[i-1])
        else:
            y_new_smooth_fixed.append(y_new_smooth[i])
    y_new_smooth = np.array(y_new_smooth_fixed)

# Flip the y-coordinates of the spline curve back to the original orientation
y_new_smooth = np.max(y_values) - y_new_smooth

# Create a figure with two subfigures
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
#
# # Plot the image in the first subfigure
# ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')
# # Plot the spline curve in the second subfigure
# ax2.plot(x_new, y_new_smooth)
# ax2.tick_params(axis='y', which='both', length=0)

# Set the x and
# Fit a new spline curve to the new data
smoothing_factor = 0.1
spline_new = UnivariateSpline(x_new, y_new, k=3, s=smoothing_factor)

# Evaluate the new spline curve at a range of x-values
x_range = np.arange(x_new[0], x_new[-1], 1)
y_new_smooth = spline_new(x_range)

# Check for discontinuous curves and apply fix
if np.any(np.diff(y_new_smooth) < -500):
    print('Discontinuous curve detected, applying fix...')
    y_new_smooth_fixed = []
    for i in range(len(x_range)):
        if i == 0:
            y_new_smooth_fixed.append(y_new_smooth[i])
        elif y_new_smooth[i] - y_new_smooth[i-1] < -500:
            y_new_smooth_fixed.append(y_new_smooth_fixed[i-1])
        else:
            y_new_smooth_fixed.append(y_new_smooth[i])
    y_new_smooth = np.array(y_new_smooth_fixed)

# Flip the y-coordinates of the spline curve back to the original orientation
# y_new_smooth = np.max(y_new) - y_new_smooth

# Create a figure with two subfigures
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Plot the image in the first subfigure
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')
# Plot the spline curve in the second subfigure
ax2.plot(x_range, y_new_smooth)
ax2.tick_params(axis='y', which='both', length=0)

# Set the x and y limits of both subfigures to be the same
ax1.set_xlim([0, img.shape[1]])
ax2.set_xlim([0, img.shape[1]])
ax1.set_ylim([img.shape[0], 0])
ax2.set_ylim([img.shape[0], 0])

# Add titles and axis labels
ax1.set_title('Original Image')
ax2.set_title('Spline Curve')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# Show the figure
plt.show()



