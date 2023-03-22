import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev

# Load the image
image = cv2.imread('./input_images/IMG_4141.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define red color range in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# Combine the masks
mask = mask1 + mask2

# Apply morphology operations to remove noise
kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Get the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour with a curve
epsilon = 0.005 * cv2.arcLength(largest_contour, True)
curve = cv2.approxPolyDP(largest_contour, epsilon, True)

# Calculate midpoints for each segment of the curve
midpoints = []
for i in range(len(curve) - 1):
    x1, y1 = curve[i][0]
    x2, y2 = curve[i+1][0]
    mid_x = int((x1 + x2) / 2)
    mid_y = int((y1 + y2) / 2)
    midpoints.append((mid_x, mid_y))

# print(np.shape(midpoints))

# Plot the curve and midpoints
plt.figure()
# for point in curve:
#     x, y = point.ravel()
#     plt.scatter(x, image.shape[0]-y, color='blue')
for point in midpoints:
    x, y = point
    print(point)
    plt.scatter(x, image.shape[0]-y, color='red')
plt.show()

# Save the curve points and midpoints to a CSV file
with open('test_data/output/curve_points_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y'])
    for point in curve:
        x, y = point.ravel()
        writer.writerow([x, y])










