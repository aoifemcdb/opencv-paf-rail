import cv2
import os
import csv
from color_boundaries import ColorBoundaries
from curvature_methods import CurvatureCalculator
from colour_shape_sensing.calibration import Calibration
from colour_shape_sensing.shape_sensing import ShapeSensing
import matplotlib.pyplot as plt
import numpy as np

# Function to flip and normalize the curve
def flip_and_normalize_curve(curve_downsampled):
    x_values = [point[0] for point in curve_downsampled]
    y_values = [point[1] for point in curve_downsampled]

    x_start = x_values[0]
    y_start = y_values[0]

    x_values_flipped = [x - x_start for x in x_values]
    y_values_flipped = [y - y_start for y in y_values]

    return x_values_flipped, y_values_flipped

# Load calibration images
calibration_images_path = './experiment_images_260623/checkerboard_calibration/*.jpg'
checkerboard_size = (5, 7)

calibration = Calibration()
camera_matrix, dist_coeffs = calibration.calibrate_camera(calibration_images_path, checkerboard_size)

# Load the test image
image_path = './experiment_images_260623/kidney_phantom/colorbands/WIN_20230626_14_15_34_Pro.jpg'
image = cv2.imread(image_path)

# Define color range for shape sensing
color = 'green'
color_boundaries = ColorBoundaries()
lower_color, upper_color = color_boundaries.get_color_boundaries(color)

# Undistort the image
undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

# Perform shape sensing
shape_sensing = ShapeSensing()
result = shape_sensing.process_image(undistorted_image, lower_color, upper_color, plot_images=True)

# Downsample the spline curve
curve_downsampled = shape_sensing.downsample_spline_curve(result[2], result[3], 100, plot_curve=True)

# Flip and normalize the curve
x_values_flipped, y_values_flipped = flip_and_normalize_curve(curve_downsampled)

# Write the flipped and normalized curve to the CSV file
output_file = './experiment_data_260623/kidney_phantom/single_rail_test.csv'

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Flipped_X', 'Flipped_Y'])
    for x, y in zip(x_values_flipped, y_values_flipped):
        writer.writerow([x, y])
