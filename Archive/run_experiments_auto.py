import csv
import os
import re
import cv2
import numpy as np
from approximate_spline import *
from get_curvature import *

def load_data(colour, image_path, real_width, real_length):
    # real_width = 100  # mm
    # real_length = 20  # mm
    lower_color, upper_color = get_color_boundaries(colour)
    img, mask, x_new, y_new_smooth = process_image(image_path, lower_color, upper_color)
    image, pixels_per_mm_x, pixels_per_mm_y = get_calibration_matrix(img, real_width, real_length, colour)
    x, y = calibrate_image(x_new, y_new_smooth, pixels_per_mm_x, pixels_per_mm_y)
    spline_curve_downsampled = get_spline_curve(x, y, 20)

    return spline_curve_downsampled

def get_radius_error(data, radius):
    data_curvature = calculate_curvature(data)
    data_radius = np.reciprocal(data_curvature)
    data_radius = remove_outliers_iqr(data_radius)
    data_radius = data_radius[1:-1]
    data_radius_mean, data_radius_stddev = get_radius_mean_stddev(data_radius)
    data_radius_error = data_radius_mean - radius
    return data_radius_mean, data_radius_stddev, data_radius_error

def get_radius_mean_stddev(radius):
    radius_mean = np.mean(radius)
    radius_stddev = np.std(radius)
    return radius_mean, radius_stddev

def save_to_csv(filename, column_names, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the column names if the file is empty
        if file.tell() == 0:
            writer.writerow(column_names)
        # Write the data
        writer.writerow(data)

def process_images(directory_path, csv_filename):
    column_names = ['Colour', 'Radius', 'Iteration', 'Data Radius Mean', 'Data Radius Error', 'Data Radius Std Dev']

    # Create or open the CSV file
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the column names if the file is empty
        if file.tell() == 0:
            writer.writerow(column_names)

        for filename in os.listdir(directory_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Extract radius and colour from image path
                match = re.search(r'\d+', filename)
                if match:
                    radius = float(match.group(0))
                else:
                    radius = 'N/A'
                match = re.search(r'green|yellow|orange', filename)
                if match:
                    colour = match.group(0)
                else:
                    colour = 'N/A'
                # Extract iteration number from filename
                match = re.search(r'iter(\d+)', filename)
                if match:
                    iteration = int(match.group(1))
                else:
                    iteration = 'N/A'

                # Load and process the image
                image_path = os.path.join(directory_path, filename)
                """Update with real measured values for each rail"""
                real_width = 10  # mm
                real_length = 80  # mm
                """"""""""""""""""""""""""""""""""""""""""""""""""""""
                data = load_data(colour, image_path, real_width, real_length)
                data_radius_mean, data_radius_stddev, data_radius_error = get_radius_error(data, radius)

                # Save the data to the CSV file
                data_to_save = [colour, radius, iteration, data_radius_mean, data_radius_error, data_radius_stddev]
                save_to_csv(csv_filename, column_names, data_to_save)


directory_path = '../colour_shape_sensing/experiment_images_220423/green'
csv_filename = os.path.join(directory_path, 'results_green.csv')
process_images(directory_path, csv_filename)

