"""
 This script is to test the shape sensing in stereo vision
 1. load experiment image
    2. load calibration image
    3. calibrate image (get camera matrix and distortion coeffs and undistort image)
    4. specify the colour boundaries that you want to detect
    5. process the image (perform shape detection)
    6. downsample the spline approximation for analysis and visualisation purposes
    7. (optional) plot image and spline curve
    8. calculate curvature
    9. calculate radius
    10. print values """

import cv2
import os
import csv
from color_boundaries import ColorBoundaries
from curvature_methods import CurvatureCalculator
from colour_shape_sensing.calibration import Calibration
from colour_shape_sensing.shape_sensing import ShapeSensing
import matplotlib.pyplot as plt
import numpy as np

def stereo_calibration(left_filepath, right_filepath):
    calibration = Calibration()
    checkerboard_size = (5,7)
    camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, \
    T = calibration.calibrate_stereo_camera(left_filepath, right_filepath, checkerboard_size)

    point_left = []   # this is the point in one image that you want to transform to the other image

    point_transformed = calibration.stereo_transform_point(point_left, camera_matrix_left, dist_coeffs_left,
                                                           amera_matrix_right, dist_coeffs_right, R, T)
    return camera_matrix_left, camera_matrix_right, dist_coeffs_left, dist_coeffs_right, point_transformed


def main(folder_path_left, folder_path_right, output_file):
    color = 'green'
    color_boundaries = ColorBoundaries()
    lower_color, upper_color = color_boundaries.get_color_boundaries(color)


    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Radius'])

        for filename_left, filename_right in zip(os.listdir(folder_path_left), os.listdir(folder_path_right)):
            if filename_left.endswith('.jpg') and filename_right.endswith('.jpg'):
                file_path_left = os.path.join(folder_path_left, filename_left)
                file_path_right = os.path.join(folder_path_right, filename_right)

                image_left = cv2.imread(file_path_left)
                image_right = cv2.imread(file_path_right)

                undistorted_image_left = cv2.undistort(image_left, camera_matrix_left, dist_coeffs_left)
                undistorted_image_right = cv2.undistort(image_right, camera_matrix_right, dist_coeffs_right)

                curve_downsampled = shape_sensing.process_image_stereo(undistorted_image_left, undistorted_image_right,
                                                                       lower_color, upper_color, plot_images=False)

                curvature, _ = CurvatureCalculator.circle_fitting_method(curve_downsampled)
                radius = 1.0 / curvature
                radius = radius * 0.1

                # writer.writerow([filename, radius])
                return radius, undistorted_image, curve_downsample