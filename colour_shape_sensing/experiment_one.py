""" 1. load experiment image
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

def main():
    calibration_images_path = './experiment_images_260723/calibration/*.jpg'
    # patch_file = './experiment_images_210623/patch/patch_1.jpg'
    folder_path = './experiment_images_260723/90mm/train'
    output_file = './experiment_data_260723/90mm_angle_0.csv'

    calibration = Calibration()
    real_width = 24.5  # mm
    real_length = 24.5  # mm
    color = 'green'
    checkerboard_size = (5, 7)
    camera_matrix, dist_coeffs = calibration.calibrate_camera(calibration_images_path, checkerboard_size)
    shape_sensing = ShapeSensing()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Radius'])

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                file_path = os.path.join(folder_path, filename)
                image = cv2.imread(file_path)
                color_boundaries = ColorBoundaries()
                lower_color, upper_color = color_boundaries.get_color_boundaries(color)
                undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
                # warped_image = shape_sensing.apply_perspective_transform(undistorted_image, lower_color, upper_color, plot_images=True)
                # patch_calibrated = calibration.calibrate_image(undistorted_image, pixels_per_mm_x, pixels_per_mm_y)
                result = shape_sensing.process_image(undistorted_image, lower_color, upper_color, plot_images=True)
                curve_downsampled = shape_sensing.downsample_spline_curve(result[2], result[3], 100, plot_curve=True)

                curvature, _ = CurvatureCalculator.circle_fitting_method(curve_downsampled)
                radius = 1.0 / curvature
                radius = radius * 0.1

                writer.writerow([filename, radius])
    return curve_downsampled


if __name__ == '__main__':
    main()

