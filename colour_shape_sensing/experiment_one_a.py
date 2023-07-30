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
import matplotlib.pyplot as plt
import numpy as np



def main():
    calibration_images_path = './experiment_images_260623/checkerboard_calibration/*.jpg'
    # patch_file = './experiment_images_210623/patch/patch_1.jpg'
    folder_path = './experiment_images_260623/kidney_phantom/colorbands/'
    output_file = './experiment_data_260623/kidney_phantom/colorbands_test.csv'

    calibration = Calibration()

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
                # patch_image = calibration.read_image(patch_file)
                # patch_undistort = cv2.undistort(patch_image, camera_matrix, dist_coeffs)
                # pixels_per_mm_x, pixels_per_mm_y = calibration.get_calibration_matrix(patch_undistort,
                #                                                                       real_width, real_length, color,
                #                                                                       plot_images=False,
                #                                                                       print_values=False)

                image = cv2.imread(file_path)
                color_boundaries = ColorBoundaries()
                lower_color, upper_color = color_boundaries.get_color_boundaries(color)

                undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
                # patch_calibrated = calibration.calibrate_image(undistorted_image, pixels_per_mm_x, pixels_per_mm_y)
                result = shape_sensing.process_image(undistorted_image, lower_color, upper_color, plot_images=True)

                curve_downsampled = shape_sensing.downsample_spline_curve(result[2], result[3], 100, plot_curve=True)

                curvature, _ = CurvatureCalculator.circle_fitting_method(curve_downsampled)
                radius = 1.0 / curvature
                radius = radius * 0.1

                # writer.writerow([filename, radius])
                return undistorted_image, curve_downsampled


def test_shape_sensing():
    folder_path_rail = './experiment_images_260623/kidney_phantom/rails/'

    folder_path_colorbands = './experiment_images_260623/kidney_phantom/colorbands/'

    output_file = './experiment_data_260623/kidney_phantom/rails_v2.csv'

    undistorted_image_colorband, curve_colorband = main(folder_path_colorbands, output_file)
    undistorted_image_rail, curve_rail = main(folder_path_rail, output_file)

    # Flip the image along the x-axis
    flipped_image_rail = np.flipud(undistorted_image_rail)
    flipped_image_colorband = np.flipud(undistorted_image_colorband)

    # Get the aspect ratio of the flipped image
    aspect_ratio = flipped_image_rail.shape[1] / flipped_image_rail.shape[0]

    # Set the desired width of the plot (in inches)
    plot_width = 8.0


    # Calculate the height of the plot based on the aspect ratio
    plot_height = plot_width / aspect_ratio
    #
    # # Set the plot dimensions
    # plt.figure(figsize=(plot_width, plot_height))
    #
    # # Set the x and y limits for the plot
    # x_min, x_max = 0, undistorted_image_rail.shape[1]
    # y_min, y_max = 1000, 0
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    #
    # plt.plot(curve_rail[:, 0], curve_rail[:, 1], label='Rail', color='#C1292E')
    # plt.plot(curve_colorband[:, 0], curve_colorband[:, 1], label='Colorband', color='#235789')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # # plt.ylim(0, 200)
    # plt.title('Spline Curve Shape Approximation Rail vs Colorband on Phantom')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(undistorted_image_rail)
    # plt.show()

    # Set the plot dimensions and create a subplot
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # Set the x and y limits for the plot
    x_min, x_max = 0, flipped_image_rail.shape[1]
    y_min, y_max = 0, flipped_image_rail.shape[0]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Superimpose the line plot on the flipped image
    ax.imshow(flipped_image_colorband)
    # ax.plot(curve_rail[:, 0], flipped_image_rail.shape[0] - curve_rail[:, 1], label='Rail', color='#C1292E')
    ax.plot(curve_colorband[:, 0], flipped_image_rail.shape[0] - curve_colorband[:, 1], label='Colorband',
            color='#F0D000')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Spline Curve Shape Approximation Rail vs Colorband on Phantom')
    ax.legend()

    plt.show()
    # dpi = 300
    # plt.savefig('./experiment_data_260623/kidney_phantom/phantom_rail_colorband_approximation.png', dpi=dpi)
    # print('Image Saved')

    return


# test_shape_sensing()
if __name__ == '__main__':
    main()