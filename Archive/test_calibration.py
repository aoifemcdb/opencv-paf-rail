from Archive.approximate_spline import *
import cv2
import re

# calibration_image_path = './experiment_images_010523/yellow/blue_yellow_0/calibration_yellow_iter1.jpg'
# test_image_path = './experiment_images_010523/yellow/blue_yellow_0/calibration_yellow_iter1.jpg'
# color_name = 'yellow'
# real_length = 7 #mm    #actually width of the rail
# real_width = 70 #mm    #actually height of the rail


def calculate_average_pixels_per_mm(filepath, real_length, real_width):
    # Extract color from filepath
    color_match = re.search(r'experiment_images_010523/(\w+)/', filepath)
    if color_match:
        color = color_match.group(1)
    else:
        color = 'yellow'

    pixels_per_mm_x_list = []
    pixels_per_mm_y_list = []

    for i in range(1, 6):  # loop through 1-5
        image_path = filepath.format(i)
        image = cv2.imread(image_path)
        img, pixels_per_mm_x, pixels_per_mm_y = get_calibration_matrix(image, real_width, real_length, color)
        pixels_per_mm_x_list.append(pixels_per_mm_x)
        pixels_per_mm_y_list.append(pixels_per_mm_y)

    average_pixels_per_mm_x = round(sum(pixels_per_mm_x_list) / len(pixels_per_mm_x_list), 2)
    average_pixels_per_mm_y = round(sum(pixels_per_mm_y_list) / len(pixels_per_mm_y_list), 2)
    height_mm = int(image.shape[0] / pixels_per_mm_y)
    width_mm = int(image.shape[0] / pixels_per_mm_x)

    resized_image = cv2.resize(image, (width_mm, height_mm))


    return resized_image, average_pixels_per_mm_x, average_pixels_per_mm_y

# pixels_per_mm_x, pixels_per_mm_y = calculate_average_pixels_per_mm(calibration_image_path, real_length, real_width)
# print(pixels_per_mm_x, pixels_per_mm_y)
#
# lower_color, upper_color = get_color_boundaries(color_name)
# img, mask, x_new, y_new = process_image(test_image_path, lower_color, upper_color)
# x, y = calibrate_image(x_new, y_new, pixels_per_mm_x, pixels_per_mm_y)
# spline_curve_downsampled = get_spline_curve(x,y,50)
# plot_calibrated_spline(spline_curve_downsampled[:,0],spline_curve_downsampled[:,1])