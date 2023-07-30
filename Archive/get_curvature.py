from Archive.approximate_spline import *
from curvature_methods import *

def calculate_curvature(points):
    # Calculate the first and second derivatives of the x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Calculate the curvature using the formula:
    # curvature = |dx*ddy - ddx*dy| / (dx^2 + dy^2)^(3/2)
    numerator = np.abs(dx * ddy - ddx * dy)
    denominator = (dx ** 2 + dy ** 2) ** (3/2)
    curvature = numerator / denominator

    # Return the curvature as a NumPy array
    print(curvature)
    print(len(curvature))
    return curvature

# def test(file_path, color_name):
#     calibration_file_path = './test_images/CAD_models/yellow/yellow_50mm.jpg'
#     image = cv2.imread(calibration_file_path)
#     lower_color, upper_color = get_color_boundaries(color_name)
#     img, mask, x_new, y_new_smooth = process_image(image, lower_color, upper_color)
#     real_length = 30  # mm    #length is y direction
#     real_width = 50  # mm #width is x direction
#     # color_name = 'yellow'
#     # image, pixels_per_mm_x, pixels_per_mm_y = get_calibration_matrix(img, real_width, real_length, color_name)
#
#     resized_image, pixels_per_mm_x, pixels_per_mm_y = calculate_average_pixels_per_mm(calibration_file_path, real_length, real_width)
#     print(pixels_per_mm_x, pixels_per_mm_y) #check
#     x_new, y_new_smooth = calibrate_image(x_new, y_new_smooth, pixels_per_mm_x, pixels_per_mm_y)
#     return x_new, y_new_smooth

def remove_outliers_std_dev(data):
    # Calculate the mean and standard deviation of the data
    mean = np.mean(data)
    std = np.std(data)

    # Create a boolean array indicating which data points are within 1 std of the mean
    mask = np.abs(data - mean) <= 1.5 * std
    # Return the data with outliers removed
    # print(mean, std)
    return data[mask]

def remove_outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    q2 = np.mean(data)
    iqr = q3 - q1
    print('IQR:')
    print(q1, q3)
    lower_bound = q2 - .5 * iqr
    upper_bound = q2 + .5 * iqr
    print('Range:')
    print(lower_bound, upper_bound)
    data_filtered = []
    for element in data:
        if lower_bound <= element <= upper_bound:
            data_filtered.append(element)
    a = np.array(data_filtered)
    # print(len(a))
    return np.array(data_filtered)

def remove_systematic_error(radius, geometric_distance):
    "To account for the error between the suction cup and the rail. Measure geometric distance before. "
    # might actually involve some trigonometry to be correct - check
    radius_fixed = radius - geometric_distance
    return radius_fixed

def get_radius():
    filepath = '../colour_shape_sensing/experiment_images_110523/blue/110mm_blue_iter1.jpg'
    color_name = 'blue'
    real_width = 80 #mm
    real_length = 8 #mm
    calibration_filepath = '../colour_shape_sensing/experiment_images_110523/blue/calibration_blue_iter1.jpg'
    curve = approximate_spline(filepath, color_name, real_width, real_length, calibration_filepath)
    # column_a = curve[15:-20, 0]
    # column_b = curve[15:-20, 1]
    # curve = np.column_stack((column_a, column_b))
    plot_calibrated_spline(curve[:,0], curve[:,1])


    # print(curve)
    # change curvature method here
    curvature = circle_fitting_method(curve)
    # print(curvature)
    radius = np.reciprocal(curvature)
    print('Radius:')
    print(radius)
    return radius

def perform_statistics(radius, geometric_distance=5):
    # select which method to remove outliers
    radius = remove_outliers_std_dev(radius)
    # print(radius)
    # radius = remove_outliers_std_dev(radius)
    # radius = radius[1:-3]
    # print(radius)
    # radius_fixed = remove_systematic_error(radius, geometric_distance)
    mean_radius = radius
    print(mean_radius)
    return mean_radius

def plot_radius():
    radius = get_radius()
    mean_radius = perform_statistics(radius, geometric_distance=5)
    # Plot the curvature
    # indices = np.linspace(0, len(radius) - 1, len(radius), dtype=int)
    fig, ax = plt.subplots()
    ax.scatter(radius,1)
    ax.set_xlabel('Position along curve')
    ax.set_ylabel('Radius')
    # text = 'Average sensed radius:' + "{:.0f}".format(float(str(mean_radius))) + 'mm \n ' \
    #                                                                              'Geometric radius: 50 mm'

    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props)

    plt.show()

def main():
    plot_radius()
    #
    # # x_new, y_new_smooth = test('./test_images/CAD_models/yellow/yellow_50mm.jpg', 'yellow')
    #
    # # curve = get_spline_curve(x_new, y_new_smooth, 30)
    # curve = test_curve('./experiment_images_010523/yellow/blue_yellow_20/50mm_yellow_iter1.jpg')
    # print(curve)
    #
    # # Calculate the curvature of the points
    #
    # # curve = test_curve()
    # curvature = calculate_curvature(curve)
    # radius = np.reciprocal(curvature)
    # # radius = radius/10 #scale (?)
    # radius = remove_outliers_iqr(radius)
    # # remove points at either end - erroneous
    # radius = radius[1:-1]
    # geometric_distance = 5 #mm
    # radius_fixed = remove_systematic_error(radius, geometric_distance)
    # mean_radius = np.average(radius_fixed)
    # # print(radius)
    # print(mean_radius)
    #
    # # Plot the curvature
    # indices = np.linspace(0, len(radius)-1, len(radius), dtype=int)
    # fig, ax = plt.subplots()
    # ax.scatter(indices, radius)
    # ax.set_xlabel('Position along curve')
    # ax.set_ylabel('Radius')
    # text = 'Average sensed radius:' + "{:.0f}".format(float(str(mean_radius))) + 'mm \n ' \
    #                                                                              'Geometric radius: 70 mm'
    #
    #
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props)
    #
    # plt.show()

if __name__ == '__main__':
    main()
