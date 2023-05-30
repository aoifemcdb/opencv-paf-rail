import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from color_boundaries import get_color_boundaries
from curvature_methods import circle_fitting_method
from checkerboard_calibration import calibrate_camera


### THIS SCRIPT NEEDS TIDYING! ###

def get_calibration_matrix(calibration_image, real_width, real_length, color):
    # Get a mask of the pixels in the image that match the specified color
    hsv = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = get_color_boundaries(color)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find the contours of the pixels in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the contour
    ax.plot(largest_contour[:, 0, 0], largest_contour[:, 0, 1], 'b-')

    # Plot the bounding rectangle
    rectangle = patches.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)

    # Set plot limits
    ax.set_xlim(0, calibration_image.shape[1])
    ax.set_ylim(calibration_image.shape[0], 0)

    # Show the plot
    plt.show()

    print(h)
    print(w)

    # Calculate the number of pixels per mm in the x and y directions
    pixels_per_mm_x = w / real_width
    pixels_per_mm_y = h / real_length

    return pixels_per_mm_x, pixels_per_mm_y

 ## need to incorporate this into the below


def calibrate_image(test_image, pixels_per_mm_x, pixels_per_mm_y):
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title('Image fed to calibration')
    # plt.show()
    height_mm = int(test_image.shape[0] / pixels_per_mm_y)
    width_mm = int(test_image.shape[0]/pixels_per_mm_x)
    resized_image = cv2.resize(test_image, (width_mm, height_mm))
    # y_values = np.max(y) - y
    return resized_image

def calibrate_image_2(test_image, pixels_per_mm_x, pixels_per_mm_y):
    plt.figure()
    plt.imshow(test_image)
    plt.title('Image fed to calibration')
    plt.show()
    hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:, 0] /= pixels_per_mm_x
    hsv_image[:,:, 1] /= pixels_per_mm_y
    # calibrated_bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return hsv_image

def process_image(image, lower_color, upper_color, pixels_per_mm_x, pixels_per_mm_y, kernel_size=5):
    image_calibrated = calibrate_image(image, pixels_per_mm_x, pixels_per_mm_y)
    print(pixels_per_mm_x, pixels_per_mm_y)
    hsv = cv2.cvtColor(image_calibrated, cv2.COLOR_BGR2HSV)
    plt.figure()
    plt.imshow(image)
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.title('Original Image')
    plt.figure()
    plt.imshow(image_calibrated)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Calibrated Image')
    # plt.show()
    plt.figure()
    plt.imshow(hsv)
    plt.title('Calibrated HSV Image')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.show()

    #as using hsv image, the lower_color and upper_color bounds need to be in hsv colorspace
    mask = cv2.inRange(hsv, lower_color, upper_color)
    plt.figure()
    plt.imshow(mask)
    plt.title('Mask')
    plt.show()

    #blur mask
    # mask = cv2.medianBlur(mask, kernel_size)
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()
    #find contours of mask
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    max_contour = max(contours, key=cv2.contourArea)

    # max contour doesn't always find the whole rail!!!!!!!!!!!
    image_copy = image.copy()
    cv2.drawContours(image_copy, [max_contour], 0, (255,0,0),2)
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB), interpolation='none')
    plt.axis('off')
    plt.show()

    x_vals = np.unique(max_contour[:, :, 0])
    y_values = np.zeros_like(x_vals)
    for i, x_val in enumerate(x_vals):
        y_values[i] = np.max(max_contour[max_contour[:, :, 0] == x_val][:, 1])
    smoothing_factor = .1
    x_new = np.linspace(x_vals.min(), x_vals.max(), num=10000)
    y_new = np.interp(x_new, x_vals, y_values)
    spline_new = UnivariateSpline(x_new, y_new, k=3, s=smoothing_factor)
    y_new_smooth = spline_new(x_new)
    if np.any(np.diff(y_new_smooth) < -500):
        print('Discontinuous curve detected, applying fix...')
        y_new_smooth_fixed = []
        for i in range(len(x_new)):
            if i == 0:
                y_new_smooth_fixed.append(ynew_smooth[i])
            elif y_new_smooth[i] - y_new_smooth[i-1] < -500:
                y_new_smooth_fixed.append(y_new_smooth_fixed[i-1])
            else:
                y_new_smooth_fixed.append(y_new_smooth[i])
        y_new_smooth = np.array(y_new_smooth_fixed)
    # y_new_smooth = np.max(y_values) - y_new_smooth
    return image, mask, x_new, y_new_smooth

# def plot_image_and_spline(img, mask, x_vals, y_smooth):
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
#     ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto')
#     ax2.imshow(mask, aspect='auto')
#     ax3.plot(x_vals, y_smooth)
#     ax3.tick_params(axis='y', which='both', length=0)
#     ax2.set_xlim([0, img.shape[1]])
#     ax3.set_xlim([0, img.shape[1]])
#     ax2.set_ylim([img.shape[0], 0])
#     ax3.set_ylim([img.shape[0], 0])
#     ax1.set_title('Original Image')
#     ax2.set_title('Mask')
#     ax3.set_title('Spline Curve')
#     ax3.set_xlabel('X')
#     ax3.set_ylabel('Y')
#     plt.show()

def downsample_spline_curve(x_new, y_new_smooth, num_points):
    " This downsamples the spline curve for visualisation and analysis purposes"
    # Generate a range of x-values for the spline curve
    x_range = np.linspace(x_new.min(), x_new.max(), num=len(x_new)*10)

    # Interpolate the y-values of the spline curve at each x_range value
    y_range = np.interp(x_range, x_new, y_new_smooth)

    # Fit a spline curve to the interpolated data
    smoothing_factor = 0.1
    spline_new = UnivariateSpline(x_range, y_range, k=3, s=smoothing_factor)

    # Evaluate the spline curve at the x_range values
    y_range_smooth = spline_new(x_range)

    # Stack the x_range and y_range arrays into a 2D array of (x, y) coordinate pairs
    spline_curve = np.column_stack((x_range, y_range_smooth))

    # Downsample the spline curve to num_points points
    length = len(spline_curve)
    stride = int(np.ceil(length / num_points))
    spline_curve_downsampled = spline_curve[::stride]

    return spline_curve_downsampled


def plot_calibrated_spline(x, y):
    plt.figure(figsize=(10,8))
    plt.plot(x,y)
    plt.title('Calibrated Curve')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    label_text = 'Radius: 50 mm, Camera angle: 20 degrees (CAD model)'
    plt.text(0.5, 0.1, label_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    return

def plot_all_data():
    curve_110 = test_curve('./experiment_images_010523/green/red_green_30/110mm_green_iter1.jpg')
    curve_90 = test_curve('./experiment_images_010523/green/red_green_20/90mm_green_iter1.jpg')
    curve_70 = test_curve('./experiment_images_010523/green/red_green_20/70mm_green_iter1.jpg')
    curve_50 = test_curve('./experiment_images_010523/green/red_green_20/50mm_green_iter1.jpg')
    plt.figure(figsize=(10, 8))
    plt.plot(curve_110[:,0], curve_110[:,1], label = '110mm')
    plt.plot(curve_90[:, 0], curve_90[:, 1], label = '90mm')
    plt.plot(curve_70[:, 0], curve_70[:, 1], label = '70mm')
    plt.plot(curve_50[:, 0], curve_50[:, 1], label = '50mm')
    plt.title('Calibrated Curves')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    # plt.xlim(0, 125)
    # plt.ylim(0, 200)
    label_text = 'Camera angle: 0 degrees (CAD model)'
    plt.text(0.5, 0.1, label_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(title='Radius', loc='upper right')
    plt.show()
    return



def test_curve():
    calibration_images_path = './checkerboard_calibration_images/*.jpg'

    checkerboard_size = (5,7)
    camera_matrix, dist_coeffs = calibrate_camera(calibration_images_path, checkerboard_size)
    print(camera_matrix, dist_coeffs)

    file_path = './experiment_images_110523/blue/calibration_blue_iter1.jpg'

    color_name = 'blue'
    real_width = 85  # mm
    real_length = 4  # mm
    calibration_filepath = './experiment_images_110523/blue/calibration_blue_iter1.jpg'
    calibration_image = cv2.imread(calibration_filepath)
    # plt.figure()
    # plt.imshow(calibration_image)
    # plt.show()
    lower_color, upper_color = get_color_boundaries(color_name)
    image = cv2.imread(file_path)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # cv2.imshow(image)
    pixels_per_mm_x, pixels_per_mm_y = get_calibration_matrix(calibration_image, real_width, real_length, color_name)
    # image_resized = calibrate_image(undistorted_image, pixels_per_mm_x, pixels_per_mm_y)
    img, mask, x_new, y_new_smooth = process_image(undistorted_image, lower_color, upper_color, pixels_per_mm_x, pixels_per_mm_y)

    spline_curve_downsampled = downsample_spline_curve(x_new,y_new_smooth,100)

    # plot_calibrated_spline(spline_curve_downsampled[:,0],spline_curve_downsampled[:,1])
    return spline_curve_downsampled


def approximate_spline(filepath, color_name, real_width, real_length, calibration_filepath):
    calibration_image = cv2.imread(calibration_filepath)
    lower_color, upper_color = get_color_boundaries(color_name)
    image = cv2.imread(filepath)
    pixels_per_mm_x, pixels_per_mm_y = get_calibration_matrix(calibration_image, real_width, real_length, color_name)
    img, mask, x_new, y_new_smooth = process_image(image, lower_color, upper_color, pixels_per_mm_x, pixels_per_mm_y)
    spline_curve_downsampled = downsample_spline_curve(x_new, y_new_smooth, 100)
    return spline_curve_downsampled

def get_circle(radius, center):
    theta = np.linspace(0,2*np.pi, 100)

    # Calculate circle coordinates
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y

def main():
    curve = test_curve()
    curvature, circle_center = circle_fitting_method(curve)
    radius = 1.0 / curvature
    circle_x, circle_y = get_circle(radius, circle_center)
    # print(curve)
    # curve = test_curve('./experiment_images_110523/blue/70mm_blue_iter1.jpg')
    # print(curve)
    # print(len(curve))
    # column_a = curve[15:-20,0]
    # column_b = curve[15:-20,1]
    # main_curve = np.column_stack((column_a, column_b))
    # # print(main_curve)
    # plot_calibrated_spline(curve[:,0], curve[:,1])
    plt.figure()
    plt.figure(figsize=(10, 8))
    plt.plot(circle_x, circle_y)
    plt.plot(curve[:,0], curve[:,1])
    plt.title('Calibrated Curve/Derivatives Approximation')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    # plt.xlim(0, 200)
    # plt.ylim(0, 200)
    # plt.show()



    # print('Curvature:')
    # print(curvature)
    print('Radius:')
    print(radius)


if __name__ == '__main__':
    main()


