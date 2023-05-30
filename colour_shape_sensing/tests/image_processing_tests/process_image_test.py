import cv2
import matplotlib.pyplot as plt
from color_boundaries import get_color_boundaries
from scipy.interpolate import UnivariateSpline
import numpy as np


def process_image(image, lower_color, upper_color):
    image_calibrated = image
    hsv = cv2.cvtColor(image_calibrated, cv2.COLOR_BGR2HSV)
    # plt.figure()
    # plt.imshow(image)
    # plt.xlabel('Pixels')
    # plt.ylabel('Pixels')
    # plt.title('Original Image')
    # plt.figure()
    # plt.imshow(image_calibrated)
    # plt.xlabel('X (mm)')
    # plt.ylabel('Y (mm)')
    # plt.title('Calibrated Image')
    # # plt.show()
    # plt.figure()
    # plt.imshow(hsv)
    # plt.title('Calibrated HSV Image')
    # plt.show()

    #as using hsv image, the lower_color and upper_color bounds need to be in hsv colorspace
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

    #blur mask
    # mask = cv2.medianBlur(mask, kernel_size=5)
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()
    #find contours of mask
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    max_contour = max(contours, key=cv2.contourArea)

    # Create a blank image with the same dimensions as the original image
    blank_image = np.zeros_like(image_calibrated)

    # Draw the contour on the blank image
    cv2.drawContours(blank_image, [max_contour], 0, (0, 255, 0), 2)

    # max contour doesn't always find the whole rail!!!!!!!!!!!
    image_copy = image_calibrated.copy()
    cv2.drawContours(image_copy, [max_contour], 0, (100,255,0),2)
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB), interpolation='none')
    plt.axis('off')
    plt.show()

    x_vals = np.unique(max_contour[:, :, 0])
    y_values = np.zeros_like(x_vals)
    for i, x_val in enumerate(x_vals):
        y_values[i] = np.max(max_contour[max_contour[:, :, 0] == x_val][:, 1])
    smoothing_factor = .1
    x_new = np.linspace(x_vals.min(), x_vals.max(), num=1000)
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

    plt.imshow(blank_image)
    plt.plot(x_new, y_new_smooth, color='red', linewidth=2, label='Spline Approximation')
    plt.plot([], [], color='green', linewidth=2, label='Contour')
    plt.legend()
    plt.axis('off')
    plt.show()
    return image_calibrated, mask, x_new, y_new_smooth


filepath = '../../experiment_images_110523/green/30mm_green_iter5.jpg'
image = cv2.imread(filepath)
color = 'green'
lower_color, upper_color = get_color_boundaries(color)
process_image(image, lower_color, upper_color)