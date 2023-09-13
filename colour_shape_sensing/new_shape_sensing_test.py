import cv2
import numpy as np
import matplotlib.pyplot as plt
from colour_shape_sensing.calibration import Calibration
from colour_shape_sensing.color_boundaries import ColorBoundaries
from scipy.interpolate import UnivariateSpline
import matplotlib as mpl
from curvature_methods import CurvatureCalculator

# Set the font family to match LaTeX font
mpl.rcParams['font.family'] = 'serif'

import cv2
import numpy as np

# Global list to store clicked points
points = []


def click_event(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)

        if len(points) == 2:
            cv2.destroyAllWindows()


def select_points_on_image(img_path, lower_color, upper_color, save_plots=False):
    global img, points

    img = cv2.imread(img_path)
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    if save_plots:
        plt.savefig('original_image.png', dpi=300)
    plt.show()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)  # Draw the contour in green

    # # Resize the image for display
    # display_scale = 0.5  # Modify this value as needed
    # img = cv2.resize(img, (int(img.shape[1] * display_scale), int(img.shape[0] * display_scale)))

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)

    # Plot and save (if required) original image


    # Plot and save (if required) HSV thresholding
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.imshow(hsv)
    plt.title('HSV Image')
    plt.axis('off')
    if save_plots:
        plt.savefig('hsv_image.png', dpi=300)
    plt.show()

    # Plot and save (if required) mask
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    if save_plots:
        plt.savefig('mask.png', dpi=300)
    plt.show()

    cv2.waitKey(0)
    return points, mask


def get_contour_portion(contour, point1, point2):
    # Calculate the distances of all contour points to the first clicked point
    distances_to_point1 = np.sqrt(((contour - point1) ** 2).sum(axis=2))

    # Get the index of the closest contour point to the first clicked point
    index1 = np.argmin(distances_to_point1)

    # Repeat for the second clicked point
    distances_to_point2 = np.sqrt(((contour - point2) ** 2).sum(axis=2))
    index2 = np.argmin(distances_to_point2)

    # Extract the portion of the contour between the two indices
    if index1 < index2:
        portion = contour[index1:index2 + 1]
    else:
        portion = contour[index2:index1 + 1]

    return portion


def approximate_spline(portion):
    x_vals = np.unique(portion[:, :, 0])
    y_values = np.zeros_like(x_vals)
    for i, x_val in enumerate(x_vals):
        y_values[i] = np.max(portion[portion[:, :, 0] == x_val][:, 1])
    smoothing_factor = .1
    x_new = np.linspace(x_vals.min(), x_vals.max(), num=10000)
    y_new = np.interp(x_new, x_vals, y_values)
    spline_new = UnivariateSpline(x_new, y_new, k=3, s=smoothing_factor)
    y_new_smooth = spline_new(x_new)

    return x_new, y_new_smooth

def overlay_spline_on_image(img_path, x_new, y_new_smooth):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using imshow
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.imshow(image_rgb)

    # Overlay the spline curve using plot

    plt.plot(x_new, y_new_smooth, 'r-', linewidth=2)
    plt.title('Spline Approximation')
    plt.axis('off')  # Hide the axes for better visualization
    # plt.savefig('spline_30.png', dpi=300)
    plt.show()


def plot_contour_points(img_path, contour, point1, point2, save_plots=False):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find the nearest contour points to the selected points
    idx1 = np.argmin(np.sum((contour - point1) ** 2, axis=2))
    idx2 = np.argmin(np.sum((contour - point2) ** 2, axis=2))

    nearest_point1 = tuple(contour[idx1][0])
    nearest_point2 = tuple(contour[idx2][0])

    # Plot the original contour in green
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'b', linewidth=1)

    cv2.circle(image_rgb, nearest_point1, 10, (255, 0, 0), -1)  # Draw a red circle around the nearest point
    cv2.circle(image_rgb, nearest_point2, 10, (255, 0, 0), -1)

    plt.imshow(image_rgb)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Contour Region for Spline Approximation')

    if save_plots:
        plt.savefig('contour_points.png', dpi=300)
    plt.show()







# Call the function
calibration = Calibration()
checkerboard_size = (5, 7)
calibration_images_path = './experiment_images_260723/calibration/*.jpg'
camera_matrix, dist_coeffs = calibration.calibrate_camera(calibration_images_path, checkerboard_size)

color = 'green'
color_boundaries=ColorBoundaries()
lower_color, upper_color = color_boundaries.get_color_boundaries(color)


(point1, point2), mask = select_points_on_image('./experiment_images_260623/rails/30mm/WIN_20230626_10_47_45_Pro.jpg', lower_color, upper_color, save_plots=True)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)
portion = get_contour_portion(contour, point1, point2)
x_new, y_new_smooth = approximate_spline(portion)
overlay_spline_on_image('./experiment_images_260623/rails/30mm/WIN_20230626_10_47_45_Pro.jpg', x_new, y_new_smooth)
plot_contour_points('./experiment_images_260623/rails/30mm/WIN_20230626_10_47_45_Pro.jpg', contour, point1, point2, save_plots=False)
curve_downsampled = np.column_stack((x_new, y_new_smooth))
curvature, _ = CurvatureCalculator.circle_fitting_method(curve_downsampled)
radius = 1.0 / curvature
radius = radius * 0.1
print(radius )