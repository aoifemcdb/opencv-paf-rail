import cv2
import numpy as np
from colour_shape_sensing.shape_sensing import ShapeSensing
from calibration import Calibration
import os
from color_boundaries import ColorBoundaries
from scipy.interpolate import UnivariateSpline
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

def transform_points(points):
    # Get the x and y coordinates of the first point
    x_offset = points[0][0][0]
    y_offset = points[0][0][1]

    # Apply the transformation to all points
    transformed_points = [(point[0][0] - x_offset, point[0][1] - y_offset) for point in points]

    return transformed_points

def inverse_transform_points(points, x_offset, y_offset):
    # Apply the inverse transformation to all points
    original_points = [(point[0] + x_offset, point[1] + y_offset) for point in points]

    return original_points



# def process_image(image, color, kernel_size=5):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     color_boundaries = ColorBoundaries()
#     lower_color, upper_color = color_boundaries.get_color_boundaries(color)
#     mask = cv2.inRange(hsv, lower_color, upper_color)
#
#     contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
#
#     if len(contours) > 0:
#         largest_contour = max(contours, key=cv2.contourArea)
#
#         # Convert the contour to a NumPy array
#         transformed_contour = np.squeeze(largest_contour)
#
#         # Get the first coordinate pair as the reference point
#         x_offset, y_offset = transformed_contour[0]
#
#         # Transform the contour points by subtracting the reference point
#         transformed_contour[:, 0] -= x_offset
#         transformed_contour[:, 1] -= y_offset
#
#         # Perform the selection of maximum y-values on the transformed contour
#         x_vals = np.unique(transformed_contour[:, 0])
#         y_values = np.zeros_like(x_vals)
#         for i, x_val in enumerate(x_vals):
#             indices = np.where(transformed_contour[:, 0] == x_val)[0]
#             if len(indices) > 0:
#                 y_values[i] = np.max(transformed_contour[indices, 1])
#
#         # Inverse transform the selected points
#         selected_points = np.column_stack((x_vals, y_values))
#         selected_points[:, 0] += x_offset
#         selected_points[:, 1] += y_offset
#
#         # Additional code for smoothing and fixing the curve
#         if len(selected_points) >= 4:  # Check if there are at least 4 points for spline interpolation
#             smoothing_factor = 0.1
#             spline_new = UnivariateSpline(selected_points[:, 0], selected_points[:, 1], k=3, s=smoothing_factor)
#             x_new = np.linspace(selected_points[:, 0].min(), selected_points[:, 0].max(), num=10000)
#             y_new_smooth = spline_new(x_new)
#
#             if np.any(np.diff(y_new_smooth) < -500):
#                 print('Discontinuous curve detected, applying fix...')
#                 y_new_smooth_fixed = []
#                 for i in range(len(x_new)):
#                     if i == 0:
#                         y_new_smooth_fixed.append(y_new_smooth[i])
#                     elif y_new_smooth[i] - y_new_smooth[i - 1] < -500:
#                         y_new_smooth_fixed.append(y_new_smooth_fixed[i - 1])
#                     else:
#                         y_new_smooth_fixed.append(y_new_smooth[i])
#                 y_new_smooth = np.array(y_new_smooth_fixed)
#
#             # Inverse transform the smoothed curve
#             smoothed_curve = np.column_stack((x_new, y_new_smooth))
#             smoothed_curve[:, 0] += x_offset
#             smoothed_curve[:, 1] += y_offset
#
#             # Additional code for visualization or further processing
#             # ...
#
#             return image, mask, smoothed_curve[:, 0], smoothed_curve[:, 1]
#
#         else:
#             print("Insufficient data points for spline interpolation.")
#             return None, None, None, None
#
#     else:
#         print("No contours found.")
#         return None, None, None, None

# #
def process_image(image, color, kernel_size=5):
    if image is None:
        return None, None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_boundaries = ColorBoundaries()
    lower_color, upper_color = color_boundaries.get_color_boundaries(color)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Sort contour points by y-coordinate in descending order
        sorted_contour = sorted(largest_contour, key=lambda point: point[0, 1], reverse=True)

        # Extract x and y coordinates of the bottom edge of the contour
        x_coords = [point[0, 0] for point in sorted_contour]
        y_coords = [point[0, 1] for point in sorted_contour]

        smoothing_factor = 0.1
        x_new = np.linspace(min(x_coords), max(x_coords), num=1000)
        y_new = np.interp(x_new, x_coords, y_coords)
        spline_new = UnivariateSpline(x_new, y_new, k=3, s=smoothing_factor)
        y_new_smooth = spline_new(x_new)

        return x_new, y_new_smooth
    else:
        return None, None

# def process_image(image, color):
#     # if image is None:
#     #     return None, None
#
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     color_boundaries = ColorBoundaries()
#     lower_color, upper_color = color_boundaries.get_color_boundaries(color)
#
#     mask = cv2.inRange(hsv, lower_color, upper_color)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) > 0:
#         # Find the contour with the largest area
#         largest_contour = max(contours, key=cv2.contourArea)
#
#         # Get the bounding rectangle of the contour
#         x, y, w, h = cv2.boundingRect(largest_contour)
#
#         # Extract the region of interest (ROI) from the mask based on the bounding rectangle
#         roi_mask = mask[y:y + h, x:x + w]
#
#         # Find contours within the ROI
#         roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         if len(roi_contours) > 0:
#             # Find the contour with the largest area within the ROI
#             largest_roi_contour = max(roi_contours, key=cv2.contourArea)
#
#             # Find the top and bottom edges of the contour
#             top_edge = largest_roi_contour[largest_roi_contour[:, :, 1].argmin()][0] + (x, y)
#             bottom_edge = largest_roi_contour[largest_roi_contour[:, :, 1].argmax()][0] + (x, y)
#
#             # Calculate the midpoints between the top and bottom edges
#             midpoints = np.vstack((top_edge, bottom_edge))
#
#             # Extract the x and y coordinates of the midpoints
#             x_coords = midpoints[:, 0]
#             y_coords = midpoints[:, 1]
#
#             smoothing_factor = 0.1
#             x_new = np.linspace(min(x_coords), max(x_coords), num=1000)
#             y_new = np.interp(x_new, x_coords, y_coords)
#             spline_new = UnivariateSpline(x_new, y_new, k=3, s=smoothing_factor)
#             y_new_smooth = spline_new(x_new)
#
#             return x_new, y_new_smooth
#         else:
#             return None, None
#     else:
#         return None, None


def downsample_spline_curve(x_new, y_new_smooth, num_points=10):
    x_range = np.linspace(x_new.min(), x_new.max(), num=len(x_new)*10)
    y_range = np.interp(x_range, x_new, y_new_smooth)
    smoothing_factor = 0.1
    spline_new = UnivariateSpline(x_range, y_range, k=3, s=smoothing_factor)
    y_range_smooth = spline_new(x_range)
    spline_curve = np.column_stack((x_range, y_range_smooth))
    length = len(spline_curve)
    stride = int(np.ceil(length / num_points))
    spline_curve_downsampled = spline_curve[::stride]
    return spline_curve_downsampled

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 0)

    # Process the frame to detect shapes and approximate spline
    color = 'green'
    x_new, y_new_smooth = process_image(frame, color)

    if x_new is not None and y_new_smooth is not None:
        # y_new_smooth = -y_new_smooth
        spline_curve_downsampled = downsample_spline_curve(x_new, y_new_smooth)

        # Display the processed frame on the screen
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.polylines(frame, [np.int32(spline_curve_downsampled)], False, (255, 0, 0), 2)
        cv2.imshow('frame', frame)

        # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()