import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from calibration import Calibration
from colour_shape_sensing.color_boundaries import ColorBoundaries
import matplotlib as mpl

# Set the font family to match LaTeX font
mpl.rcParams['font.family'] = 'serif'


""" to get the full shape, not all x values have a unique mapping, 
they go back on themselves particularly for tighter curves. need to add a
statement for this. """

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

class ShapeSensing:
    def __init__(self):
        self.calibration = Calibration()


    def process_image(self, image, lower_color, upper_color, kernel_size=5, plot_images=False):
        # image = rotate_image(image, -90)
        # rot_h, rot_w = image.shape[:2]
        # rot_corners = np.array([[0, 0], [rot_w, 0], [rot_w, rot_h], [0, rot_h]], dtype=np.float32)
        # M = cv2.getRotationMatrix2D((rot_w / 2, rot_h / 2), 90, 1.0)
        # rotated_corners = cv2.transform(np.array([rot_corners]), M)[0]


        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if plot_images:
            plt.figure()
            plt.imshow(image)
            plt.xlabel('Pixels')
            plt.ylabel('Pixels')
            plt.title('Original Image')
            plt.figure()
            plt.imshow(hsv)
            plt.title('Calibrated HSV Image')
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.show()

        mask = cv2.inRange(hsv, lower_color, upper_color)
        if plot_images:
            plt.figure()
            plt.imshow(mask)
            plt.title('Mask')
            plt.show()

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        # print(contours)

        max_contour = max(contours, key=cv2.contourArea)

        image_copy = image.copy()
        cv2.drawContours(image_copy, [max_contour], 0, (255, 0, 0), 2)
        if plot_images:
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
        # if np.any(np.diff(y_new_smooth) < -500):
        #     print('Discontinuous curve detected, applying fix...')
        #     y_new_smooth_fixed = []
        #     for i in range(len(x_new)):
        #         if i == 0:
        #             y_new_smooth_fixed.append(y_new_smooth[i])
        #         elif y_new_smooth[i] - y_new_smooth[i - 1] < -500:
        #             y_new_smooth_fixed.append(y_new_smooth_fixed[i - 1])
        #         else:
        #             y_new_smooth_fixed.append(y_new_smooth[i])
        #     y_new_smooth = np.array(y_new_smooth_fixed)

        image_copy_line = image.copy()
        if plot_images:
            # fig1=plt.figure()
            plt.imshow(image_copy_line, origin='upper')
            x_new = x_new[1000:9500]
            y_new_smooth = y_new_smooth[1000:9500]
            plt.plot(x_new, y_new_smooth, color = 'y', label='Sensed Shape')
            # plt.xlabel('x (mm)')
            # plt.ylabel('y (mm)')
            # plt.xlim(rotated_corners[:, 0].min(), rotated_corners[:, 0].max())
            # plt.ylim(rotated_corners[:, 1].min(), rotated_corners[:, 1].max())
            plt.axis('off')

            plt.legend()

            # dpi=300
            # plt.savefig('phantom_with_sensed_rail.png', dpi=dpi)
            # fig1_width, fig1_height = fig1.get_size_inches()
            # print(fig1_width, fig1_height)
            plt.show()


        return image, mask, x_new, y_new_smooth

    def downsample_spline_curve(self, x_new, y_new_smooth, num_points, plot_curve=False):
        x_range = np.linspace(x_new.min(), x_new.max(), num=len(x_new) * 10)
        y_range = np.interp(x_range, x_new, y_new_smooth)
        smoothing_factor = 0.1
        spline_new = UnivariateSpline(x_range, y_range, k=3, s=smoothing_factor)
        y_range_smooth = spline_new(x_range)
        spline_curve = np.column_stack((x_range, y_range_smooth))
        length = len(spline_curve)
        stride = int(np.ceil(length / num_points))
        spline_curve_downsampled = (spline_curve[::stride])

        max_y = np.max(spline_curve_downsampled[:, 1])
        y_flipped = max_y - spline_curve_downsampled[:, 1] + max_y  # Flip the y-coordinates
        spline_curve_flipped = np.column_stack((spline_curve_downsampled[:, 0], y_flipped))

        if plot_curve:
            plt.plot(x_range, y_range_smooth, color='white')
            plt.plot(spline_curve_flipped[:, 0], spline_curve_flipped[:, 1], label='Spline Curve')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Spline Curve Shape Approximation')
            plt.legend()
            plt.show()

        return spline_curve_flipped


    def transform_points(self, points):
        # Get the x and y coordinates of the first point
        x_offset = points[0][0][0]
        y_offset = points[0][0][1]

        # Apply the transformation to all points
        transformed_points = [(point[0][0] - x_offset, point[0][1] - y_offset) for point in points]

        return transformed_points

    def inverse_transform_points(self, points, x_offset, y_offset):
        # Apply the inverse transformation to all points
        original_points = [(point[0] + x_offset, point[1] + y_offset) for point in points]

        return original_points

    def process_image_stereo(self, image_left, image_right, lower_color, upper_color, kernel_size=5, plot_images=False):
        hsv_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2HSV)
        hsv_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2HSV)

        if plot_images:
            plt.figure()
            plt.imshow(image_left)
            plt.xlabel('Pixels')
            plt.ylabel('Pixels')
            plt.title('Original Left Image')
            plt.figure()
            plt.imshow(image_right)
            plt.xlabel('Pixels')
            plt.ylabel('Pixels')
            plt.title('Original Right Image')
            plt.show()

        mask_left = cv2.inRange(hsv_left, lower_color, upper_color)
        mask_right = cv2.inRange(hsv_right, lower_color, upper_color)

        if plot_images:
            plt.figure()
            plt.imshow(mask_left)
            plt.title('Left Mask')
            plt.figure()
            plt.imshow(mask_right)
            plt.title('Right Mask')
            plt.show()

        contours_left = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        max_contour_left = max(contours_left, key=cv2.contourArea)

        contours_right = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        max_contour_right = max(contours_right, key=cv2.contourArea)

        image_left_copy = image_left.copy()
        image_right_copy = image_right.copy()

        cv2.drawContours(image_left_copy, [max_contour_left], 0, (255, 0, 0), 2)
        cv2.drawContours(image_right_copy, [max_contour_right], 0, (255, 0, 0), 2)

        if plot_images:
            plt.imshow(cv2.cvtColor(image_left_copy, cv2.COLOR_BGR2RGB), interpolation='none')
            plt.axis('off')
            plt.title('Processed Left Image')
            plt.show()

            plt.imshow(cv2.cvtColor(image_right_copy, cv2.COLOR_BGR2RGB), interpolation='none')
            plt.axis('off')
            plt.title('Processed Right Image')
            plt.show()

        x_vals_left = np.unique(max_contour_left[:, :, 0])
        y_values_left = np.zeros_like(x_vals_left)
        for i, x_val in enumerate(x_vals_left):
            y_values_left[i] = np.max(max_contour_left[max_contour_left[:, :, 0] == x_val][:, 1])

        x_vals_right = np.unique(max_contour_right[:, :, 0])
        y_values_right = np.zeros_like(x_vals_right)
        for i, x_val in enumerate(x_vals_right):
            y_values_right[i] = np.max(max_contour_right[max_contour_right[:, :, 0] == x_val][:, 1])

        smoothing_factor = 0.1

        x_new_left = np.linspace(x_vals_left.min(), x_vals_left.max(), num=10000)
        y_new_left = np.interp(x_new_left, x_vals_left, y_values_left)
        spline_new_left = UnivariateSpline(x_new_left, y_new_left, k=3, s=smoothing_factor)
        y_new_smooth_left = spline_new_left(x_new_left)

        x_new_right = np.linspace(x_vals_right.min(), x_vals_right.max(), num=10000)
        y_new_right = np.interp(x_new_right, x_vals_right, y_values_right)
        spline_new_right = UnivariateSpline(x_new_right, y_new_right, k=3, s=smoothing_factor)
        y_new_smooth_right = spline_new_right(x_new_right)

        if np.any(np.diff(y_new_smooth_left) < -500):
            print('Discontinuous curve detected in the left image, applying fix...')
            y_new_smooth_fixed = []
            for i in range(len(x_new_left)):
                if i == 0:
                    y_new_smooth_fixed.append(y_new_smooth_left[i])
                elif y_new_smooth_left[i] - y_new_smooth_left[i - 1] < -500:
                    y_new_smooth_fixed.append(y_new_smooth_fixed[i - 1])
                else:
                    y_new_smooth_fixed.append(y_new_smooth_left[i])
            y_new_smooth_left = np.array(y_new_smooth_fixed)

        if np.any(np.diff(y_new_smooth_right) < -500):
            print('Discontinuous curve detected in the right image, applying fix...')
            y_new_smooth_fixed = []
            for i in range(len(x_new_right)):
                if i == 0:
                    y_new_smooth_fixed.append(y_new_smooth_right[i])
                elif y_new_smooth_right[i] - y_new_smooth_right[i - 1] < -500:
                    y_new_smooth_fixed.append(y_new_smooth_fixed[i - 1])
                else:
                    y_new_smooth_fixed.append(y_new_smooth_right[i])
            y_new_smooth_right = np.array(y_new_smooth_fixed)

        # Combine the left and right curves
        x_new_combined = np.concatenate((x_new_left, x_new_right))
        y_new_smooth_combined = np.concatenate((y_new_smooth_left, y_new_smooth_right))

        # Sort the combined curve based on x-values
        sorted_indices = np.argsort(x_new_combined)
        x_new_combined_sorted = x_new_combined[sorted_indices]
        y_new_smooth_combined_sorted = y_new_smooth_combined[sorted_indices]

        spline_curve_downsampled = self.downsample_spline_curve(
            x_new_combined_sorted, y_new_smooth_combined_sorted, num_points=100, plot_curve=plot_images
        )

        return image_left, image_right, mask_left, mask_right, x_new_combined_sorted, y_new_smooth_combined_sorted, spline_curve_downsampled

    def process_video(self, image, color, kernel_size=5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_boundaries = ColorBoundaries()
        lower_color, upper_color = color_boundaries.get_color_boundaries(color)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            # Convert the contour to a NumPy array
            transformed_contour = np.squeeze(largest_contour)

            # Get the first coordinate pair as the reference point
            x_offset, y_offset = transformed_contour[0]

            # Transform the contour points by subtracting the reference point
            transformed_contour[:, 0] -= x_offset
            transformed_contour[:, 1] -= y_offset

            # Perform the selection of maximum y-values on the transformed contour
            x_vals = np.unique(transformed_contour[:, 0])
            y_values = np.zeros_like(x_vals)
            for i, x_val in enumerate(x_vals):
                indices = np.where(transformed_contour[:, 0] == x_val)[0]
                if len(indices) > 0:
                    y_values[i] = np.max(transformed_contour[indices, 1])

            # Inverse transform the selected points
            selected_points = np.column_stack((x_vals, y_values))
            selected_points[:, 0] += x_offset
            selected_points[:, 1] += y_offset

            # Additional code for smoothing and fixing the curve
            if len(selected_points) >= 4:  # Check if there are at least 4 points for spline interpolation
                smoothing_factor = 0.1
                spline_new = UnivariateSpline(selected_points[:, 0], selected_points[:, 1], k=3, s=smoothing_factor)
                x_new = np.linspace(selected_points[:, 0].min(), selected_points[:, 0].max(), num=10000)
                y_new_smooth = spline_new(x_new)

                if np.any(np.diff(y_new_smooth) < -500):
                    print('Discontinuous curve detected, applying fix...')
                    y_new_smooth_fixed = []
                    for i in range(len(x_new)):
                        if i == 0:
                            y_new_smooth_fixed.append(y_new_smooth[i])
                        elif y_new_smooth[i] - y_new_smooth[i - 1] < -500:
                            y_new_smooth_fixed.append(y_new_smooth_fixed[i - 1])
                        else:
                            y_new_smooth_fixed.append(y_new_smooth[i])
                    y_new_smooth = np.array(y_new_smooth_fixed)

                # Inverse transform the smoothed curve
                smoothed_curve = np.column_stack((x_new, y_new_smooth))
                smoothed_curve[:, 0] += x_offset
                smoothed_curve[:, 1] += y_offset

                # Additional code for visualization or further processing
                # ...

                return image, mask, smoothed_curve[:, 0], smoothed_curve[:, 1]

            else:
                print("Insufficient data points for spline interpolation.")
                return None, None, None, None

        else:
            print("No contours found.")
            return None, None, None, None




# # Example usage (plot_images=True will plot all the images of the intermediary steps):
# shape_sensing = ShapeSensing()
# image = cv2.imread('./experiment_images_180923/pilot_test/results/original_image.jpg')
# lower_color = (35, 50, 50)
# upper_color = (90, 255, 255)
# result = shape_sensing.process_image(image, lower_color, upper_color, plot_images=True)
# curve_downsampled = shape_sensing.downsample_spline_curve(result[2], result[3], 100)

