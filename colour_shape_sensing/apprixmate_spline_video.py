import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from color_boundaries import get_color_boundaries


def process_image(image, color):
    if image is None:
        return None, None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = get_color_boundaries(color)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        x_vals = np.unique(largest_contour[:, :, 0])
        y_values = np.zeros_like(x_vals)
        for i, x_val in enumerate(x_vals):
            y_values[i] = np.max(largest_contour[largest_contour[:, :, 0] == x_val][:, 1])

            # # Adjust the y-coordinate of contour points
            # max_y = np.max(y_values)
            # y_values = max_y - y_values

        smoothing_factor = 0.1
        x_new = np.linspace(x_vals.min(), x_vals.max(), num=1000)
        y_new = np.interp(x_new, x_vals, y_values)
        spline_new = UnivariateSpline(x_new, y_new, k=3, s=smoothing_factor)
        y_new_smooth = spline_new(x_new)

        if np.any(np.diff(y_new_smooth) < -500):
            print('Discontinuous curve detected, applying fix...')
            y_new_smooth_fixed = []
            for i in range(len(x_new)):
                if i == 0:
                    y_new_smooth_fixed.append(y_new_smooth[i])
                # elif y_new_smooth[i] - y_new_smooth[i-1] < -500:
                    y_new_smooth_fixed.append(y_new_smooth_fixed[i-1])
                else:
                    y_new_smooth_fixed.append(y_new_smooth[i])
            y_new_smooth = np.array(y_new_smooth_fixed)

        # y_new_smooth = np.max(y_values) - y_new_smooth

        return x_new, y_new_smooth
    else:
        return None, None


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

    # Process the frame to detect shapes and approximate spline curve
    x_new, y_new_smooth = process_image(frame, 'green')

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



