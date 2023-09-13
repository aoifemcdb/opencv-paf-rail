from shape_sensing import ShapeSensing
import cv2
import numpy as np
from colour_shape_sensing.shape_sensing import ShapeSensing
from calibration import Calibration
import os
from color_boundaries import ColorBoundaries
from scipy.interpolate import UnivariateSpline
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
out = cv2.VideoWriter('shape_sensing_8.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))


while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()
    frame_height = frame.shape[0]
    print(frame_height)
    # frame = cv2.flip(frame, 0)

    # Process the frame to detect shapes and approximate spline
    color = 'green'
    colour_boundaries = ColorBoundaries()
    lower_color, upper_color = colour_boundaries.get_color_boundaries(color)
    shape_sensor = ShapeSensing()
    image, mask, x_new, y_new_smooth = shape_sensor.process_image(frame, lower_color, upper_color)
    print(y_new_smooth)
    # y_new_smooth = y_new_smooth[0] - y_new_smooth

    if x_new is not None and y_new_smooth is not None:
        y_mean = np.mean(y_new_smooth)
        y_new_smooth = 1.9 * y_mean - y_new_smooth
        spline_curve_downsampled = shape_sensor.downsample_spline_curve(x_new, y_new_smooth, num_points=25)

        # Display the processed frame on the screen
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.polylines(frame, [np.int32(spline_curve_downsampled)], False, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        out.write(frame)
        # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windowsq
cap.release()
cv2.destroyAllWindows()