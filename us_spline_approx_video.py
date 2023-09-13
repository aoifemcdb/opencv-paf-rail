import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import threading
from sksurgerybk.interface.bk5000 import BK5000


# Setup and connect to BK
timeout = 5
frames_per_second = 25

ip = '128.16.0.3'  # Default IP of BK5000
port = 7915

# # bk = BK5000(timeout=timeout, frames_per_second=frames_per_second)
# # bk.connect_to_host(ip, port)
# bk.query_win_size()
# bk.start_streaming()
#
# # Get a single frame
# bk.get_frame()
#
# # Ultrasound video settings
# ultrasound_output_file = "ultrasound_video.mp4"
# ultrasound_frame_width, ultrasound_frame_height = bk.img.shape[1], bk.img.shape[0]
# ultrasound_fps = 30  # Adjust the frame rate as needed
# ultrasound_video_writer = cv2.VideoWriter(ultrasound_output_file, cv2.VideoWriter_fourcc(*"mp4v"), ultrasound_fps, (ultrasound_frame_width, ultrasound_frame_height))

# Webcam video settings
webcam_output_file = "webcam_video.mp4"
webcam_frame_width, webcam_frame_height = None, None  # To be determined from the first webcam frame
webcam_fps = 30  # Adjust the frame rate as needed
webcam_video_writer = None

# Webcam capture
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change the index if you have multiple webcams connected

# Variables for synchronization
ultrasound_frame_lock = threading.Lock()
webcam_frame_lock = threading.Lock()

# Initialize webcam frame
webcam_frame = None

# Function to get color boundaries
def get_color_boundaries(color):
    # Define color boundaries in HSV color space
    if color == 'blue':
        lower_color = (0, 128, 128)
        upper_color = (120, 255, 255)
    elif color == 'red':  # this is actually red
        lower_color = (0, 128, 64)
        upper_color = (150, 255, 255)
    elif color == 'yellow':
        lower_color = (15, 100, 100)
        upper_color = (35, 255, 255)
    elif color == 'green':  # this is actually green
        lower_color = (35, 50, 50)
        upper_color = (90, 255, 255)
    return lower_color, upper_color

# Function to process the image
def process_image(image, color):
    if image is None:
        return None, None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = get_color_boundaries(color)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        x_vals = np.unique(largest_contour[:, :, 0])
        y_values = np.zeros_like(x_vals)
        for i, x_val in enumerate(x_vals):
            y_values[i] = np.max(largest_contour[largest_contour[:, :, 0] == x_val][:, 1])

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
                else:
                    y_new_smooth_fixed.append(y_new_smooth_fixed[i-1])
            y_new_smooth = np.array(y_new_smooth_fixed)

        return x_new, y_new_smooth
    else:
        return None, None

# Function to downsample the spline curve
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

# Function to capture and process ultrasound frames
def capture_ultrasound():
    global bk, ultrasound_frame, ultrasound_video_writer
    while True:
        # Capture a frame from the ultrasound video feed
        bk.get_frame()
        with ultrasound_frame_lock:
            ultrasound_frame = bk.img.copy()
        ultrasound_video_writer.write(ultrasound_frame)

# Function to capture and process webcam frames
def capture_webcam():
    global webcam_frame, webcam_video_writer
    while True:
        # Capture a frame from the webcam video feed
        _, frame = webcam.read()
        if frame is not None:
            # Create the webcam video writer if it hasn't been created yet
            if webcam_video_writer is None:
                webcam_frame_height, webcam_frame_width = frame.shape[:2]
                webcam_video_writer = cv2.VideoWriter(webcam_output_file, cv2.VideoWriter_fourcc(*"mp4v"), webcam_fps, (webcam_frame_width, webcam_frame_height))
            with webcam_frame_lock:
                webcam_frame = frame.copy()
            webcam_video_writer.write(webcam_frame)

# Function to process and display webcam frames
def process_webcam():
    global webcam_frame
    while True:
        with webcam_frame_lock:
            frame = webcam_frame
        if frame is not None:
            x_new, y_new_smooth = process_image(frame, 'green')
            if x_new is not None and y_new_smooth is not None:
                spline_curve_downsampled = downsample_spline_curve(x_new, y_new_smooth)
                processed_frame = frame.copy()
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                cv2.polylines(processed_frame, [np.int32(spline_curve_downsampled)], False, (255, 0, 0), 2)
                cv2.imshow('Webcam Frame', processed_frame)
            else:
                cv2.imshow('Webcam Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()


#
# # Start ultrasound capture thread
# ultrasound_thread = threading.Thread(target=capture_ultrasound)
# ultrasound_thread.daemon = True
# ultrasound_thread.start()


# Start webcam capture thread
webcam_thread = threading.Thread(target=capture_webcam)
webcam_thread.daemon = True
webcam_thread.start()

# Start webcam processing thread
webcam_processing_thread = threading.Thread(target=process_webcam)
webcam_processing_thread.daemon = True
webcam_processing_thread.start()

# Create named windows for display
# cv2.namedWindow('Ultrasound Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Webcam Frame', cv2.WINDOW_NORMAL)

# while True:
#     # Display ultrasound frame
#     with ultrasound_frame_lock:
#         cv2.imshow('Ultrasound Frame', ultrasound_frame)
#
#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# ultrasound_video_writer.release()
# webcam_video_writer.release()
# webcam.release()
# cv2.destroyAllWindows()

