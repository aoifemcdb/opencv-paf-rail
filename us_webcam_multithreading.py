import cv2
import threading
import numpy as np
from scipy.interpolate import UnivariateSpline
from sksurgerybk.interface.bk5000 import BK5000

# Function to get color boundaries
def get_color_boundaries(color):
    # Define color boundaries in HSV color space
    if color == 'blue':
        lower_color = (0, 128, 128)
        upper_color = (120, 255, 255)
    elif color == 'red':
        lower_color = (0, 128, 64)
        upper_color = (150, 255, 255)
    elif color == 'yellow':
        lower_color = (15, 100, 100)
        upper_color = (35, 255, 255)
    elif color == 'green':
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

def capture_ultrasound(ultrasound_frame_lock, ultrasound_video_writer):
    timeout = 5
    frames_per_second = 25
    ip = '128.16.0.3'  # Default IP of BK5000
    port = 7915
    bk = BK5000(timeout=timeout, frames_per_second=frames_per_second)
    bk.connect_to_host(ip, port)
    bk.query_win_size()
    bk.start_streaming()
    bk.get_frame()
    frame_width, frame_height = bk.img.shape[1], bk.img.shape[0]
    ultrasound_frame_lock.acquire()
    ultrasound_frame = bk.img.copy()
    ultrasound_frame_lock.release()
    ultrasound_video_writer.write(ultrasound_frame)
    n_frames = 1000
    for _ in range(n_frames):
        bk.get_frame()
        ultrasound_frame_lock.acquire()
        ultrasound_frame = bk.img.copy()
        ultrasound_frame_lock.release()
        ultrasound_video_writer.write(ultrasound_frame)
    ultrasound_video_writer.release()

def capture_process_webcam(webcam_frame_lock, webcam_video_writer):
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        _, frame = webcam.read()
        if frame is not None:
            webcam_frame_lock.acquire()
            webcam_frame = frame.copy()
            webcam_frame_lock.release()
            webcam_video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    webcam_video_writer.release()

if __name__ == '__main__':
    ultrasound_output_file = "ultrasound_stream_robot.mp4"
    webcam_output_file = "webcam_stream_robot.mp4"
    fps = 30

    ultrasound_frame_lock = threading.Lock()
    webcam_frame_lock = threading.Lock()

    ultrasound_video_writer = cv2.VideoWriter(ultrasound_output_file, cv2.VideoWriter_fourcc(*"avc1"), fps, (0, 0))
    webcam_video_writer = cv2.VideoWriter(webcam_output_file, cv2.VideoWriter_fourcc(*"avc1"), fps, (0, 0))

    ultrasound_thread = threading.Thread(target=capture_ultrasound, args=(ultrasound_frame_lock, ultrasound_video_writer))
    webcam_thread = threading.Thread(target=capture_process_webcam, args=(webcam_frame_lock, webcam_video_writer))

    ultrasound_thread.start()
    webcam_thread.start()

    ultrasound_thread.join()
    webcam_thread.join()

    cv2.destroyAllWindows()
