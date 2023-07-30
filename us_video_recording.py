import cv2
from sksurgerybk.interface.bk5000 import BK5000

# Setup and connect to BK

timeout = 5
frames_per_second = 25

ip = '128.16.0.3' # Default IP of BK5000
port = 7915

bk = BK5000(timeout=timeout, frames_per_second=frames_per_second)
bk.connect_to_host(ip, port)
bk.query_win_size()
bk.start_streaming()


# Ultrasound video settings
ultrasound_output_file = "ultrasound_video.mp4"
ultrasound_frame_width, ultrasound_frame_height = bk.img.shape[1], bk.img.shape[0]
ultrasound_fps = 30  # Adjust the frame rate as needed
ultrasound_video_writer = cv2.VideoWriter(ultrasound_output_file, cv2.VideoWriter_fourcc(*"mp4v"), ultrasound_fps, (ultrasound_frame_width, ultrasound_frame_height))

# Webcam video settings
webcam_output_file = "webcam_video.mp4"
webcam_frame_width, webcam_frame_height = None, None  # To be determined from the first webcam frame
webcam_fps = 30  # Adjust the frame rate as needed
webcam_video_writer = None

# Webcam capture
webcam = cv2.VideoCapture(0)  # Change the index if you have multiple webcams connected

# Get the first frame to determine the webcam frame dimensions
_, webcam_frame = webcam.read()
if webcam_frame is not None:
    webcam_frame_height, webcam_frame_width = webcam_frame.shape[:2]
else:
    raise ValueError("Failed to capture the first frame from the webcam.")

# Loop to capture and save frames
n_frames = 1000

for i in range(n_frames):
    # Ultrasound frame capture
    bk.get_frame()
    cv2.imshow('bk_frame', bk.img)

    # Write the ultrasound frame to the video file
    ultrasound_video_writer.write(bk.img)

    # Webcam frame capture
    _, webcam_frame = webcam.read()
    if webcam_frame is not None:
        cv2.imshow('webcam_frame', webcam_frame)

        # Create the webcam video writer if it hasn't been created yet
        if webcam_video_writer is None:
            webcam_video_writer = cv2.VideoWriter(webcam_output_file, cv2.VideoWriter_fourcc(*"mp4v"), webcam_fps, (webcam_frame_width, webcam_frame_height))

        # Write the webcam frame to the video file
        webcam_video_writer.write(webcam_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
ultrasound_video_writer.release()
webcam_video_writer.release()
webcam.release()
cv2.destroyAllWindows()
