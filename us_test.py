import cv2
from sksurgerybk.interface.bk5000 import BK5000
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

def main():
    # Setup and connect to BK
    timeout = 5
    frames_per_second = 25
    ip = '128.16.0.3' # Default IP of BK5000
    port = 7915

    bk = BK5000(timeout=timeout, frames_per_second=frames_per_second)
    bk.connect_to_host(ip, port)
    bk.query_win_size()
    bk.start_streaming()

    # Get a single frame
    bk.get_frame()

    # Display image to check it is ok
    cv2.imshow('bk_frame', bk.img)

    # Create a VideoWriter object
    output_file = "ultrasound_stream_test_human6.mp4"
    frame_width, frame_height = bk.img.shape[1], bk.img.shape[0]
    fps = 30  # Adjust the frame rate as needed
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"avc1"), fps, (frame_width, frame_height))

    # Get multiple frames
    n_frames = 1000

    for i in range(n_frames):
      bk.get_frame()
      cv2.imshow('bk_frame', bk.img)
      # Write the frame to the video file
      video_writer.write(bk.img)

      # Exit the loop if the 'q' key is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the video writer and close any open windows
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()