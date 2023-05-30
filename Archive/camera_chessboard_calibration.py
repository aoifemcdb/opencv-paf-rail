import cv2
import numpy as np

# Define the number of chessboard corners
num_corners = (9, 6)

# Define the size of each square on the chessboard in millimeters
square_size = 20

# Set up the object points for the chessboard corners
object_points = np.zeros((num_corners[0] * num_corners[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:num_corners[0], 0:num_corners[1]].T.reshape(-1, 2) * square_size

# Define the arrays to store the object points and image points
obj_points = []
img_points = []

# Set up the camera capture object
cap = cv2.VideoCapture(0)

# Start the camera calibration loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

    # If corners are found, add the object and image points to their respective arrays
    if ret == True:
        obj_points.append(object_points)
        img_points.append(corners)

        # Draw the corners on the frame and display it
        cv2.drawChessboardCorners(frame, num_corners, corners, ret)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(500)

    # If the user presses the 'q' key, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture object and destroy the calibration window
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera using the object and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
