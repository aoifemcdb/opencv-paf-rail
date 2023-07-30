import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

import numpy as np
import cv2
import glob

def calibrate_camera(images_path, checkerboard_size):
    # Prepare object points
    objp = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    obj_points = []  # 3D points in real-world space
    img_points = []  # 2D points in image plane

    # Load calibration images
    calibration_images = glob.glob(images_path)

    for img_file in calibration_images:
        # Read the calibration image
        img = cv2.imread(img_file)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # If corners are found, add object points and image points
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Calibrate the camera using cv2.calibrateCamera()
    if len(img_points) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        # Return the camera matrix and distortion coefficients
        return camera_matrix, dist_coeffs

    else:
        print("No valid calibration images found.")
        return None, None




# camera_matrix, dist_coeffs = calibrate_camera('./colour_shape_sensing/checkerboard_calibration_images/*.jpg', (5,7))
# print(camera_matrix, dist_coeffs)
