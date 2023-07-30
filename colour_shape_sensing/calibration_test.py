from calibration import Calibration
import glob
import cv2


# Create an instance of the Calibration class
calibration = Calibration()

# Provide the path to the calibration images and the checkerboard size
images_path = './experiment_images_260723/calibration/*.jpg'
checkerboard_size = (5, 7)

camera_matrix, dist_coeffs = calibration.calibrate_camera(images_path, checkerboard_size)

# Extrinsic calibration using ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_size = 0.008  # Size of the ArUco marker in meters (assuming square markers)

images_path = './experiment_images_260723/30mm/train/*.jpg'
rvecs, tvecs = calibration.calibrate_extrinsic(images_path, aruco_dict, aruco_size, camera_matrix, dist_coeffs)

# Print the intrinsic and extrinsic parameters
print("Camera matrix:")
print(camera_matrix)

print("Distortion coefficients:")
print(dist_coeffs)

for i, (R, tvec) in enumerate(zip(rvecs, tvecs)):
    print(f"Extrinsic parameters for image {i+1}:")
    print("Rotation vector:")
    print(R)
    print("Translation vector:")
    print(tvec)




