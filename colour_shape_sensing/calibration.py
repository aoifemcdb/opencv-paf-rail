import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import glob
from colour_shape_sensing.color_boundaries import ColorBoundaries


class Calibration:
    def read_image(self, filepath):
        image = cv2.imread(filepath)
        return image

    # def get_calibration_matrix(self, calibration_image, real_width, real_length, color, plot_images=False, print_values=False):
    #     hsv = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2HSV)
    #     color_boundaries = ColorBoundaries()
    #     lower_color, upper_color = color_boundaries.get_color_boundaries(color)
    #     mask = cv2.inRange(hsv, lower_color, upper_color)
    #
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(largest_contour)
    #
    #     if plot_images:
    #         fig, ax = plt.subplots()
    #         ax.plot(largest_contour[:, 0, 0], largest_contour[:, 0, 1], 'b-')
    #
    #         rectangle = patches.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
    #         ax.add_patch(rectangle)
    #
    #         ax.set_xlim(0, calibration_image.shape[1])
    #         ax.set_ylim(calibration_image.shape[0], 0)
    #
    #         plt.show()
    #
    #     if print_values:
    #         print(h)
    #         print(w)
    #
    #     pixels_per_mm_x = w / real_width
    #     pixels_per_mm_y = h / real_length
    #
    #     return pixels_per_mm_x, pixels_per_mm_y
    #
    # def calibrate_image(self, test_image, pixels_per_mm_x, pixels_per_mm_y):
    #     height_mm = int(test_image.shape[0] / pixels_per_mm_y)
    #     width_mm = int(test_image.shape[0] / pixels_per_mm_x)
    #     resized_image = cv2.resize(test_image, (width_mm, height_mm))
    #     return resized_image
    #
    # def calibrate_image_2(self, test_image, pixels_per_mm_x, pixels_per_mm_y):
    #     plt.figure()
    #     plt.imshow(test_image)
    #     plt.title('Image fed to calibration')
    #     plt.show()
    #     hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    #     hsv_image[:, :, 0] /= pixels_per_mm_x
    #     hsv_image[:, :, 1] /= pixels_per_mm_y
    #     return hsv_image
    def calibrate_camera(self, images_path, checkerboard_size):
        objp = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

        obj_points = []  # List to store object points for each calibration image
        img_points = []  # List to store image points for each calibration image

        calibration_images = glob.glob(images_path)

        for img_file in calibration_images:
            img = cv2.imread(img_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            if ret:
                obj_points.append(objp)
                img_points.append(corners)

        if len(img_points) > 0:
            ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(obj_points, img_points,
                                                                        gray.shape[::-1], None, None)

            return camera_matrix, dist_coeffs
        else:
            print("No valid calibration images found.")
            return None, None

    def calibrate_extrinsic(self, images_path, aruco_dict, aruco_size, camera_matrix, dist_coeffs):
        obj_points = []  # List to store object points for each calibration image
        img_points_list = []  # List to store image points for each calibration image
        rvecs_list = []  # List to store rotation vectors
        tvecs_list = []  # List to store translation vectors

        calibration_images = glob.glob(images_path)

        for img_file in calibration_images:
            img = cv2.imread(img_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

            if ids is not None and len(ids) >= 2:  # Make sure at least 2 markers are detected
                img_points = []  # List to store image points for this calibration image

                for i in range(2):  # Assuming you have two markers in each image
                    # Define the 3D coordinates of the ArUco marker corners in the real world (assuming Z=0)
                    obj_points.append(np.array([[-aruco_size / 2, -aruco_size / 2, 0],
                                                [aruco_size / 2, -aruco_size / 2, 0],
                                                [aruco_size / 2, aruco_size / 2, 0],
                                                [-aruco_size / 2, aruco_size / 2, 0]], dtype=np.float32))

                    # Get the corners of the i-th detected marker
                    marker_corners = corners[i][0]

                    # Append the detected marker corners to the image points list
                    img_points.append(marker_corners)

                # Estimate the pose for the detected markers
                _, rvecs, tvecs = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

                rvecs_list.append(rvecs)
                tvecs_list.append(tvecs)

                img_points_list.append(img_points)

        if len(img_points_list) > 0:
            return rvecs_list, tvecs_list
        else:
            print("No valid ArUco markers found.")
            return None, None

    def calibrate_stereo_camera(self, images_left, images_right, checkerboard_size):
            objp = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

            obj_points = []
            img_points_left = []
            img_points_right = []

            for img_left, img_right in zip(images_left, images_right):
                gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
                ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
                ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_size, None)
                if ret_left and ret_right:
                    obj_points.append(objp)
                    img_points_left.append(corners_left)
                    img_points_right.append(corners_right)

            if len(img_points_left) > 0 and len(img_points_right) > 0:
                ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = \
                    cv2.stereoCalibrate(obj_points, img_points_left, img_points_right,
                                        cameraMatrix1=None, distCoeffs1=None,
                                        cameraMatrix2=None, distCoeffs2=None,
                                        imageSize=gray_left.shape[::-1],
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                                        flags=cv2.CALIB_FIX_INTRINSIC)
                return camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T
            else:
                print("No valid calibration images found for both cameras.")
                return None, None, None, None


    def stereo_transform_point(self,point_left, camera_matrix_left, dist_coeffs_left,
                    camera_matrix_right, dist_coeffs_right, R, T):
        # Undistort the point coordinates in the left image
        point_left_undistorted = cv2.undistortPoints(np.array([point_left]),
                                                 camera_matrix_left, dist_coeffs_left)

     # Apply stereo rectification to the undistorted point in the left image
        point_left_rectified, _ = cv2.correctMatches(R, T, None, point_left_undistorted, None, None)

        # Project the rectified point from the left image to the right image
        point_right_projected, _ = cv2.projectPoints(point_left_rectified, np.zeros((3,)), np.zeros((3,)),
                                                 camera_matrix_right, dist_coeffs_right)

        # Extract the transformed point coordinates in the right image
        point_right_transformed = point_right_projected.squeeze()

        return point_right_transformed