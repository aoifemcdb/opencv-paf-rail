import cv2
import matplotlib.pyplot as plt
import numpy as np
from feature_detection import detect_and_compute_features

# Load the images
image1 = cv2.imread('./experiment_images_180923/camera_angle_0/WIN_20230918_11_45_12_Pro.jpg')
image2 = cv2.imread('./experiment_images_180923/camera_angle_20/WIN_20230918_11_47_06_Pro.jpg')

if image1 is None:
    print("Image1 not loaded.")
else:
    print("Image1 loaded successfully.")

if image2 is None:
    print("Image2 not loaded.")
else:
    print("Image2 loaded successfully.")

# Convert the images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Threshold the images to obtain binary masks
_, binary_mask1 = cv2.threshold(gray_image1, 100, 255, cv2.THRESH_BINARY)
_, binary_mask2 = cv2.threshold(gray_image2, 100, 255, cv2.THRESH_BINARY)

# Specify the feature detection methods to test
feature_detection_methods = ['Harris', 'Fast', 'Orb', 'Sift', 'Surf', 'Shi Tomasi']

# Set the number of features (for ORB, SIFT, and SURF)
num_features = 100

# Create a figure for keypoints and warping visualization with subplots
fig_kw, plots_kw = plt.subplots(len(feature_detection_methods), 2, figsize=(12, 6*len(feature_detection_methods)))

# Create a figure for feature matching visualization with subplots
fig_matching, plots_matching = plt.subplots(len(feature_detection_methods), figsize=(12, 6*len(feature_detection_methods)))

# # Detect ArUco markers in the images
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
# parameters = cv2.aruco.DetectorParameters_create()
# corners1, ids1, rejected1 = cv2.aruco.detectMarkers(image1, aruco_dict, parameters=parameters)
# corners2, ids2, rejected2 = cv2.aruco.detectMarkers(image2, aruco_dict, parameters=parameters)
#
# # Extract corner points of the detected markers as keypoints
# aruco_keypoints1 = [cv2.KeyPoint(c[0][0], c[0][1], 10) for c in corners1]
# aruco_keypoints2 = [cv2.KeyPoint(c[0][0], c[0][1], 10) for c in corners2]

# Initialize a dictionary to store the number of good matches for each method
good_matches_count = {}


# Loop through each method and visualize keypoints, feature matching, and warped images
for i, method in enumerate(feature_detection_methods):
    # Detect and compute features using the specified method with color weighting
    keypoints_method1, descriptors1 = detect_and_compute_features(image1, method, num_features, color_weighting_mask=binary_mask1)
    keypoints_method2, descriptors2 = detect_and_compute_features(image2, method, num_features, color_weighting_mask=binary_mask2)

    # Combine keypoints from feature detection methods and ArUco markers
    keypoints1 = keypoints_method1
    keypoints2 = keypoints_method2

    # Draw keypoints with size
    keypoints_with_size1 = np.copy(image1)
    keypoints_with_size2 = np.copy(image2)
    cv2.drawKeypoints(image1, keypoints1, keypoints_with_size1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(image2, keypoints2, keypoints_with_size2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display image with keypoints and their size in the keypoints and warping visualization figure
    plots_kw[i, 0].set_title(f"{method.capitalize()} Keypoints")
    plots_kw[i, 0].imshow(cv2.cvtColor(keypoints_with_size1, cv2.COLOR_BGR2RGB))
    plots_kw[i, 0].axis('off')

    # Calculate the homography matrix between the two images using the keypoint matches

    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Perform the matching between the descriptors of the training image and the test image
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Take only the top matches (adjust this threshold if needed)
    num_top_matches = 100   # Try using a lower value here, like 4 to 10
    good_matches = matches[:num_top_matches]

    # Store the number of good matches in the dictionary
    good_matches_count[method] = len(good_matches)

    # Draw feature matching lines on the images in the feature matching visualization figure
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                    matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=0)
    plots_matching[i].set_title(f"Feature Matching for {method.capitalize()}")
    plots_matching[i].imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plots_matching[i].axis('off')

    # Check if there are enough good matches to estimate homography
    if len(good_matches) >= 4:
        # Extract the keypoints from the good matches
        src_pts = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        # Calculate the homography matrix using RANSAC
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(homography_matrix)

        # Check if homography_matrix is not None
        if homography_matrix is not None:
            # Convert the homography matrix to float32
            homography_matrix = homography_matrix.astype(np.float32)

            # Warp the test image to align with the training image
            warped_image = cv2.warpPerspective(image2, homography_matrix, (image1.shape[1], image1.shape[0]))

            # Display the warped image in the keypoints and warping visualization figure
            plots_kw[i, 1].set_title(f"Warped Image using {method.capitalize()}")
            plots_kw[i, 1].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
            plots_kw[i, 1].axis('off')

        else:
            print(f"Failed to estimate homography for method {method}. Not enough good matches.")
    else:
        print(f"Not enough good matches for method {method}.")

# Plot bar chart of good_matches_count
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
ax_bar.bar(good_matches_count.keys(), good_matches_count.values(), color=['b', 'g', 'r', 'c', 'm', 'y'])
ax_bar.set_title('Number of Good Matches for Each Feature Detector')
ax_bar.set_ylabel('Number of Good Matches')
ax_bar.set_xlabel('Feature Detector Methods')

# Adding the num_features text box to the bar chart
# fig_bar.text(0.95, 0.95, f'num_features = {num_features}', fontsize=12, verticalalignment='top',
#              horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.tight_layout()
plt.show()










