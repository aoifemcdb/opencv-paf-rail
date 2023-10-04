import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set the font family to match LaTeX font
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 15

def detect_and_compute_features(image, method, nfeatures=None, color_weighting_mask=None):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the color weighting mask (if provided) to the grayscale image
    if color_weighting_mask is not None:
        gray_image = cv2.bitwise_and(gray_image, color_weighting_mask)

    if method == 'Harris':
        # Initialize Harris corner detector
        detector = cv2.cornerHarris(gray_image, 2, 3, 0.04)
        # Perform non-maximum suppression to get keypoints
        detector = cv2.dilate(detector, None)
        keypoints = np.argwhere(detector > 0.01 * detector.max())
        keypoints = [cv2.KeyPoint(x[1], x[0], 3) for x in keypoints]

        # Compute descriptors (Harris corner does not provide descriptors, so we set them to None)
        descriptors = None

    elif method == 'Fast':
        # Initialize FAST detector
        fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        keypoints = fast.detect(gray_image, None)

        # Compute descriptors (FAST does not provide descriptors, so we set them to None)
        descriptors = None

    elif method == 'Orb':
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures)
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    elif method == 'Sift':
        # Initialize SIFT detector
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    elif method == 'Surf':
        # Initialize SURF detector
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4, nOctaveLayers=3)
        keypoints, descriptors = surf.detectAndCompute(gray_image, None)
    elif method == 'Shi Tomasi':
        keypoints = cv2.goodFeaturesToTrack(gray_image, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        keypoints = [cv2.KeyPoint(x[0][0], x[0][1], 3) for x in keypoints]
        descriptors = None
    else:
        raise ValueError("Invalid method. Please choose 'Harris', 'Fast', 'Orb', 'Sift', 'Surf', or 'Shi Tomasi'.")
    return keypoints, descriptors

# image1 = cv2.imread('./experiment_images_180923/camera_angle_0/WIN_20230918_11_45_12_Pro.jpg')
# image2 = cv2.imread('./experiment_images_180923/camera_angle_20/WIN_20230918_11_47_06_Pro.jpg')
# feature_detection_methods = ['Harris', 'Fast', 'Orb', 'Sift', 'Surf', 'Shi Tomasi']
# num_features = 1000
#
# #
# # fig, plots = plt.subplots(2, 3, figsize=(18, 10))
# #
# # keypoints_counts = []
# #
# # for i, method in enumerate(feature_detection_methods):
# #     keypoints1, _ = detect_and_compute_features(image1, method, num_features)
# #     keypoints2, _ = detect_and_compute_features(image2, method, num_features)
# #     keypoints_with_size1 = np.copy(image1)
# #     keypoints_with_size2 = np.copy(image2)
# #     cv2.drawKeypoints(image1, keypoints1, keypoints_with_size1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
# #                           color=(0, 255, 0))
# #     cv2.drawKeypoints(image2, keypoints2, keypoints_with_size2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
# #                           color=(0, 255, 0))
# #     plots[i // 3, i % 3].set_title(f"{method.capitalize()} Keypoints")
# #     plots[i // 3, i % 3].imshow(cv2.cvtColor(keypoints_with_size1, cv2.COLOR_BGR2RGB))
# #     plots[i // 3, i % 3].axis('off')
# #
# #     keypoints_counts.append(len(keypoints1))
# #     # Print the number of keypoints detected in the image
# #     print(f"Number of {method.capitalize()} Keypoints: {len(keypoints1)}")
# #
# #
# # plt.tight_layout()
# # # plt.show()
# #
# # # Separate Figure for Bar Chart
# # plt.figure(figsize=(10, 6))
# # plt.bar(feature_detection_methods, keypoints_counts, color='teal')
# # plt.title('Number of Keypoints for each Detection Method')
# # plt.xlabel('Detection Method')
# # plt.ylabel('Number of Keypoints')
# # # Add num_features text box to the bar chart
# # plt.gca().text(0.95, 0.95, f'num_features = {num_features}', fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), transform=plt.gca().transAxes)
# #
# # # plt.show()














