import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_and_compute_features(image, method, nfeatures=None):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'harris':
        # Initialize Harris corner detector
        detector = cv2.cornerHarris(gray_image, 2, 3, 0.04)
        # Perform non-maximum suppression to get keypoints
        detector = cv2.dilate(detector, None)
        keypoints = np.argwhere(detector > 0.01 * detector.max())
        keypoints = [cv2.KeyPoint(x[1], x[0], 3) for x in keypoints]

        # Compute descriptors (Harris corner does not provide descriptors, so we set them to None)
        descriptors = None

    elif method == 'fast':
        # Initialize FAST detector
        fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        keypoints = fast.detect(gray_image, None)

        # Compute descriptors (FAST does not provide descriptors, so we set them to None)
        descriptors = None

    elif method == 'orb':
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures)
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    elif method == 'sift':
        # Initialize SIFT detector
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    elif method == 'surf':
        # Initialize SURF detector
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4, nOctaveLayers=3)
        keypoints, descriptors = surf.detectAndCompute(gray_image, None)

    else:
        raise ValueError("Invalid method. Please choose 'harris', 'fast', 'orb', 'sift', or 'surf'.")

    return keypoints, descriptors


def detect_and_compute_features(image, method, num_features, color_weighting_mask=None):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the color weighting mask (if provided) to the grayscale image
    if color_weighting_mask is not None:
        gray_image = cv2.bitwise_and(gray_image, color_weighting_mask)

    # Detect and compute features based on the specified method
    if method == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        detector = cv2.xfeatures2d.SURF_create()
        detector.setHessianThreshold(num_features)
    elif method == 'orb':
        detector = cv2.ORB_create(nfeatures=num_features)
    else:
        raise ValueError("Invalid feature detection method. Use 'sift', 'surf', or 'orb'.")

    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return keypoints, descriptors


# # Load the images
# image1 = cv2.imread('./experiment_images_040723/parallel/angle_10/30mm/WIN_20230704_10_28_41_Pro.jpg')
# image2 = cv2.imread('./experiment_images_040723/parallel/angle_20/colorbands/30mm/WIN_20230704_10_06_17_Pro.jpg')
#
# # Specify the feature detection methods to test
# feature_detection_methods = ['harris', 'fast', 'orb', 'sift', 'surf']
#
# # Set the number of features (for ORB, SIFT, and SURF)
# num_features = 1000
#
# # Create a figure for subplots with 2 rows and 3 columns
# fig, plots = plt.subplots(2, 3, figsize=(18, 10))
#
# # Loop through each method and visualize keypoints
# for i, method in enumerate(feature_detection_methods):
#     # Detect and compute features using the specified method
#     keypoints1, _ = detect_and_compute_features(image1, method, num_features)
#     keypoints2, _ = detect_and_compute_features(image2, method, num_features)
#
#     # Draw keypoints with size
#     keypoints_with_size1 = np.copy(image1)
#     keypoints_with_size2 = np.copy(image2)
#     cv2.drawKeypoints(image1, keypoints1, keypoints_with_size1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0))
#     cv2.drawKeypoints(image2, keypoints2, keypoints_with_size2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0))
#
#     # Display image with keypoints and their size
#     plots[i // 3, i % 3].set_title(f"{method.capitalize()} Keypoints")
#     plots[i // 3, i % 3].imshow(cv2.cvtColor(keypoints_with_size1, cv2.COLOR_BGR2RGB))
#     plots[i // 3, i % 3].axis('off')  # Turn off axes for this subplot
#
#     # Print the number of keypoints detected in the image
#     print(f"Number of {method.capitalize()} Keypoints: {len(keypoints1)}")
#
# # Turn off axes for the third plot on the second row
# plots[1, 2].axis('off')
#
# plt.tight_layout()
# plt.show()




























