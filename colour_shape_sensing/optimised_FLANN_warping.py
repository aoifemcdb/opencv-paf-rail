# from matching_experiments_FLANN import visualize_best_matches, warp_and_visualize
import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_detection import detect_and_compute_features  # Assuming you have this module
import os

def visualize_best_matches(image1, kp1, des1, image2, kp2, des2, best_flann_params):
    flann = cv2.FlannBasedMatcher(best_flann_params, {})
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match in matches:
        if len(match) < 2:  # If less than 2 matches are returned
            continue
        m, n = match
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Print the number of good matches
    print(f"Number of Good Matches: {len(good_matches)}")

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       flags=0)

    # Drawing and displaying the matches
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, **draw_params)
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()
    return good_matches


def warp_and_visualize(image1, kp1, image2, kp2, good_matches, num_matches):
    if len(good_matches) < num_matches:
        print(f"Not enough good matches to use {num_matches} matches.")
        return

    subset_matches = good_matches[:num_matches]  # Use the first num_matches from good_matches

    # Extract the keypoints from the good matches
    src_pts = np.float32([kp1[match.queryIdx].pt for match in subset_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[match.trainIdx].pt for match in subset_matches]).reshape(-1, 1, 2)

    # Calculate the homography matrix using RANSAC
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(homography_matrix)

    if homography_matrix is not None:
        # Warp the second image to align with the first image
        warped_image = cv2.warpPerspective(image2, homography_matrix, (image1.shape[1], image1.shape[0]))

        # Visualize original and warped images side by side
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        plt.title('Warped Image')
        plt.axis('off')
        plt.show()

        # Save the images in 'pilot_results' folder
        if not os.path.exists('pilot_results'):
            os.mkdir('pilot_results')

        cv2.imwrite(os.path.join('pilot_results', 'original_image.jpg'), image2)
        cv2.imwrite(os.path.join('pilot_results', 'warped_image.jpg'), warped_image)

def main(image1, image2):
    # Hardcode your determined best FLANN parameters here
    best_flann_params = {'algorithm': 6, 'table_number': 6, 'key_size': 18, 'multi_probe_level': 1}

    # Assuming the method corresponding to the best FLANN params is the same for both images,
    # otherwise adjust accordingly.
    best_method = 'Orb'  # Replace 'your_method_here' with the method name, for example 'Orb', 'Sift' etc.

    kp1, des1 = detect_and_compute_features(image1, best_method, nfeatures=100)
    print('kp1: ', kp1, 'des1: ', des1)
    kp2, des2 = detect_and_compute_features(image2, best_method, nfeatures=100)
    print('kp1: ',kp2,'des2: ', des2)

    good_matches = visualize_best_matches(image1, kp1, des1, image2, kp2, des2, best_flann_params)
    warp_and_visualize(image1, kp1, image2, kp2, good_matches, 6)
    plt.show()



image1 = cv2.imread('./experiment_images_180923/pilot_test/train_frame.jpg')
image2 = cv2.imread('./experiment_images_180923/pilot_test/test_frame.jpg')
main(image1,image2)
