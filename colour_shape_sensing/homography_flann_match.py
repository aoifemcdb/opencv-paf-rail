import cv2
import numpy as np
from feature_detection import detect_and_compute_features

# Just consider Sift and Surf for FLANN feature matching, use BF with ORB

methods = ['Sift', 'Surf']

flann_algorithm_params = {
    'Orb': {'algorithm': 6, 'table_number': 6, 'key_size': 12, 'multi_probe_level': 1},
    'Sift': {'algorithm': 0, 'trees': 5},
    'Surf': {'algorithm': 0, 'trees': 5}
}


def flann_matcher(features1, features2, ratio, flann_params):
    flann = cv2.FlannBasedMatcher(flann_params, {})

    if len(features1) < 2 or len(features2) < 2:
        return []  # Not enough features to match

    features1 = np.float32(features1)
    features2 = np.float32(features2)

    matches = flann.knnMatch(features1, features2, k=2)

    good_matches = []
    for match in matches:
        if len(match) < 2:  # If less than 2 matches are returned
            continue
        m, n = match
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def run_test(image1, image2, flann_params, method):
    kp1, des1 = detect_and_compute_features(image1, method, nfeatures=100)
    kp2, des2 = detect_and_compute_features(image2, method, nfeatures=100)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print(f"No descriptors found for {method}!")
        return float('inf')  # Return infinity if no descriptors are found

    good_matches = flann_matcher(des1, des2, ratio=0.7, flann_params=flann_params)

    if len(good_matches) < 4:
        return float('inf')  # Return infinity if not enough good matches are found

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is not None:
        matchesMask = mask.ravel().tolist()
        correct_matches = np.count_nonzero(matchesMask)
        total_matches = len(matchesMask)
        mean_distance = np.mean([cv2.norm(src_pts[i], dst_pts[i]) for i in range(total_matches) if matchesMask[i]])
        return mean_distance
    else:
        return float('inf')  # Return infinity if homography matrix can't be computed


def main(image1, image2):
    results = []
    for method in methods:
        flann_params = flann_algorithm_params.get(method, {})
        mean_distance = run_test(image1, image2, flann_params, method)

        if mean_distance is not None:
            results.append({'method': method, 'params': flann_params, 'mean_distance': mean_distance})

    results.sort(key=lambda x: x['mean_distance'])
    for res in results:
        print(res)


if __name__ == "__main__":
    image1 = cv2.imread('./experiment_images_180923/pilot_test/train_frame.jpg')
    image2 = cv2.imread('./experiment_images_180923/pilot_test/test_frame.jpg')
    main(image1, image2)
