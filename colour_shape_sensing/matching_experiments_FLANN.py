import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_detection import detect_and_compute_features  # Assuming you have this module
import os


def flann_matcher(features1, features2, ratio, flann_params):
    flann = cv2.FlannBasedMatcher(flann_params, {})
    if len(features1) < 2 or len(features2) < 2:
        return []  # Not enough features to match
    matches = flann.knnMatch(features1, features2, k=2)
    good_matches = []
    for match in matches:
        if len(match) < 2:
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
        return float('inf')

    good_matches = flann_matcher(des1, des2, ratio=0.7, flann_params=flann_params)
    if len(good_matches) < 4:
        return float('inf')

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
        return float('inf')


def test_kdtree(image1, image2, methods=['Sift', 'Surf']):
    results = []
    for method in methods:
        flann_params = {'algorithm': 0, 'trees': 5}  # 0 for KDTree
        for trees in range(1, 11):  # Loop through the number of trees from 1 to 10
            flann_params['trees'] = trees
            mean_distance = run_test(image1, image2, flann_params, method)
            results.append({'method': method, 'params': flann_params.copy(), 'mean_distance': mean_distance})
    return results


def test_lsh(image1, image2, methods=['Orb']):
    results = []
    for method in methods:
        flann_params = {'algorithm': 6}  # 6 for LSH
        for table_number in range(6, 13):
            for key_size in range(10, 21):
                for multi_probe_level in range(0, 3):
                    flann_params.update({
                        'table_number': table_number,
                        'key_size': key_size,
                        'multi_probe_level': multi_probe_level
                    })
                    mean_distance = run_test(image1, image2, flann_params, method)
                    results.append({'method': method, 'params': flann_params.copy(), 'mean_distance': mean_distance})
    return results


def test_composite(image1, image2, methods=['Sift', 'Surf']):
    results = []
    for method in methods:
        flann_params = {'algorithm': cv2.FLANN_INDEX_COMPOSITE}

        # loop through the different possible values for each parameter
        for trees in range(1, 11):  # example range
            for branching in [2, 5, 10]:  # example values
                for iterations in [5, 10, 20]:  # example values
                    flann_params.update({
                        'trees': trees,
                        'branching': branching,
                        'iterations': iterations,
                    })
                    mean_distance = run_test(image1, image2, flann_params, method)
                    results.append({'method': method, 'params': flann_params.copy(), 'mean_distance': mean_distance})
    return results

def test_linear(image1, image2, methods=['Sift', 'Surf', 'Orb']):
    results = []
    for method in methods:
        flann_params = {'algorithm': cv2.FLANN_INDEX_LINEAR}  # Setting algorithm to LINEAR
        mean_distance = run_test(image1, image2, flann_params, method)
        results.append({'method': method, 'params': flann_params, 'mean_distance': mean_distance})
    return results

def test_hierarchical_clustering(image1, image2, methods=['Sift', 'Surf', 'Orb']):
    results = []
    for method in methods:
        for branching in [2, 5, 10]:  # Replace with the range of your interest
            for centers_init in ['RANDOM', 'GONZALES', 'KMEANSPP']:  # Replace with the possible options for your OpenCV version
                for trees in [1, 5, 10]:  # Replace with the range of your interest
                    for leaf_size in [10, 30, 50]:  # Replace with the range of your interest
                        flann_params = {
                            'algorithm': cv2.FLANN_INDEX_HIERARCHICAL,  # Replace with the correct constant if this is not valid
                            'branching': branching,
                            'centers_init': centers_init,
                            'trees': trees,
                            'leaf_size': leaf_size
                        }
                        mean_distance = run_test(image1, image2, flann_params, method)
                        results.append({
                            'method': method,
                            'params': flann_params,
                            'mean_distance': mean_distance
                        })
    return results


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
    results = []
    results.extend(test_kdtree(image1, image2))
    results.extend(test_lsh(image1, image2))
    # pilot_results.extend(test_composite(image1, image2))  # adding the test for Composite Index Parameters
    # pilot_results.extend(test_linear(image1, image2))
    # pilot_results.extend(test_hierarchical_clustering(image1, image2))
    # Sorting the pilot_results based on mean_distance to find the method with the smallest mean_distance
    results.sort(key=lambda x: x['mean_distance'])

    best_method = results[0]['method']
    print("The best method is: ", best_method)
    best_flann_params = results[0]['params']
    print("The best FLANN params are: ", best_flann_params)
    # best_flann_params = {'algorithm': 6, 'table_number': 12, 'key_size': 11, 'multi_probe_level': 0} # found from previous experiments and hardcoded to speedup

    kp1, des1 = detect_and_compute_features(image1, best_method, nfeatures=100)
    kp2, des2 = detect_and_compute_features(image2, best_method, nfeatures=100)


    good_matches =  visualize_best_matches(image1, kp1, des1, image2, kp2, des2, best_flann_params)
    warp_and_visualize(image1, kp1, image2, kp2, good_matches, 10)
    plt.show()
    return results


def find_min_distance_params(results):
    min_distance = float('inf')
    best_params = None
    for res in results:
        if res['mean_distance'] < min_distance:
            min_distance = res['mean_distance']
            best_params = res['params']
    return best_params, min_distance




if __name__ == "__main__":
    image1 = cv2.imread('./experiment_images_180923/pilot_test/train_frame.jpg')
    image2 = cv2.imread('./experiment_images_180923/pilot_test/test_frame.jpg')
    results = main(image1, image2)
    best_params, min_distance = find_min_distance_params(results)
    plt.show()


    print(f"The minimum mean_distance is {min_distance} with parameters {best_params}")






