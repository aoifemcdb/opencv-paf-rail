import cv2 as cv2

flann_parameters = [
    {'algorithm': cv2.FLANN_INDEX_KDTREE, 'trees': 5, 'checks': 50},

    {'algorithm': cv2.FLANN_INDEX_KMEANS, 'branching': 32, 'iterations': 7, 'centers_init': cv2.FLANN_CENTERS_RANDOM,
     'checks': 50},

    {'algorithm': cv2.FLANN_INDEX_COMPOSITE, 'trees': 5, 'branching': 32, 'iterations': 7, 'checks': 50},

    {'algorithm': cv2.FLANN_INDEX_LINEAR},

    {'algorithm': cv2.FLANN_INDEX_HIERARCHICAL, 'branching': 32, 'centers_init': cv2.FLANN_CENTERS_RANDOM, 'trees': 5,
     'leaf_size': 30, 'checks': 50},

    {'algorithm': cv2.FLANN_INDEX_LSH, 'table_number': 6, 'key_size': 12, 'multi_probe_level': 1, 'checks': 50},

    {'algorithm': cv2.FLANN_INDEX_AUTOTUNED, 'target_precision': 0.9, 'build_weight': 0.
