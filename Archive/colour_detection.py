import numpy as np
import argparse
import cv2

# construct argument parse and arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to image")
args = vars(ap.parse_args())

# load image
image = cv2.imread(args["image"])
# image = cv2.imread('pinkrail.jpg')
# cv2.imshow('image', image)

# define colour boundaries in BGR colour space
boundaries = [
    ([38, 12, 5], [154, 119, 117])
    # ([86, 31, 4], [220, 88, 50]),
	# ([25, 146, 190], [62, 174, 250]),
	# ([103, 86, 65], [145, 133, 128])
]
# loop over boundaries
for (lower, upper) in boundaries:
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    #find colours within boundaries and apply mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    #show images
    scale = 0.2
    width = int(output.shape[1]*scale)
    height = int(output.shape[0]*scale)
    dim = (width, height)
    resized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("images", resized)

    cv2.waitKey(0)


