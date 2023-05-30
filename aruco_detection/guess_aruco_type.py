import argparse
import imutils
import cv2
from Archive.aruco_dict import ARUCO_DICT

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing ArUCo tag")
args = vars(ap.parse_args())

# load image and resize
print('[INFO] loading image...')
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
cv2.imshow('image', image)
cv2.waitKey(0)

# loop over type of aruco dictionaries
for (arucoName, arucoDict) in ARUCO_DICT.items():
	arucoDict = cv2.aruco.Dictionary_get(arucoDict)
	arucoParams = cv2.aruco.DetectorParameters_create()
	(corners, ids, rejected) = cv2.aruco.detectMarkers(
		image, arucoDict, parameters=arucoParams)
	if len(corners) > 0:
		print("[INFO] detected {} markers for '{}'".format(len(corners), arucoName))
