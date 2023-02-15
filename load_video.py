import cv2
import numpy as np


cap = cv2.VideoCapture('./test_images/IMG_3623.mp4')

if (cap.isOpened() == False):
    print("Unable to read video file")

while(cap.isOpened()):
    # while open read frame by frame
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

