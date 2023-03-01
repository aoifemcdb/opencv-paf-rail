import numpy as np
import matplotlib.pyplot as plt
# from skimage.io import imshow, imread
# from skimage.color import rgb2hsv, hsv2rgb
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

image = cv2.imread("./test_images/thresholded_image_samples.jpg")

# convert mask area to white for contouring
green_area = np.where(
    (image[:,:,0] != 0) &
    (image[:,:,1] != 0) &
    (image[:,:,2] != 0)
)

image[green_area] = [255, 255, 255]
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 30, 200)
plt.figure()
plt.imshow(image)
plt.show()

# find contours of white area
contours, hierarchy = cv2. findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# filter out shadows etc
# areas = [cv2.contourArea(c) for c in contours]
# # max_index = np.argmax(areas)
# # cnt = contours[max_index]

#contour areas
h_list = []
w_list = []
x_list = []
y_list = []

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    x_list.append(x)
    y_list.append(y)
    if w >= 50 and h >= 40:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 15)
        h_list.append(h)
        w_list.append(w)
        pass

x = np.array(x_list)
y = np.array(y_list)
x_y = np.vstack((x,y))
x_y = x_y.T
# print(x_y)
# ind = np.unravel_index(np.argmin(x_y, axis=None), x_y.shape)
# print(x_y[ind])
min = np.min(x_y)
print(min)

h = np.array(h_list)
w = np.array(w_list)
h_w = np.vstack((h,w))
h_w = h_w.T
# print(h_w)

plt.figure()
plt.imshow(image)
plt.show()

# FROM SOLIDWORKS
CALIBRATION_MATRIX = 0.023
h_w_mm = np.multiply(CALIBRATION_MATRIX, h_w)

cv2.imwrite('./output_images/contoured_image_samples.jpg', image)
print("Image saved")
