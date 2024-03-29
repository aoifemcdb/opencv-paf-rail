import numpy as np
import matplotlib.pyplot as plt
# from skimage.io import imshow, imread
# from skimage.color import rgb2hsv, hsv2rgb
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

img = cv2.imread('50mm_yellow_iter1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# <<<<<<< Updated upstream:thresholding.py
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()
# =======
def load_hsv_image(filename):
    img = cv2.imread(filename)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2HSV)
    return img, hsv_img
# >>>>>>> Stashed changes:Archive/thresholding.py

hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(hsv_img)
plt.title('hsv image')
h, s, v = cv2.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

# <<<<<<< Updated upstream:thresholding.py
# R G B
# colour_1 = (255, 255, 255) #white
# colour_2 = (0, 255, 0) # green
# colour_3 = (255, 0, 0) # red
# mask_1 = cv2.inRange(img, colour_2, colour_1)
# mask_2 = cv2.inRange(img, colour_3, colour_1)
# result_1 = cv2.bitwise_and(img, img, mask=mask_1)
# result_2 = cv2.bitwise_and(img, img, mask=mask_2)
# plt.subplot(2, 2, 1)
# plt.imshow(mask_1, cmap="gray")
# plt.subplot(2, 2, 2)
# plt.imshow(mask_2, cmap="gray")
# plt.subplot(2, 2, 3)
# plt.imshow(result_1)
# plt.subplot(2, 2, 4)
# plt.imshow(result_2)
# plt.show()
# =======
def threshold_red(image):
    lower1 = (0, 100, 20)
    upper1 = (10, 255, 255)
    lower2 = (160, 100, 20)
    upper2 = (179, 255, 255)
    img, hsv_img = load_hsv_image(image)
    # img, hsv_img = load_hsv_image('./test_images/print_samples.jpg')
    red_mask, result = generate_mask(hsv_img, lower1, upper1, lower2, upper2)
    # visualise_thresholding(img, hsv_img, red_mask, result)
    return result

def save_thresholded_image(image, savename: str):
    result = threshold_red(image)
    cv2.imwrite('./output_images/' + '{}'.format(savename), result)
    print("Image saved")
    return

result = threshold_red('./input_images/IMG_4141.jpg')
plt.imshow(result)
plt.show()
# save_thresholded_image('./input_images/print_samples_longer.jpg','thresholded_print_samples_longer.jpg')
# >>>>>>> Stashed changes:Archive/thresholding.py

#########################################
# # # HSV # # #
# lower boundary RED colour values
lower1 = (0, 100, 20)
upper1 = (10,255, 255)

# upper boundary RED colour range values
lower2 = (160, 100, 20)
upper2 = (179, 255, 255)

lower_mask = cv2.inRange(hsv_img, lower1, upper1)
upper_mask = cv2.inRange(hsv_img, lower2, upper2)

full_mask = lower_mask + upper_mask

result = cv2.bitwise_and(hsv_img, hsv_img, mask=full_mask)

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(hsv_img)
plt.subplot(2, 2, 3)
plt.imshow(full_mask)
plt.subplot(2, 2, 4)
plt.imshow(result)
plt.show()

#save
cv2.imwrite('./test_images/thresholded_image_samples.jpg', result)
print("Image saved")





