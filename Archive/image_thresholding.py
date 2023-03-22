import numpy as np
import matplotlib.pyplot as plt
# from skimage.io import imshow, imread
# from skimage.color import rgb2hsv, hsv2rgb
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

def load_image(filename: str):
    img = cv2.imread(filename)
    return img

def split_image_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img)
    return r, g, b

# def visualise_split_image(r,g,b):
#     fig = plt.figure()
#     axis = fig.add_subplot(1, 1, 1, projection="3d")
#     pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
#     norm = colors.Normalize(vmin=-1., vmax=1.)
#     norm.autoscale(pixel_colors)
#     pixel_colors = norm(pixel_colors).tolist()
#     axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
#     axis.set_xlabel("Red")
#     axis.set_ylabel("Green")
#     axis.set_zlabel("Blue")
#     plt.show()

def generate_mask(img, colour_1: np.array, colour_2: np.array):
    mask = cv2.inRange(img, colour_2, colour_1)
    return mask

def threshold_green(img):
    colour_1 = (255, 255, 255) #white
    colour_2 = (0, 245, 0) # green
    mask = generate_mask(img, colour_1, colour_2)
    return mask

# def visualise_thresholding(img, mask):
#     result = cv2.bitwise_and(img, img, mask=mask)
#     # plt.imshow(result)
#     return

def visualise_process(img):
    mask = threshold_green(img)
    result = cv2.bitwise_and(img, img, mask=mask)

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.title.set_text('Original Image')
    plt.imshow(img)

    ax2 = fig.add_subplot(222)
    ax2.title.set_text('Mask')
    plt.imshow(mask, cmap = "gray")
    ax3 = fig.add_subplot(223)
    ax3.title.set_text('Thresholded Image')
    plt.imshow(result)

    plt.show()
    return



img = load_image('../colour_shape_sensing/test_images/print_samples.jpg')
visualise_process(img)






