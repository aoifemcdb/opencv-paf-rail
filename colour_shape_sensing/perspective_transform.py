import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from skimage.filters import threshold_local
from scipy.interpolate import UnivariateSpline
from calibration import Calibration
from colour_shape_sensing.color_boundaries import ColorBoundaries

# def order_points_old(pts):
#     rect = np.zeros((4, 2), dtype="float32")
#     if len(pts) != 4:
#         raise ValueError("Contour does not have exactly four points for perspective transformation.")
#     x_coords = pts[:, 0]
#     y_coords = pts[:, 1]
#     min_x_index = np.argmin(x_coords)
#     max_x_index = np.argmax(x_coords)
#     min_y_index = np.argmin(y_coords)
#     max_y_index = np.argmax(y_coords)
#     rect[0] = pts[min_x_index]
#     rect[1] = pts[min_y_index]
#     rect[2] = pts[max_x_index]
#     rect[3] = pts[max_y_index]
#     return rect

# def four_point_transform_old(image, pts):
#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#     # return the warped image
#     return warped


# def edge_detection_nonuniform(image):
#     # load the image and compute the ratio of the old height
#     # to the new height, clone it, and resize it
#     ratio = image.shape[0] / 500.0
#     orig = image.copy()
#     image = imutils.resize(image, height=500)
#     # convert the image to grayscale, blur it, and find edges
#     # in the image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(gray, 75, 200)
#     # show the original image and the edge detected image
#     print("STEP 1: Edge Detection")
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(image)
#     plt.title("Original Image")
#     plt.subplot(122)
#     plt.imshow(edged, cmap='gray')
#     plt.title("Edged Image")
#     plt.show()
#
#     # find the contours in the edged image, keeping only the
#     # largest ones, and initialize the screen contour
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
#
#     # Find the extreme points of the contour
#     min_x = np.min(cnts[0][:, :, 0])
#     max_x = np.max(cnts[0][:, :, 0])
#     min_y = np.min(cnts[0][:, :, 1])
#     max_y = np.max(cnts[0][:, :, 1])
#
#     min_x_point = cnts[0][np.argmin(np.abs(cnts[0][:, :, 0] - min_x))]
#     max_x_point = cnts[0][np.argmin(np.abs(cnts[0][:, :, 0] - max_x))]
#     min_y_point = cnts[0][np.argmin(np.abs(cnts[0][:, :, 1] - min_y))]
#     max_y_point = cnts[0][np.argmin(np.abs(cnts[0][:, :, 1] - max_y))]
#
#     screenCnt = np.array([min_x_point, max_x_point, min_y_point, max_y_point], dtype=np.float32)
#
#     # Visualize the contours and extreme points
#     plt.figure()
#     plt.imshow(image)
#     plt.title("Contours with Extreme Points")
#     for c in cnts:
#         plt.plot(c[:, 0, 0], c[:, 0, 1], 'r', linewidth=2)
#     plt.scatter(screenCnt[:, 0, 0], screenCnt[:, 0, 1], c='blue', s=50)
#     plt.show()
#
#     return orig, ratio, screenCnt
# def order_points(pts):
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#
#     # # Rotate the point labels anticlockwise by 90 degrees
#     # rect = rect[[1, 2, 3, 0]]
#
#     return rect

def order_points(pts):
    # Find the two points with the highest y value
    highest_y_points = pts[np.argsort(pts[:, 1])[2:]]

    # Sort the two points based on x value
    sorted_highest_y_points = highest_y_points[np.argsort(highest_y_points[:, 0])]

    # Assign the points to bl and br
    bl, br = sorted_highest_y_points

    # Find the remaining points with lower y values
    remaining_points = pts[np.argsort(pts[:, 1])[:2]]

    # Find the point with the lowest x value among the remaining points (tl)
    tl = remaining_points[np.argmin(remaining_points[:, 0])]

    # Find the point with the highest x value among the remaining points (tr)
    tr = remaining_points[np.argmax(remaining_points[:, 0])]

    return np.array([bl, br, tl, tr], dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    (bl, br, tl, tr) = rect
    print("Co-ords fed to 4 point transform:")
    print("bl:", bl)
    print("br:", br)
    print("tl:", tl)
    print("tr:", tr)

    # print(tl, tr, bl, br)
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped, bl, br, tl, tr

def edge_detection(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    plt.figure()
    plt.imshow(image)
    plt.imshow(edged)
    plt.show()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    plt.figure()
    plt.imshow(image)
    plt.show()
    return orig, ratio, screenCnt

#
# filepath = './experiment_images_210623/patch/patch_2.jpg'
# image= cv2.imread(filepath)
# orig, ratio, screenCnt = edge_detection(image)
# # ordered_pts = order_points(screenCnt)
#
# warped = four_point_transform(orig, screenCnt.reshape(4,2)*ratio)
# # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# # warped = (warped > T).astype("uint8") * 255
#
# print("STEP 3: Apply perspective transform")
# plt.figure()
# plt.imshow(orig)
# plt.imshow(warped)
# plt.show()





