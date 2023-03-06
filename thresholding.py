import matplotlib.pyplot as plt
import cv2


def load_hsv_image(filename: str):
    img = cv2.imread(filename)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2HSV)
    return img, hsv_img

def generate_mask(hsv_img, lower1, upper1, lower2, upper2):
    lower_mask = cv2.inRange(hsv_img, lower1, upper1)
    upper_mask = cv2.inRange(hsv_img, lower2, upper2)
    full_mask = lower_mask + upper_mask
    result = cv2.bitwise_and(hsv_img, hsv_img, mask = full_mask)
    return full_mask, result

def visualise_thresholding(img, hsv_img, full_mask, result):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.imshow(hsv_img)
    plt.subplot(2, 2, 3)
    plt.imshow(full_mask)
    plt.subplot(2, 2, 4)
    plt.imshow(result)
    plt.show()

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

def save_thresholded_image(savename: str):
    result = threshold_red()
    cv2.imwrite('./output_images/' + '{}'.format(savename), result)
    print("Image saved")
    return

threshold_red('./input_images/print_samples.jpg')
# save_thresholded_image('thresholded_print_samples.jpg')







