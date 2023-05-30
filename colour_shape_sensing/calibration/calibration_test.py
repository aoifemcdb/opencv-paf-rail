import cv2
import numpy as np
import matplotlib.pyplot as plt
from colour_shape_sensing.get_curvature import get_color_boundaries

def calibrate_image(image, real_width, real_length, color):
    # Get a mask of the pixels in the image that match the specified color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = get_color_boundaries(color)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find the contours of the pixels in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the number of pixels per mm in the x and y directions
    pixels_per_mm_x = w / real_width
    pixels_per_mm_y = h / real_length

    # Resize the image based on the pixels per mm in the x and y directions
    resized_image = cv2.resize(image, (int(w/pixels_per_mm_x), int(h/pixels_per_mm_y)), interpolation=cv2.INTER_LANCZOS4)

    # Flip the image in the y direction
    resized_image = np.flipud(resized_image)


    # Draw a grid of 0.5 mm intervals on the image
    # Draw a grid of 0.5 mm intervals on the image
    grid_size = 0.5  # mm
    grid_thickness = 1  # pixels
    grid_color = (255, 255, 255)
    for i in range(0, int(round(real_length * 10 * grid_size)), int(pixels_per_mm_y * grid_size * 10)):
        cv2.line(cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY), (0, int(i * pixels_per_mm_y)),
                 (int(resized_image.shape[1]), int(i * pixels_per_mm_y)), grid_color, grid_thickness)
    for i in range(0, int(round(real_width * 10 * grid_size)), int(pixels_per_mm_x * grid_size * 10)):
        cv2.line(cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY), (int(i * pixels_per_mm_x), 0),
                 (int(i * pixels_per_mm_x), int(resized_image.shape[0])), grid_color, grid_thickness)
    # Return the calibrated image and the pixels per mm in x and y
    return resized_image, pixels_per_mm_x, pixels_per_mm_y


# Load the input image
image = cv2.imread('./test_images/CAD_models/red/red_50mm_iter1.jpg')

# Specify the color of the pixels to measure
color = 'red' # Red

# Measure the real-world width and length of the area containing the pixels of interest
real_width = 100  # mm
real_length = 20  # mm

# Calibrate the image
calibrated_image, pixels_per_mm_x, pixels_per_mm_y = calibrate_image(image, real_width, real_length, color)

# Plot the calibrated image
plt.imshow(calibrated_image)
plt.xlabel('Length (mm)')
plt.ylabel('Width (mm)')

height, width, _ = image.shape

# Set the x and y axis limits of the plot in mm
plt.xlim(0, width / pixels_per_mm_x)
plt.ylim(0, height / pixels_per_mm_y)


# Show the plot
plt.show()

