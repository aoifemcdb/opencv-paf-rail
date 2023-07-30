import numpy as np
import matplotlib.pyplot as plt
from curvature_methods import *
from approximate_spline import calibrate_image, get_calibration_matrix

def generate_arc_points(radius, arc_length):
    # Compute the angle that subtends the arc of length arc_length when laid flat
    angle = arc_length / radius

    # Set up an array of angles evenly spaced between the midpoint angle and the endpoints
    angles = np.linspace(-angle/2, angle/2, 50)

    # Compute the x and y coordinates of the points on the arc
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    # Stack the x and y coordinates into a 2 x 50 array
    points = np.vstack((y_coords, x_coords)).T

    # Plot the points on a graph
    fig, ax = plt.subplots()
    ax.plot(points[:,0], points[:,1], '-')
    ax.set_aspect('equal')
    ax.set_title(f"Circle arc with radius {radius} and length {arc_length}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # plt.show()

    return points


calibration_image = './test_images/CAD_model/red_2/cropped_red_calibration.jpg'
test_image = './test_images/CAD_model/red_2/cropped_red_50mm.jpg'
real_width = 80 #mm
real_length = 8 #mm
pixels_per_mm_x, pixels_per_mm_y = get_calibration_matrix(calibration_image, real_width, real_length, 'red')
height_mm, width_mm, resized_image = calibrate_image(test_image, pixels_per_mm_x, pixels_per_mm_y)
print(height_mm, width_mm)
