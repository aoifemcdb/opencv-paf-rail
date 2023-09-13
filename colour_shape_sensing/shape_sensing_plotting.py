""" This is for visualising the shape sensing and conversion to trajectory points """

import cv2
import os
import csv
from color_boundaries import ColorBoundaries
from curvature_methods import CurvatureCalculator
from colour_shape_sensing.calibration import Calibration
from colour_shape_sensing.shape_sensing import ShapeSensing
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set the font family to match LaTeX font
mpl.rcParams['font.family'] = 'serif'

#blue #235789
#red #C1292E

def main():
    calibration_images_path = './experiment_images_260723/calibration/*.jpg'
    # patch_file = './experiment_images_210623/patch/patch_1.jpg'
    folder_path = './experiment_images_260623/kidney_phantom/rails_plotting'
    output_file = 'trajectory_test.csv'

    calibration = Calibration()
    real_width = 24.5  # mm
    real_length = 24.5  # mmfor x, y in combined_coordinates:
    color = 'green'
    checkerboard_size = (5, 7)
    camera_matrix, dist_coeffs = calibration.calibrate_camera(calibration_images_path, checkerboard_size)
    shape_sensing = ShapeSensing()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Radius'])

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                csv_writer = csv.writer(csvfile)
                file_path = os.path.join(folder_path, filename)
                image = cv2.imread(file_path)
                color_boundaries = ColorBoundaries()
                lower_color, upper_color = color_boundaries.get_color_boundaries(color)
                undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
                # warped_image = shape_sensing.apply_perspective_transform(undistorted_image, lower_color, upper_color, plot_images=True)
                # patch_calibrated = calibration.calibrate_image(undistorted_image, pixels_per_mm_x, pixels_per_mm_y)
                result = shape_sensing.process_image(undistorted_image, lower_color, upper_color, plot_images=True)
                curve_downsampled = shape_sensing.downsample_spline_curve(result[2], result[3], 100, plot_curve=True)

                csv_writer.writerow(['x', 'y'])
                for x, y in curve_downsampled:
                    csv_writer.writerow([x, y])

    return curve_downsampled




    # Write each coordinate pair to the CSV file

curve_downsampled = main()
curve_downsampled[:, 0] /= 10  # scale down x-coordinates
curve_downsampled[:, 1] /= 10  # scale down y-coordinates
# Create the plot
plt.figure(figsize=(6.4, 4.0))

# Plot dashed line segments and red circles
for i, (x, y) in enumerate(curve_downsampled):
    if i % 10 == 0:
        # Every 10 points, plot a larger red circle
        plt.scatter(x, y, facecolors='None', edgecolors='#C1292E', marker='o', s=50, label='Waypoints')
    else:
        # Plot dashed line segments between points
        if i > 0:
            prev_x, prev_y = curve_downsampled[i - 1]
            plt.plot([prev_x, x], [prev_y, y], linestyle='dashed', color='#235789', label='Trajectory')

# Show the legend with only two entries (filtering out duplicates)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
# Set axes limits
x_min = 0  # Define the minimum value for the x-axis
x_max = 200  # Define the maximum value for the x-axis
y_min = 0  # Define the minimum value for the y-axis
y_max = 100  # Define the maximum value for the y-axis
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Trajectory from Sensed Shape')

# Show the plot

dpi=300
plt.savefig('trajectory_plot.png', dpi=dpi)
plt.show()
# if __name__ == '__main__':
#     main()