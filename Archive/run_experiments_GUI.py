import PySimpleGUI as sg
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from approximate_spline import *
from get_curvature import *

def load_data(colour, image_path, real_width, real_length):
    # real_width = 100  # mm
    # real_length = 20  # mm
    lower_color, upper_color = get_color_boundaries(colour)
    img, mask, x_new, y_new_smooth = process_image(image_path, lower_color, upper_color)
    image, pixels_per_mm_x, pixels_per_mm_y = get_calibration_matrix(img, real_width, real_length, colour)
    #calibrate image
    x, y = calibrate_image(x_new, y_new_smooth, pixels_per_mm_x, pixels_per_mm_y)
    spline_curve_downsampled = get_spline_curve(x, y, 20)
    return spline_curve_downsampled

def get_radius_error(data, radius):
    data_curvature = calculate_curvature(data)
    data_radius = np.reciprocal(data_curvature)
    data_radius = remove_outliers_iqr(data_radius)
    data_radius = data_radius[1:-1]
    data_radius_mean, data_radius_stddev = get_radius_mean_stddev(data_radius)
    data_radius_error = data_radius_mean - radius
    return data_radius_mean, data_radius_stddev, data_radius_error

def get_radius_mean_stddev(radius):
    radius_mean = np.mean(radius)
    radius_stddev = np.std(radius)
    return radius_mean, radius_stddev

def save_to_csv(filename, column_names, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the column names if the file is empty
        if file.tell() == 0:
            writer.writerow(column_names)
        # Write the data
        writer.writerow(data)

# Define the GUI layout
layout = [
    [sg.Text('Image Path:'), sg.Input(key='image_path'), sg.FileBrowse()],
    [sg.Text('CSV File Path:'), sg.Input(key='csv_path'), sg.FileSaveAs()],
    [sg.Button('Load'), sg.Button('Save')],
    [sg.Column(layout=[
        [sg.Text('Image Preview:'), sg.Image(key='image_preview')]
    ], element_justification='c', vertical_alignment='top'),
    sg.Column(layout=[
        [sg.Canvas(key='plot_canvas')]
    ], element_justification='c', vertical_alignment='top')],
    [sg.Text('', size=(30, 1), key='output')]
]

# Create the GUI window
window = sg.Window('Experiment GUI', layout)

# Define the Matplotlib plot
fig, ax = plt.subplots()
ax.set_xlabel('X')
ax.set_ylabel('Y')
plot_canvas = FigureCanvasTkAgg(fig, master=window['plot_canvas'].TKCanvas)
plot_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)


# Initialize variables for data analysis
data_radius_mean = 0.0
data_radius_stddev = 0.0
data_radius_error = 0.0

column_names = ['Colour', 'Radius', 'Iteration', 'Data Radius Mean', 'Data Radius Error', 'Data Radius Std Dev']

# Event loop to process GUI events
iteration = 1
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == 'Load':
        # Extract radius and colour from image path
        match = re.search(r'\d+', values['image_path'])
        if match:
            radius = float(match.group(0))
        else:
            radius = 'N/A'
        match = re.search(r'red|yellow|orange', values['image_path'])
        if match:
            colour = match.group(0)
        else:
            colour = 'N/A'
        # Update the output text
        output_text = f'Radius: {radius}, Colour: {colour}, Iteration: {iteration}, Data Radius Mean: {data_radius_mean:.2f}, Data Radius Error: {data_radius_error:.2f}, Data Radius Std Dev: {data_radius_stddev:.2f}'
        window['output'].update(output_text)
        # Load and display the image
        image_path = values['image_path']
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                image_preview = cv2.resize(image, (256, 256))
                # image_preview = cv2.cvtColor(image_preview, cv2.COLOR_BGR2RGB)
                window['image_preview'].update(data=cv2.imencode('.png', image_preview)[1].tobytes())

        # Update the plot with new data
        """Update these with the real measurements before running"""
        real_width = 100 #mm
        real_length = 20 #mm
        """"""""""""

        data = load_data(colour, image_path, real_width, real_length)
        data_radius_mean, data_radius_stddev, data_radius_error = get_radius_error(data, radius)
        # data_radius_mean, data_radius_stddev = get_radius_mean_stddev(data[:, 1])
        # data_radius_error = error
        x = data[:,0]  # Replace with your own x data
        y = data[:,1]  # Replace with your own y data
        ax.plot(x, y)
        ax.set_title(f'Data for Experiment {iteration}')
        plot_canvas.draw()
    elif event == 'Save':
        # Save the data to the CSV file
        filename = values['csv_path']
        if filename:
            data_to_save = [colour, radius, iteration, data_radius_mean, data_radius_error, data_radius_stddev]
            save_to_csv(filename, column_names, data_to_save)
            sg.popup('Data saved to CSV file!')
    # Increment the iteration counter
    iteration += 1

# Close the GUI window
window.close()






