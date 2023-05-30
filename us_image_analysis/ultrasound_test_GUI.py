import cv2
import math
import os
import csv
import PySimpleGUI as sg

# Define the GUI layout
layout = [
    [sg.Text("Select an image file:")],
    [sg.Input(key="FILE"), sg.FileBrowse()],
    [sg.Image(key="IMAGE")],
    [sg.Text(key="LENGTH", visible=False)],
    [sg.Button("Measure"), sg.Button("Clear"), sg.Button("Exit")]
]

# Create the GUI window
window = sg.Window("Image Measurement Tool", layout)

# Initialize variables
start_point = None
end_point = None
line_fixed = False


# Define a function to draw a line on the image
def draw_line(event, x, y, flags, params):
    global start_point, end_point, line_fixed

    # If the left mouse button is clicked for the first time, start drawing
    if event == cv2.EVENT_LBUTTONDOWN and not line_fixed:
        start_point = (x, y)
        line_fixed = True

    # If the mouse is moved while drawing, draw the temporary line
    elif event == cv2.EVENT_MOUSEMOVE and line_fixed:
        img_copy = img.copy()
        cv2.line(img_copy, start_point, (x, y), (0, 0, 255), 2)
        window["IMAGE"].update(data=cv2.imencode(".png", img_copy)[1].tobytes())

    # If the left mouse button is clicked for the second time, stop drawing and draw the final line
    elif event == cv2.EVENT_LBUTTONDOWN and line_fixed:
        end_point = (x, y)
        cv2.line(img, start_point, end_point, (0, 0, 255), 2)
        window["IMAGE"].update(data=cv2.imencode(".png", img)[1].tobytes())

        # Measure the length of the line
        length = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        window["LENGTH"].update("Length of line: {:.2f} pixels".format(length), visible=True)

        # Save the length to a CSV file
        csv_file = os.path.splitext(values["FILE"])[0] + ".csv"
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x1", "y1", "x2", "y2", "length"])
            writer.writerow([start_point[0], start_point[1], end_point[0], end_point[1], length])

        # Reset the start and end points
        start_point = end_point = None
        line_fixed = False


# Event loop to handle GUI events
while True:
    event, values = window.read()

    # Exit the program if the window is closed or the "Exit" button is clicked
    if event == sg.WINDOW_CLOSED or event == "Exit":
        break

    # If the "Measure" button is clicked, open the image file and display it
    if event == "Measure":
        image_path = values["FILE"]
        img = cv2.imread(image_path)
        window["IMAGE"].update(data=cv2.imencode(".png", img)[1].tobytes())

        # Register the draw_line function as the callback for mouse events
        cv2.namedWindow("Image")

        cv2.setMouseCallback("Image", draw_line)

        # If the "Clear" button is clicked, reset the image and measurement text
    if event == "Clear":
        img = cv2.imread(image_path)
        window["IMAGE"].update(data=cv2.imencode(".png", img)[1].tobytes())
        window["LENGTH"].update(visible=False)

    # Destroy the OpenCV window
    # cv2.destroyAllWindows()

    # Close the GUI window
    # window.close()





