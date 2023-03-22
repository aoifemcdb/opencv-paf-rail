import cv2
import math
import os
import csv

# Load the image
image_path = "../colour_shape_sensing/input_images/ultrasound.png"
img = cv2.imread(image_path)

# Create a window to display the image
cv2.namedWindow("Image")

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
        cv2.imshow("Image", img_copy)

    # If the left mouse button is clicked for the second time, stop drawing and draw the final line
    elif event == cv2.EVENT_LBUTTONDOWN and line_fixed:
        end_point = (x, y)
        cv2.line(img, start_point, end_point, (0, 0, 255), 2)
        cv2.imshow("Image", img)

        # Measure the length of the line
        length = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        print("Length of line: {:.2f} pixels".format(length))

        # Save the length to a CSV file
        csv_file = os.path.splitext(image_path)[0] + ".csv"
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x1", "y1", "x2", "y2", "length"])
            writer.writerow([start_point[0], start_point[1], end_point[0], end_point[1], length])

        # Reset the start and end points
        start_point = end_point = None
        line_fixed = False


# Register the draw_line function as the callback for mouse events
cv2.setMouseCallback("Image", draw_line)

# Display the image and wait for a key to be pressed
while True:
    if start_point:
        img_copy = img.copy()
        cv2.drawMarker(img_copy, start_point, (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2,
                       line_type=cv2.LINE_AA)
        cv2.imshow("Image", img_copy)
    else:
        cv2.imshow("Image", img)

    key = cv2.waitKey()

    # If the "q" key is pressed, exit the loop
    if key == ord("q"):
        break

# Close the window and exit the program
cv2.destroyAllWindows()
