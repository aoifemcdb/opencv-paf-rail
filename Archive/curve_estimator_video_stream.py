import cv2
import numpy as np

# Define the color range for the red object
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_red_2 = np.array([170, 50, 50])
upper_red_2 = np.array([180, 255, 255])

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)

# Define a function to estimate the shape of the object
def estimate_shape(contour):
    # Simplify the contour
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check the number of corners in the polygon
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "rectangle"
    else:
        return "unknown"

# Main loop to read frames from the camera and estimate the shape of the object
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask to extract the red object
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find the contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and estimate the shape
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Ignore contours that are too small
        if area < 100:
            continue

        # Estimate the shape of the object
        shape = estimate_shape(contour)

        # Draw the contour and shape label on the frame
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.putText(frame, shape, (contour[0][0][0], contour[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
