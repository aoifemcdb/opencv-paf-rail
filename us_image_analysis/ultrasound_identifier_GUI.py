import PySimpleGUI as sg

# Define the layout for the window
layout = [
    [sg.Text("Which image was done by a robot?")],
    [sg.Column([[sg.Image("./input_images/ultrasound.png", size=(400, 400)), sg.Combo(["Robot", "Human"], key="-IMAGE1-", size=(10, 20))]], element_justification='c'),
     sg.Column([[sg.Image("./input_images/ultrasound.png", size=(400, 400)), sg.Combo(["Robot", "Human"], key="-IMAGE2-", size=(10, 20))]], element_justification='c'),
     sg.Column([[sg.Image("./input_images/ultrasound.png", size=(400, 400)), sg.Combo(["Robot", "Human"], key="-IMAGE3-", size=(10, 20))]], element_justification='c')],
    [sg.Button("Submit")]
]

# Create the window
window = sg.Window("Ultrasound Image Classifier", layout)

# Start the event loop
while True:
    event, values = window.read()

    # If the window is closed, exit the program
    if event == sg.WIN_CLOSED:
        break

    # If the user clicks the "Submit" button, check their answers
    if event == "Submit":
        # Get the user's answers from the drop-down menus
        answer1 = values["-IMAGE1-"]
        answer2 = values["-IMAGE2-"]
        answer3 = values["-IMAGE3-"]

        # Check the answers and display a message
        if answer1 == "Robot" and answer2 == "Human" and answer3 == "Human":
            sg.popup("Congratulations! You correctly identified the robot image.")
        else:
            sg.popup("Sorry, one or more of your answers were incorrect.")

# Close the window and exit the program
window.close()


