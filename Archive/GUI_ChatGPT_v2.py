import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create a list of available radii
radii = [30, 50, 70, 90, 110]

# Create the layout for the window
layout = [
    [sg.Text("Experiment Iteration: "), sg.InputCombo([i for i in range(1, 6)], default_value=1)],
    [sg.Text("Radius: "), sg.DropDown(radii, default_value=30, key="-RADIUS-")],
    [sg.Text("Colour: "),
     sg.DropDown(["red", "green", "blue", "yellow", "orange"], default_value="red", key="-COLOUR-")],
    [sg.Text("Save Location: "), sg.InputText(), sg.FolderBrowse(button_text='Browse')],
    [sg.Button("Generate Graph", key="-GENERATE-")],
    [sg.Canvas(key="-CANVAS-")],
]

# Create the window
window = sg.Window("Graph Generator", layout)

# Create a figure and axis object for plotting
fig, ax = plt.subplots()

# Define the limits for the x and y axis
x_limit = (0, 120)
y_limit = (0, 120)

# Run the GUI event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break

    # Get the selected radius and colour from the GUI inputs
    radius = int(values["-RADIUS-"])
    colour = values["-COLOUR-"]

    # Get the selected experiment iteration
    experiment_iteration = values[0]

    # Generate the data for the plot based on the selected radius and colour
    x = [i for i in range(radius + 1)]
    y = [(radius ** 2 - i ** 2) ** 0.5 for i in x]

    # Clear the previous plot and plot the new data
    ax.clear()
    ax.plot(x, y, color=colour)

    # Set the x and y axis limits
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    # Add title and axis labels to the plot
    ax.set_title(f"Circle with Radius {radius} ({colour})")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

    # Convert the plot to a Tkinter canvas and add it to the GUI window
    canvas = FigureCanvasTkAgg(fig, master=window["-CANVAS-"].TKCanvas)
    canvas.get_tk_widget().grid(row=0, column=0)

# Close the window and exit the program
window.close()



