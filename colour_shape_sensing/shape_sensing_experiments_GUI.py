import PySimpleGUI as sg
import matplotlib.pyplot as plt
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Create a list of available radii
radii = [30, 50, 70, 90, 110]
colors = ["red", "green", "blue", "yellow", "orange"]
# Create the layout for the window
layout = [
    [sg.Text("Experiment Iteration: "), sg.InputCombo([i for i in range(1, 6)], default_value=1)],
    [sg.Text("Radius: "), sg.DropDown(radii, default_value=30, key="-RADIUS-")],
    [sg.Text("Colour: "),
     sg.DropDown(colors, default_value="red", key="-COLOUR-")],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Text("Save Location: "), sg.InputText(key="-SAVE-LOCATION-"), sg.FolderBrowse(button_text='Browse')],
    [sg.Button("Save", key="-SAVE-"), sg.Button("Generate Graph", key="-GENERATE-")],
    [sg.Text("", size=(30,1), key="-ERROR-")],
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
    y = [(radius ** 2 - i ** 2) ** 0.5 if radius ** 2 - i ** 2 >= 0 else 0 for i in x]

    # Load the data from a CSV file containing 10 x,y co-ordinates
    try:
        with open('curve_points.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            data = [list(map(float, row)) for row in reader]
        dummy_x, dummy_y = np.array(data).T
    except FileNotFoundError:
        dummy_x = np.linspace(0, radius, num=10)
        dummy_y = np.interp(dummy_x, x, y) + np.random.normal(0, 5, size=(10,))

    # Add the dummy data to the plot
    ax.clear()
    ax.plot(x, y, color=colour)
    ax.plot(dummy_x, dummy_y, color='black')

    # Set the x and y axis limits
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    # Add title and axis labels to the plot
    ax.set_title(f"Circle with Radius {radius} ({colour})")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")

    # Compute the radius of the original curve
    original_radius = radius

    # Compute the radius of the dummy data
    # Interpolate the dummy
    # Read the data from the CSV file
    with open('curve_points.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # skip header row
        data = []
        for row in csv_reader:
            data.append([float(row[0]), float(row[1])])

    # Convert data to numpy array
    data = np.array(data)

    # Add the data to the plot
    ax.plot(data[:, 0], data[:, 1], color='black')

    # Compute the radius of the original curve
    original_radius = radius

    # Compute the radius of the dummy data
    dummy_radius = np.sqrt((data[0][0] ** 2 + data[0][1] ** 2))

    # Interpolate the dummy data
    interp_y = np.interp(dummy_x, data[:, 0], data[:, 1], left=0, right=0)

    # Compute the error between the radii and standard deviation
    error = abs(original_radius - dummy_radius) + np.std(dummy_radius - np.sqrt(dummy_x ** 2 + interp_y ** 2))

    # Update the error text in the GUI
    window["-ERROR-"].update(
        f"Error: {error:.2f} +/- {np.std(dummy_radius - np.sqrt(dummy_x ** 2 + dummy_y ** 2)):.2f}")

    # Convert the plot to a Tkinter canvas and add it to the GUI window
    canvas = FigureCanvasTkAgg(fig, master=window["-CANVAS-"].TKCanvas)
    canvas.get_tk_widget().grid(row=0, column=0)

# Save the plot and data when the Save button is clicked
    if event == "-SAVE-":
        save_location = values["-SAVE-LOCATION-"]
        filename = f"experiment_{experiment_iteration}_radius_{radius}_colour_{colour}"


        fig.savefig(f"{save_location}/{filename}.png")
        with open(f"{save_location}/{filename}.csv", mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Experiment Iteration", "Radius", "Colour", "Error"])
            writer.writerow([experiment_iteration, radius, colour, error])
            writer.writerow(["X (dummy)", "Y (dummy)"])
            for i in range(len(dummy_x)):
                writer.writerow([dummy_x[i], dummy_y[i]])
            writer.writerow([])
            writer.writerow(["X (original)", "Y (original)"])
            for i in range(len(data)):
                writer.writerow([data[i][0], data[i][1]])
            writer.writerow([])
            writer.writerow(["X (interpolated)", "Y (interpolated)"])
            for i in range(len(dummy_x)):
                writer.writerow([dummy_x[i], interp_y[i]])

window.close()










