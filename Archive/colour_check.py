import matplotlib.pyplot as plt

# Define the upper and lower RGB values for each color
colors1 = {
    'red': [(255, 0, 0), (128, 0, 0)],
    'green': [(0, 255, 0), (0, 128, 0)],
    'orange': [(255, 165, 0), (128, 82, 0)],
    'blue': [(0, 0, 255), (0, 0, 128)],
    'yellow': [(255, 255, 0), (128, 128, 0)],
    'purple': [(128, 0, 128), (64, 0, 64)]
}

colors = {
    'red': [(0, 50, 50),(10, 255, 255)],
    'green': [(0, 100, 0), (0, 50, 0)],
    'orange': [(255, 165, 0), (128, 82, 0)],
    'blue': [(0, 0, 255), (0, 0, 128)],
    'yellow': [(255, 255, 0), (128, 128, 0)],
    'purple': [(128, 0, 128), (64, 0, 64)]
}

colors_bgr = {
    'red': [(128, 0, 255), (32, 0, 64)],
    'green': [(0, 100, 0), (0, 50, 0)],
    'orange': [(0, 165, 255), (0, 82, 128)],
    'blue': [(255, 0, 0), (128, 0, 0)],
    'yellow': [(0, 255, 255), (0, 128, 128)],
    'purple': [(128, 0, 128), (64, 0, 64)]
}

colors_hsv = {
    'red': [(150, 255, 255), (0, 128, 64)],
    'green': [(60, 255, 100), (60, 128, 50)],
    'orange': [(20, 255, 255), (10, 128, 128)],
    'blue': [(120, 255, 255), (0, 128, 128)],
    'yellow': [(30, 255, 255), (15, 128, 128)],
    'purple': [(150, 255, 128), (75, 128, 64)]
}

import colorsys

colors_hsv_real = {
    'red': [(0, 1, 1), (0.5, 1, 1)],
    'green': [(0.33, 1, 1), (0.83, 1, 1)],
    'orange': [(0.08, 1, 1), (0.58, 1, 1)],
    'blue': [(0.67, 1, 1), (0.17, 1, 1)],
    'yellow': [(0.17, 1, 1), (0.67, 1, 1)],
    'purple': [(0.83, 1, 0.5), (0.08, 1, 1)]
}

# Convert each color to RGB color space
for color_name, color_values in colors_hsv.items():
    hsv_color1, hsv_color2 = color_values
    rgb_color1 = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*hsv_color1))
    rgb_color2 = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*hsv_color2))
    print(f"{color_name}: Complementary colors: {hsv_color1}, {hsv_color2} | RGB colors: {rgb_color1}, {rgb_color2}")



# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3)

# Loop through each color and plot the shades of color
for i, (color_name, rgb_values) in enumerate(colors1.items()):
    row = i // 3
    col = i % 3
    axs[row, col].set_title(color_name)
    for j, (r, g, b) in enumerate([rgb_values[0], rgb_values[1]]):
        axs[row, col].axhline(j, color=(r/255, g/255, b/255), linewidth=50)
        axs[row, col].set_ylim([-0.5, 1.5])
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

        

# Set the overall title and display the plot
fig.suptitle('Shades of Color')
plt.show()
