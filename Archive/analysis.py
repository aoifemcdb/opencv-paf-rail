import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('../colour_shape_sensing/experiment_images_010623/error_stddevs_050623.csv')

# Extract the columns
radius = data['radius (mm)']
error = data['avg error']
std_dev = data['std dev']

# Create the scatter plot
plt.scatter(radius, error, marker='^', s=15)

# Add error bars using standard deviation
plt.errorbar(radius, error, yerr=std_dev, linestyle='None', capsize=3)

# Set the axis labels
plt.xlabel('Radius')
plt.ylabel('Error')
plt.title('Average sensed radius error (Ground Truth)')

# Set the x-axis limits and ticks
x_ticks = [10, 30, 50, 70, 90, 110, 130]
plt.xlim(min(x_ticks), max(x_ticks))
plt.xticks(x_ticks)

# Display the plot
plt.show()
