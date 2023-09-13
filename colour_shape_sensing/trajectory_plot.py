import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# Set the font family to match LaTeX font
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Read the data from the CSV file
df = pd.read_csv("single_rail_test.csv")


x_subset = df['x'][::10]
y_subset = df['y'][::10]
x_robot_subset = df['x_robot'][::10]
y_robot_subset = df['y_robot'][::10]

# Calculate RMSE for x and y separately
rmse_x = np.sqrt(((x_subset - x_robot_subset) ** 2).mean())
rmse_y = np.sqrt(((y_subset - y_robot_subset) ** 2).mean())



print(f"RMSE for x: {rmse_x:.4f}")
print(f"RMSE for y: {rmse_y:.4f}")

# If you want the combined RMSE for both x and y
rmse_combined = np.sqrt(((x_subset - x_robot_subset) ** 2 + (y_subset - y_robot_subset) ** 2).mean())
print(f"Combined RMSE: {rmse_combined:.4f}")

# Plotting
fig, ax = plt.subplots(figsize=(8,4))
df['y'] = -df['y']
df['y_robot'] = -df['y_robot']


# Plot all the x,y points
ax.plot(df['x'], df['y'], color='#235789', linestyle='--')

# Highlight every 10th point in x,y with a star marker
ax.scatter(df['x'][::10], df['y'][::10], s=50, color='#235789', label='Planned Trajectory', facecolors='None', edgecolors='#235789', marker='*')

# Plot every 10th x_robot, y_robot point with a dashed line
# Since scatter doesn't directly support linestyle, you can create a plot for the line
ax.plot(df['x_robot'][::10], df['y_robot'][::10], linestyle='dashed', color='#C1292E', label='Executed Trajectory')
# And use scatter for the points
ax.scatter(df['x_robot'][::10], df['y_robot'][::10], color='#C1292E', label=None)  # label set to None to avoid double legend entries

# Adding a text box with RMSE values
textstr = f'RMSE x: {rmse_x:.4f}\nRMSE y: {rmse_y:.4f}\nCombined RMSE: {rmse_combined:.4f}'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white'))



# Setting labels, legend, and title
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.legend()
ax.set_title('Planned Trajectory vs Executed Trajectory for the UR3')
# Set axes limits
x_min = -10  # Define the minimum value for the x-axis
x_max = 100  # Define the maximum value for the x-axis
y_min = -2 # Define the minimum value for the y-axis
y_max = 8  # Define the maximum value for the y-axis
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Planned Trajectory vs Executed Trajectory of UR3')
dpi=300
plt.savefig('planned_vs_executed.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

plt.show()





