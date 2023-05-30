import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = '../colour_shape_sensing/output_trajectories/trajectory_2.csv'
#load data as numpy array
def get_trajectory(filename):
    data = pd.read_csv(filename).values
    x = data[:,0]
    y = data[:,1]
    return x,y

def cubic_interpolate(x,y, number_of_points: int):
    cubic_spline_interp = scipy.interpolate.CubicSpline(x,y)
    x_new = np.linspace(0, max(x), number_of_points)
    y_new = cubic_spline_interp(x_new)
    return x_new, y_new

def get_average_curvature(x_new,y_new):
    xy_new = np.array([x_new,y_new])
    xy_new = xy_new.T

    dx = np.gradient(xy_new[:,0])
    dy = np.gradient(xy_new[:,1])

    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    curv = np.abs(d2y)/((1+dy**2)**1.5)
    average_curv = np.average(curv)
    print("Average curvature: " + str(average_curv))
    average_radius = np.reciprocal(average_curv)
    print("Average radius: " + str(average_radius))
    return average_curv, average_radius


x,y = get_trajectory(filename)
x_new, y_new = cubic_interpolate(x,y,100)
average_curv, average_radius = get_average_curvature(x_new, y_new)

plt.figure(figsize=(12,4))
plt.plot(x_new,y_new)
plt.text(300, 120, 'Average Curvature: '+ f'{average_curv:.2f} mm^{-1}', fontsize=10,
             bbox=dict(facecolor = 'None', edgecolor='black' ,alpha=0.5))
plt.text(300, 105, 'Average Radius: '+ f'{average_radius:.2f} mm', fontsize=10,
             bbox=dict(facecolor = 'None', edgecolor='black' ,alpha=0.5))
plt.xlim([-300, 1000])
plt.show()
