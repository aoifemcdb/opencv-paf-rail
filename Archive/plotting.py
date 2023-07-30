import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ----------------------------- EMBED CANVAS ------------------------

def draw_figure(canvas, figure, loc=(0,0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def plot_curvature_ground_truth(axs, radius_mm, grating_indexes):
    """ Plot the compared curvature is radius_mm is not None """
    # Get the line that plot the ground truth curvature
    line, = axs.plot([],[])
    if radius_mm is not None:
        radius_cm = radius_mm * 10 ** -1
        fixed_curvature = np.zeros(grating_indexes) + 1 / radius_cm
        line.set_data(grating_indexes, fixed_curvature)
        line.set_label('Curvature ' + str(radius_mm) + ' $mm^{-1}$')
    else:
        reset_lines(line)
    return