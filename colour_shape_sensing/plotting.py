import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set the font family to match LaTeX font
mpl.rcParams['font.family'] = 'serif'

#colors:
#yellow #F0D000
#blue #235789
#red #C1292E
#black #020100
# plt.style.use('seaborn') # I personally prefer seaborn for the graph style, but you may choose whichever you want.
# params = {"ytick.color" : "black",
#           "xtick.color" : "black",
#           "axes.labelcolor" : "black",
#           "axes.edgecolor" : "black",
#           "text.usetex" : True,
#           "font.family" : "serif",
#           "font.serif" : ["Computer Modern Serif"]}
# plt.rcParams.update(params)

def plot_data_against_fbgs():
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv('./experiment_data_260623/rails/all_data_results.csv')

    # Extract the required columns
    radius = data['radius']
    gt_measured_radius = data['gt_radius_measured']
    gt_avg_err = data['gt_mean_err']
    gt_std_dev = data['gt_std_dev']

    rails_measured_radius = data['rails_radius_measured']
    rails_error = data['rails_mean_err']
    rails_std_dev = data['rails_std_dev']
    rail_min_deviation = data['rail_min_deviation']
    rail_max_deviation = data['rail_max_deviation']

    measured_radius = data['colorbands_radius_measured']
    error = data['colorbands_mean_err']
    std_dev = data['colorbands_std_dev']

    fbgs_radius = data['fbgs_avg']
    fbgs_avg_err = data['fbgs_avg_err']
    fbgs_std_dev = data['fbgs_std_dev']

    #FBGS color #F1D302

    # Plotting
    # Add diagonal line and shaded band
    x = np.linspace(0, 150, 100)
    plt.plot(x, x, color='gray', linestyle='--', label='Geometric radius')
    # plt.fill_between(x, x - 4, x + 4, where=(x >= x - 2) & (x <= x + 2), color='lightgray', alpha=0.5,
    #                  label='Geometric radius +/- uncertainty')
    #

    # Plot shaded areas for each radius value
    for r, rail_radius, min_dev, max_dev in zip(radius, rails_measured_radius, rail_min_deviation, rail_max_deviation):
        x_range = np.linspace(r - 3, r + 3, 100)  # Adjust the range as needed
        plt.fill_between(x_range, rail_radius - max_dev, rail_radius - min_dev, color='lightgray', alpha=0.5)
        plt.fill_between(x_range, rail_radius + min_dev, rail_radius + max_dev, color='lightgray', alpha=0.5)

    # Add a single label for the deviation range
    plt.fill_between([], [], [], color='lightgray', alpha=0.5,
                     label='Deviation range \nof rail from geometric radius (mm)')
    plt.scatter(radius, fbgs_radius, marker='v', color='#F0D000', label='FBGS')
    # plt.scatter(radius, gt_measured_radius, color='#C1292E', label='Vision - Ground Truth')
    plt.scatter(radius, measured_radius, color='#235789', label='Vision - Colorbands on Staircase')
    plt.scatter(radius, rails_measured_radius, color='#020100', label='Vision - Rails on Staircase')

    plt.errorbar(radius, fbgs_radius, yerr = fbgs_std_dev, fmt = 'none', capsize=6, color='#F0D000')
    plt.errorbar(radius, gt_measured_radius, yerr=gt_std_dev, fmt='none', capsize=6, color='#C1292E')
    plt.errorbar(radius, measured_radius, yerr=std_dev, fmt='none', capsize=6, color='#235789')
    plt.errorbar(radius, rails_measured_radius, yerr=rails_std_dev, fmt='none', capsize=6, color='#020100')

    # for r, mr, e in zip(radius, measured_radius, error):
    #     if r > mr:
    #         plt.plot([r, r], [mr, mr + e], color='#C1292E', linestyle='dotted')
    #     else:
    #         plt.plot([r, r], [mr - e, mr], color='#C1292E', linestyle='dotted')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Radius (mm)')
    plt.xticks([10, 30, 50, 70, 90, 110, 130])
    plt.yticks([10, 30, 50, 70, 90, 110, 130])
    plt.title('Rails vs Vision Ground Truth Shape Sensing')
    plt.legend(fontsize='small')
    # plt.show()
    dpi=300
    plt.savefig('./experiment_data_260623/rails/experiment_one_all_data.png', dpi=dpi)
    return

def plot_data_against_data():
    data_1 = pd.read_csv('./experiment_data_210623/circle_fitting_method/experiment_iteration_two_results.csv')
    data_2 = pd.read_csv('./experiment_data_250623/experiment_one_results.csv')

    # Extract the required columns for the first dataset
    radius_1 = data_1['radius']
    measured_radius_1 = data_1['radius_measured']
    error_1 = data_1['mean_err']
    std_dev_1 = data_1['std_dev']

    # Extract the required columns for the second dataset
    radius_2 = data_2['radius']
    measured_radius_2 = data_2['radius_measured']
    error_2 = data_2['mean_err']
    std_dev_2 = data_2['std_dev']

    # Plotting
    # Add diagonal line and shaded band
    x = np.linspace(0, 150, 100)
    plt.plot(x, x, color='gray', linestyle='--')
    plt.fill_between(x, x - 2, x + 2, where=(x >= x - 2) & (x <= x + 2), color='lightgray', alpha=0.5,  label='Geometric radius +/- uncertainty')

    # plt.scatter(radius_1, radius_1, color='blue')
    # plt.scatter(radius_2, radius_2, color='#235789', label='Geometric')

    plt.scatter(radius_1, measured_radius_1, color='#235789', label='First experiment round')
    plt.scatter(radius_2, measured_radius_2, color='#C1292E', label='Second experiment round')

    plt.errorbar(radius_1, measured_radius_1, yerr=std_dev_1, fmt='none', capsize=4, color='#235789')
    plt.errorbar(radius_2, measured_radius_2, yerr=std_dev_2, fmt='none', capsize=4, color='#C1292E')

    for r1, mr1, e1, r2, mr2, e2 in zip(radius_1, measured_radius_1, error_1, radius_2, measured_radius_2, error_2):
        plt.plot([r1, r2], [mr1, mr2], color='black', linestyle='dotted')

    plt.xlabel('Radius (mm)')
    plt.ylabel('Radius (mm)')
    plt.xticks([10, 30, 50, 70, 90, 110, 130])
    plt.yticks([10, 30, 50, 70, 90, 110, 130])
    plt.title('Sensed radius, first round vs second round')
    plt.legend()



    plt.show()
    # dpi = 300
    # plt.savefig('./experiment_data_250623/experiment_one_first_round_vs_second_comparison_shaded_groundtruth.png', dpi=dpi)
    return

def plot_errors():
    data = pd.read_csv('./experiment_data_260623/colorbands/experiment_one_results_with_250623data.csv')
    radius = data['radius']
    vision_err = data['mean_err']
    vision_std_dev = data['std_dev']
    colorbands_err = data['colorbands_mean_err']
    colorbands_std_dev = data['colorbands_std_dev']
    fbgs_err = data['fbgs_avg_err']
    fbgs_std_dev = data['fbgs_std_dev']

    # Plotting
    # Add line at y=1 with shaded area of +/-2
    plt.axhline(y=0, color='gray', linestyle='--')
    x = np.linspace(10, 130, 100)
    plt.fill_between(x, 0 - 2, 0 + 2, color='lightgray', alpha=0.5, label='Geometric ground truth +/- uncertainty')
    plt.scatter(radius, colorbands_err, color='#235789', label='Colorbands Error')
    plt.scatter(radius, fbgs_err, color='#F0D000', label='FBGS Error')
    plt.scatter(radius, vision_err, color='#C1292E', label='Vision Error')

    plt.errorbar(radius, colorbands_err, yerr=colorbands_std_dev, fmt='none', capsize=4, color='#235789')
    plt.errorbar(radius, fbgs_err, yerr=fbgs_std_dev, fmt='none', capsize=4, color='#F0D000')
    plt.errorbar(radius, vision_err, yerr=vision_std_dev, fmt='none', capsize=4, color='#C1292E')

    plt.xlabel('Radius (mm)')
    plt.ylabel('Error (mm)')
    plt.xticks([10, 30, 50, 70, 90, 110, 130])
    plt.yticks([0, 5, 10, 15, 20, 25])
    plt.title('Ground Truth Shape Sensing Errors')
    plt.legend()
    plt.show()
    # dpi = 300
    # plt.savefig('./experiment_data_260623/colorbands/experiment_one_error_vs_fbgs_vs_gt.png', dpi=dpi)
    return

def plot_data_against_fbgs_corrected():
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv('./experiment_data_260623/rails/all_data_results_manipulated.csv')

    # Extract the required columns
    radius = data['radius']
    gt_measured_radius = data['gt_radius_measured']
    gt_avg_err = data['gt_mean_err']
    gt_std_dev = data['gt_std_dev']

    rails_measured_radius = data['rails_radius_corrected']
    rails_error = data['rails_mean_err']
    rails_std_dev = data['rails_std_dev']
    rail_min_deviation = data['rail_min_deviation']
    rail_max_deviation = data['rail_max_deviation']

    measured_radius = data['colorbands_radius_measured']
    error = data['colorbands_mean_err']
    std_dev = data['colorbands_std_dev']

    fbgs_radius = data['fbgs_avg']
    fbgs_avg_err = data['fbgs_avg_err']
    fbgs_std_dev = data['fbgs_std_dev']

    #FBGS color #F1D302

    # Plotting
    # Add diagonal line and shaded band
    x = np.linspace(0, 150, 100)
    plt.plot(x, x, color='gray', linestyle='--', label='Geometric radius')
    # plt.fill_between(x, x - 4, x + 4, where=(x >= x - 2) & (x <= x + 2), color='lightgray', alpha=0.5,
    #                  label='Geometric radius +/- uncertainty')
    #

    # Plot shaded areas for each radius value
    for r, rail_radius, min_dev, max_dev in zip(radius, rails_measured_radius, rail_min_deviation, rail_max_deviation):
        x_range = np.linspace(r - 3, r + 3, 100)  # Adjust the range as needed
        plt.fill_between(x_range, rail_radius - max_dev, rail_radius, color='lightgray', alpha=0.5)
        plt.fill_between(x_range, rail_radius + max_dev, rail_radius, color='lightgray', alpha=0.5)

    # Add a single label for the deviation range
    plt.fill_between([], [], [], color='lightgray', alpha=0.5,
                     label='Deviation range \n of rail from geometric radius (mm)')
    plt.scatter(radius, fbgs_radius, marker='v', color='#F0D000', label='FBGS')
    # plt.scatter(radius, gt_measured_radius, color='#C1292E', label='Vision - Ground Truth')
    # plt.scatter(radius, measured_radius, color='#235789', label='Vision - Colorbands on Staircase')
    plt.scatter(radius, rails_measured_radius, color='#020100', label='Vision - Rails on Staircase')

    plt.errorbar(radius, fbgs_radius, yerr = fbgs_std_dev, fmt = 'none', capsize=6, color='#F0D000')
    # plt.errorbar(radius, gt_measured_radius, yerr=gt_std_dev, fmt='none', capsize=6, color='#C1292E')
    # plt.errorbar(radius, measured_radius, yerr=std_dev, fmt='none', capsize=6, color='#235789')
    plt.errorbar(radius, rails_measured_radius, yerr=rails_std_dev, fmt='none', capsize=6, color='#020100')

    # for r, mr, e in zip(radius, measured_radius, error):
    #     if r > mr:
    #         plt.plot([r, r], [mr, mr + e], color='#C1292E', linestyle='dotted')
    #     else:
    #         plt.plot([r, r], [mr - e, mr], color='#C1292E', linestyle='dotted')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Radius (mm)')
    plt.xticks([10, 30, 50, 70, 90, 110, 130])
    plt.yticks([10, 30, 50, 70, 90, 110, 130])
    plt.title('Vision sensed PAF Rails vs Geometric Radius & FBGS')
    plt.legend(fontsize='small')
    dpi=300
    plt.savefig('./experiment_data_260623/rails/experiment_one_rails_gt_vs_FBGS_only.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()
    return

def plot_warped_data():
    # Replace 'your_data.csv' with the actual file path of your CSV data
    file_path = './experiment_data_260723/all_radii_data_final.csv'

    # Read the CSV data into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Assuming the first column is the radius, and the rest are mean and std dev for each set
    radius_column = 'radius'
    mean_columns = ['angle_0_mean', 'angle_10_mean', 'angle_20_mean', 'angle_30_mean']
    std_dev_columns = ['angle_0_std_dev', 'angle_10_std_dev', 'angle_20_std_dev', 'angle_30_std_dev']
    set_labels = ['Camera Angle: 0°', 'Camera Angle: 10°', 'Camera Angle: 20°', 'Camera Angle: 30°']
    set_colors = ['#F0D000', '#235789', '#C1292E', '#020100']

    # Create a scatter plot with error bars for each set of mean and std dev
    for i in range(len(mean_columns)):
        plt.errorbar(df[radius_column], df[mean_columns[i]], yerr=df[std_dev_columns[i]], label=set_labels[i],
                     linestyle='None', marker='o', color=set_colors[i], capsize=6)

    # Add custom text labels for each set in the plot
    # for i in range(len(mean_columns)):
    #     plt.text(0.95, 0.85 - i * 0.05, set_labels[i], ha='right', transform=plt.gca().transAxes)

    # Create a separate box (like a legend) for the custom labels
    # legend_box_text = '\n'.join(set_labels)
    # plt.annotate(legend_box_text, xy=(0.08, 0.8), xycoords='axes fraction', fontsize=10,
    #              bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    x = np.linspace(0, 150, 100)
    plt.plot(x, x, color='gray', linestyle='--', label='Geometric radius')

    # Set plot labels and titles
    plt.xlabel('Radius (mm)')
    plt.ylabel('Sensed Radius (mm)')
    plt.title('Sensed Radius vs Ground Truth Radius \nfor Camera Angles ranging 0° to 30°')
    plt.legend()
    plt.xticks([10, 30, 50, 70, 90, 110, 130])
    plt.yticks([10, 30, 50, 70, 90, 110])

    # Show the plot
    plt.grid(False)
    # plt.show()
    dpi=300
    plt.savefig('./experiment_data_260623/rails/experiment_two_warped_data.pdf', dpi=dpi)



# plot_errors()
plot_data_against_fbgs_corrected()
# plot_errors()
# plot_warped_data()
# def main():
# plot_data_against_data()
#     return
#
#
# if __name__ == 'main':
#     main()