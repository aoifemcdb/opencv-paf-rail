import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.colors import ListedColormap

# Set the font family to match LaTeX font
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 18})

#colors:
#yellow #F0D000
#blue #235789
#red #C1292E
#black #020100
robot_color = '#235789'  # Color for robot
human_color = '#C1292E'

def create_percentage_bar_chart(csv_file_path):
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(csv_file_path)

        # Set the 'Score' column as the index
        data.set_index('Score', inplace=True)

        # Transpose the DataFrame to swap rows and columns
        data = data.transpose()

        # Define colors for robot and human bars
        robot_color = '#235789'
        human_color = '#C1292E'

        # Define colors for each score (1 to 5)
        score_colors = ['#cc3232', '#db7b2b', '#e7b416', '#99c140', '#2dc937']

        # Create a custom colormap
        custom_cmap = ListedColormap(score_colors)

        # Create a stacked bar chart


        ax = data.plot(kind='bar', stacked=True, figsize=(5, 6), colormap=custom_cmap, width=0.6)

        # Set labels and title
        # plt.xlabel('Allocation Source (Human vs. Robot)')
        plt.ylabel('Percentage of Total Scores')
        plt.title('Reader 2: DET')

        # Show legend
        plt.legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Show the plot
        # plt.gca().legend().set_visible(False)
        plt.tight_layout()
        dpi = 300
        plt.savefig('./experiment_data_260623/rails/bar_DET_AR.png', dpi=dpi)
        plt.show()

    # Replace 'your_file.csv' with the actual paths to your CSV files



def create_stacked_bar_chart(data, x_labels, criteria_labels, title, y_labels):
    """
    Create a stacked bar chart.

    Parameters:
        data (dict): A dictionary with criteria as keys and a list of values for each criteria.
                     Each value is a dictionary with sub-criteria as keys and a list of values for each sub-criteria.
        x_labels (list): List of labels for each x value.
        criteria_labels (list): List of labels for each criteria (e.g., ['RES', 'DET', 'IQ']).
        title (str): Title of the chart.

    Returns:
        None
    """
    fig, ax = plt.subplots()

    num_x_values = len(x_labels)
    num_criteria = len(criteria_labels)
    width = 0.2
    robot_color = '#235789'  # Color for robot
    human_color = '#C1292E'  # Color for human
    space_between_bars = 0.1
    group_width = (width + space_between_bars) * num_criteria
    x_positions = [x + group_width * i for x in range(num_x_values) for i in range(num_criteria)]
    max_value = 0  # To store the maximum value in the data

    for i, criteria in enumerate(criteria_labels):
        sub_criteria_data = data[criteria]
        sub_criteria_labels = list(sub_criteria_data.keys())
        bottom = [0] * len(x_labels)


        for j, sub_criteria in enumerate(sub_criteria_labels):
            values = sub_criteria_data[sub_criteria]
            alpha = 1.0 - i * 0.2
            color = robot_color if 'robot' in sub_criteria else human_color
            ax.bar([x + i * width for x in range(len(x_labels))], values, width, bottom=bottom, label=f'{criteria} {sub_criteria}', align='center', color=color, alpha=alpha, edgecolor='black')
            bottom = [sum(x) for x in zip(bottom, values)]
            x_positions = [x + width + space_between_bars for x in x_positions]
            # Find the maximum value to adjust the y-axis limit
            max_value = max(max_value, max(bottom))


    ax.set_ylabel(y_labels)
    ax.set_title(title)

    label_offset = 3  # Adjust this value to position the labels

    for i in range(num_x_values):
        if x_labels[i] in [2, 3, 4]:  # Add label only for x values 2, 3, and 4
            x_val = x_positions[i * num_criteria] - 9 * width
            ax.text(x_val, max_value, 'RES', ha='center', va='bottom', fontsize=8, rotation=45)

    for i in range(num_x_values):
        if x_labels[i] in [2, 3, 4]:  # Add label only for x values 2, 3, and 4
            x_val = x_positions[i * num_criteria] -  7.8* width
            ax.text(x_val, max_value, 'DET', ha='center', va='bottom', fontsize=8, rotation=45)

    for i in range(num_x_values):
        if x_labels[i] in [2, 3, 4]:  # Add label only for x values 2, 3, and 4
            x_val = x_positions[i * num_criteria] -  6.8* width
            ax.text(x_val, max_value, 'IQ', ha='center', va='bottom', fontsize=8, rotation=45)

    # Extend y-axis to have space for the labels on top of the bars
    ax.set_ylim(0, max_value * 1.1)

    plt.xticks([x + width for x in range(len(x_labels))], x_labels)

    # Add labels for xticks
    labels_dict = {1: 'Very Poor', 2: 'Poor', 3: 'Good', 4: 'Very Good', 5: 'Excellent'}
    ax.set_xticklabels([labels_dict[label] for label in x_labels])

    # Create custom legend for human and robot
    human_patch = mpatches.Patch(color=human_color, label='Human')
    robot_patch = mpatches.Patch(color=robot_color, label='Robot')
    plt.legend(handles=[human_patch, robot_patch])

    plt.show()

# Data from TS Questionnaire:
# data = {
#     'RES': {
#         '% human': [0, 100, 36, 11, 00],
#         '% robot': [0, 0, 64, 89, 0]
#
#     },
#     'DET': {
#         '% human': [0, 19, 14, 0, 0],
#         '% robot': [0, 14, 29, 100, 0]
#     },
#     'IQ': {
#         '% human': [0, 100, 45, 0, 0],
#         '% robot': [0, 0, 55, 100, 0]
#     }
# }
#
# x_labels = [1, 2, 3, 4, 5]
# criteria_labels = ['RES', 'DET', 'IQ']
# title = 'Ultrasound Reviewer 1'
# y_labels = 'Percentage of scores assigned to robot or human \nfor each criteria: RES, DET, IQ (%)'

# create_stacked_bar_chart(data, x_labels, criteria_labels, title, y_labels)


def generate_confusion_matrix(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file_path)

    # Extract the scores given by Person 1 and Person 2, and the video IDs
    person1_scores = data['TS Score']
    person2_scores = data['AR Score']

    # Get unique scores from both persons and create a set of all possible scores (1 to 5)
    unique_scores = sorted(set(person1_scores) | set(person2_scores))
    all_possible_scores = list(range(1, 6))

    # Create the confusion matrix with zeros for all possible score pairs
    confusion_mat = confusion_matrix(person1_scores, person2_scores, labels=all_possible_scores)

    # Create a DataFrame for the confusion matrix for better visualization
    confusion_df = pd.DataFrame(confusion_mat, index=all_possible_scores, columns=all_possible_scores)


    # Plot the confusion matrix
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray', cbar='False', cbar_kws={'use_gridspec': False})
    plt.xlabel('Reader 2')
    plt.ylabel('Reader 1')
    plt.title('Confusion Matrix for IQ')
    # plt.show()
    dpi = 300
    plt.savefig('./experiment_data_260623/rails/confusion_IQ.png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)

    return confusion_df

# # Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'percentage_DET_AR.csv'
create_percentage_bar_chart(csv_file_path)

# confusion_matrix_df = generate_confusion_matrix(csv_file_path)
# print("Confusion Matrix:")
# print(confusion_matrix_df)






















