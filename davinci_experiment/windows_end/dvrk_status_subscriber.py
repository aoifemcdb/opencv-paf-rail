#!/usr/bin/env python3
import sys

from dataclasses import dataclass, field, Field

from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from std_msgs.msg import Int16
import rospy
import numpy as np

from common import str2array


def plot_positions_2d(positions_x, positions_y):
    # Plot the mean and error bar - first convert to mm
    plt.plot(positions_x, positions_y, linewidth=5, marker='+', ms=15, markeredgewidth=3)

    # Initialize the graph that displays the positions
    plt.title('Fiber position in 2d space')
    plt.xlabel('Axis X (cm)')
    plt.ylabel('Axis Y (cm)')

    plt.minorticks_on()
    plt.xticks(np.arange(0, positions_x[-1] + 1, step=1))
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.grid(which='minor', color='grey', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    return


@dataclass
class DvrkStatus:
    trajectory_str: str
    trajectory: np.ndarray = field(init=False)
    dvrk_status_subscriber: rospy.Subscriber = field(init=False)
    current_position_idx: int = 0
    current_position_scatter: PathCollection = field(init=False)
    figure: Figure = plt.figure()

    def __post_init__(self):
        rospy.init_node('dvrk_status')
        self.dvrk_status_subscriber = rospy.Subscriber('/case_study/dvrk_status', Int16, self.callback)
        self.trajectory = np.array(str2array(self.trajectory_str)).reshape((3, -1)) * 10 ** 2

        # First, plot the full trajectory
        plot_positions_2d(-self.trajectory[0], self.trajectory[2])
        # Then, initialize the scatter line
        self.current_position_scatter = plt.scatter(0, 0, c='r', marker='+', s=300, linewidth=5, zorder=3)

    def callback(self, data):
        """Update the trajectory if the trajectory have changed"""
        # Get the current position index from the data
        self.current_position_idx = data.data

        print(data.data)
        current_x = -self.trajectory[0, self.current_position_idx]
        current_y = self.trajectory[2, self.current_position_idx]

        # Update the scatter position
        self.current_position_scatter.set_offsets(np.array([current_x, current_y]))
        # And re-draw the figure
        self.figure.canvas.draw()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: trajectory_publisher.py trajectory_array")
    else:
        print('trajectory publisher ', sys.argv[1])
        start_recording_subscriber = DvrkStatus(sys.argv[1])

    plt.show()
    rospy.spin()