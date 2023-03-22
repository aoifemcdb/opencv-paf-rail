#!/usr/bin/env python3
import numpy as np
import PySimpleGUI as sg
import rospy
from dataclasses import field, dataclass
from std_msgs.msg import Float32MultiArray, String

from dvrk_arm import Arm
from common import start_button, stop_button


def str2array(trajectory_data: str):
    trajectory_array = np.array(trajectory_data).reshape(3, -1).T
    return trajectory_array


@dataclass
class Trajectory:
    trajectory_sub: rospy.Subscriber = field(init=False)
    start_recording_publisher: rospy.Publisher = field(init=False)
    arm: Arm = Arm(init_arm=True)

    def __post_init__(self):
        self.trajectory_sub = rospy.Subscriber("/case_study/dvrk_trajectory", Float32MultiArray, self.callback)
        self.start_recording_publisher = rospy.Publisher('/case_study/start_recording', String, queue_size=10)

    def callback(self, data):
        """Update the trajectory if the trajectory have changed"""
        # Get the new trajectory from the data
        new_trajectory = str2array(data.data)

        # Update the trajectory if new
        if not (np.array_equal(new_trajectory, self.arm.trajectory)):
            print('Updating trajectory')
            self.arm.trajectory = new_trajectory
            # Create the new poses
            self.arm.create_poses_from_trajectory()
            # Publish the new trajectory
            self.publish_path()

    def publish_path(self):
        self.arm.publish_trajectory()

    def publish_start_recording(self):
        print('Starting the experiment')
        self.start_recording_publisher.publish('Start')

    def publish_stop_recording(self):
        print('Stopping the experiment')
        self.start_recording_publisher.publish('Stop')


def create_gui():
    # Generate the window layout
    layout = [[start_button, stop_button]]
    # Create the window
    window = sg.Window('Trajectory publisher', layout, location=(100, 300), finalize=True,
                       element_justification='center')
    return window


def main():
    """Initializes and cleanup ros node"""
    tr = Trajectory()
    psm1 = tr.arm
    psm1.create_poses_from_trajectory()

    window = create_gui()
    tr.publish_path()

    while not rospy.is_shutdown():
        event, values = window.read(timeout=10)

        if event == '-START-':
            tr.publish_start_recording()
            psm1.follow_trajectory()
            tr.publish_stop_recording()

        if event == '-STOP-':
            psm1.trajectory_thread.stop()

        if event == sg.WIN_CLOSED:
            break


if __name__ == '__main__':
    main()