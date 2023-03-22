#!/usr/bin/env python3

import PySimpleGUI as sg

import rospy
import PyKDL
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from tf_conversions import posemath
from dvrk_arm import *
from common import start_button, stop_button

AXIS = ['x', 'y', 'z']
DIRECTION_DICT = {
    "-LEFT-": {'axis': 'x', 'direction': 1},
    "-RIGHT-": {'axis': 'x', 'direction': -1},
    "-AWAY-": {'axis': 'y', 'direction': -1},
    "-TOWARD-": {'axis': 'y', 'direction': 1},
    "-UP-": {'axis': 'z', 'direction': 1},
    "-DOWN-": {'axis': 'z', 'direction': -1},
}


@dataclass
class Movement:
    axis: int = 0
    displacement: int = 0
    direction: int = 2  # Z-axis')

    def move(self, arm):
        # Start position
        goal = arm.setpoint_cp()
        # Move 5cm in z direction
        goal.p[self.axis] += self.direction * self.displacement * 10 ** -3
        arm.move_cp(goal)
        return arm


def publish_current_setpoint_cp(arm, publisher):
    init_frame = arm.setpoint_cp()
    # Convert it to a ROS message
    pose = PoseStamped()
    pose.header.frame_id = 'PSM1_psm_base_link'
    pose.header.stamp = rospy.Time.now()
    pose.pose = posemath.toMsg(init_frame)

    publisher.publish(pose)


def generate_layout():
    arrow_center_column = [
        [sg.Button(key="-UP-",
                   button_color=(sg.theme_background_color(), sg.theme_background_color()),
                   image_filename='/home/dvrk/catkin_ws/src/case_study/scripts/dvrk_end/icons/up_arrow.png', image_size=(50, 50), image_subsample=10,
                   border_width=0)],
        [sg.Button(key="-LEFT-",
                   button_color=(sg.theme_background_color(), sg.theme_background_color()),
                   image_filename='/home/dvrk/catkin_ws/src/case_study/scripts/dvrk_end/icons/left_arrow.png', image_size=(50, 50), image_subsample=10,
                   border_width=0),
         sg.Button(key="-RIGHT-",
                   button_color=(sg.theme_background_color(), sg.theme_background_color()),
                   image_filename='/home/dvrk/catkin_ws/src/case_study/scripts/dvrk_end/icons/right_arrow.png', image_size=(50, 50), image_subsample=10,
                   border_width=0)],
        [sg.Button(key="-DOWN-",
                   button_color=(sg.theme_background_color(), sg.theme_background_color()),
                   image_filename='/home/dvrk/catkin_ws/src/case_study/scripts/dvrk_end/icons/down_arrow.png', image_size=(50, 50), image_subsample=10,
                   border_width=0)]]

    arrow_right_column = [
        [sg.Button(key="-TOWARD-",
                   button_color=(sg.theme_background_color(), sg.theme_background_color()),
                   image_filename='/home/dvrk/catkin_ws/src/case_study/scripts/dvrk_end/icons/toward_arrow.png', image_size=(60, 60), image_subsample=10,
                   border_width=0),
         sg.Button(key="-AWAY-",
                   button_color=(sg.theme_background_color(), sg.theme_background_color()),
                   image_filename='/home/dvrk/catkin_ws/src/case_study/scripts/dvrk_end/icons/backward_arrow.png', image_size=(60, 60), image_subsample=10,
                   border_width=0)]
    ]
    left_column = sg.Column([[sg.Text("Displacement :")],
                             [sg.Text("Direction :")],
                             [sg.Text("Axis :")],
                             [sg.Text("Number of poses :")]],
                            element_justification='left')

    right_column = sg.Column([[sg.Input(key="-MOVE-", size=(10, 1), enable_events=True), sg.Text(" mm")],
                              [sg.Text("+", key='-DISPLAY_DIR-')],
                              [sg.Radio('X', "RADIO1", default=False, key='x', enable_events=True),
                               sg.Radio('Y', "RADIO1", default=False, key='y', enable_events=True),
                               sg.Radio('Z', "RADIO1", default=True, key='z', enable_events=True)],
                              [sg.Radio('Unique', "RADIO2", default=False, key='-UNIQUE-', enable_events=True),
                               sg.Radio('Trajectory', "RADIO2", default=True, key='-MULTIPLE-', enable_events=True)]
                              ],
                             element_justification='left')

    layout = [
        [left_column, right_column],
        [sg.Column(arrow_center_column, element_justification='c'),
         sg.Column(arrow_right_column, element_justification='c')]
        ,
        [start_button, sg.Button('continue'), stop_button]
    ]
    return layout


def update_direction(direction):
    return '+' if direction > 0 else '-'


def main():
    # Generate the window layout
    layout = generate_layout()
    # Create the window
    window = sg.Window('Move PSM1', layout, location=(100, 300), finalize=True, element_justification='center')

    movement = Movement()
    psm1 = Arm(init_arm=False)
    # start_recording_publisher = rospy.Publisher('/case_study/start_recording', String, queue_size=10)
    start_setpoint_publisher = rospy.Publisher('/case_study/current_setpoint_cp', PoseStamped, queue_size=10)

    psm1.publish_trajectory()

    while True:
        event, values = window.read(timeout=10)

        if event in (sg.WIN_CLOSED, 'Exit'):
            psm1.trajectory_thread.stop()
            break

        if event in AXIS:
            movement.axis = AXIS.index(event)

        if event == '-MOVE-':
            try:
                displacement_input = int(values['-MOVE-'])
                movement.displacement = abs(displacement_input)
            except ValueError:
                movement.displacement = 0
            if movement.displacement > 150:
                print('Setting safe check, max displacement 15cm')
                movement.displacement = 150

        if event in DIRECTION_DICT.keys():
            movement.axis = AXIS.index(DIRECTION_DICT.get(event).get('axis'))
            movement.direction = DIRECTION_DICT.get(event).get('direction')

        if event == '-START-':
            if values['-UNIQUE-']:
                movement.move(psm1.arm)
            else:
                psm1.trajectory_thread.move = True
                psm1.create_poses_from_trajectory()
                # start_recording_publisher.publish('Start')
                psm1.follow_trajectory()
                # start_recording_publisher.publish('Stop')

        if event == 'continue':
            psm1.trajectory_thread.move = True

        if event in AXIS + list(DIRECTION_DICT.keys()) + ['-MOVE-']:
            window.Element(AXIS[movement.axis]).Update(value=True)
            window.Element('-DISPLAY_DIR-').Update(value=update_direction(movement.direction))

            # Compute the new path
            psm1.update_trajectory(movement.axis, movement.direction, values)
            # Publish to ROS
            psm1.publish_trajectory()
            # Display the current setpoint just because
            publish_current_setpoint_cp(psm1.arm, start_setpoint_publisher)

        if event == '-STOP-':
            psm1.trajectory_thread.stop()

    psm1.trajectory_thread.stop()


if __name__ == '__main__':
    main()