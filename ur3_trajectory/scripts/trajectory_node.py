#!/usr/bin/env python3

import rospy
import numpy as np
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import Pose
from std_msgs.msg import Char
import csv

def key_callback(msg):
    global key_pressed
    key_pressed = msg.data

# Function to adjust the trajectory data based on the current robot pose
def adjust_trajectory(coordinates, current_pose):
    adjusted_coordinates = coordinates.copy()
    adjusted_coordinates[:, 0] += current_pose.position.x
    adjusted_coordinates[:, 1] += current_pose.position.y
    return adjusted_coordinates

if __name__ == '__main__':
    rospy.init_node('trajectory_node')

    # Create a ROS publisher for the /robot_pose topic
    pose_publisher = rospy.Publisher('/robot_pose', Pose, queue_size=10)

    # Initialize the move group
    move_group = MoveGroupCommander("manipulator")

    # Get the current robot state
    current_pose = move_group.get_current_pose().pose

    # Load trajectory data from the CSV file
    with open('data.csv', 'r') as file:
        reader = csv.reader(file)
        coordinates = np.array(list(reader), dtype=np.float32)
    adjusted_coordinates = adjust_trajectory(coordinates, current_pose)

    # Execute the adjusted trajectory
    for point in adjusted_coordinates:
        target_pose = Pose()
        target_pose.position.x = float(point[0])
        target_pose.position.y = float(point[1])
        target_pose.position.z = current_pose.position.z  # Assuming z remains the same
        target_pose.orientation = current_pose.orientation

        # Move the robot to the target pose
        move_group.set_pose_target(target_pose)
        plan = move_group.go(wait=True)

        # Get the robot's current pose
        current_pose = move_group.get_current_pose().pose

        # Publish the current pose to the /robot_pose topic
        pose_publisher.publish(current_pose)

        # Sleep to control the loop rate
        rospy.sleep(0.1)

    # Once the trajectory is executed, return to the original position
    move_group.set_pose_target(current_pose)
    move_group.go(wait=True)

    rospy.signal_shutdown("Finished executing trajectory.")






