#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from dvrk import psm


def execute_trajectory(traj):
    # Initialize a new ROS node
    rospy.init_node('trajectory_executor')

    # Connect to the PSM1 arm
    rospy.loginfo('Connecting to PSM1...')
    arm = psm('PSM1')

    # Enable the arm and move to the starting position
    rospy.loginfo('Enabling the arm and moving to the starting position...')
    arm.home()
    arm.move(arm.JPS)

    # Execute the trajectory using the PSM1 arm
    rospy.loginfo('Executing the trajectory...')
    for pose in traj.poses:
        # Convert the pose to a PSM joint configuration
        joint_angles = arm.pose_to_joint(np.array([pose.pose.position.x,
                                                    pose.pose.position.y,
                                                    pose.pose.position.z,
                                                    pose.pose.orientation.x,
                                                    pose.pose.orientation.y,
                                                    pose.pose.orientation.z,
                                                    pose.pose.orientation.w]))
        # Move the arm to the new joint configuration
        arm.move(joint_angles)


def trajectory_callback(traj):
    # Execute the trajectory when a new message is received on the /trajectory topic
    execute_trajectory(traj)


if __name__ == '__main__':
    # Initialize a new ROS node
    rospy.init_node('trajectory_listener')

    # Subscribe to the /trajectory topic
    rospy.Subscriber('/trajectory', Path, trajectory_callback)

    # Spin the node to wait for messages
    rospy.spin()
