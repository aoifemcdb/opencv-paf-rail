#!/usr/bin/env python3

import rospy
import numpy as np
from moveit_commander import MoveGroupCommander

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('move_robot_node')

    # Initialize MoveIt Commander
    move_group = MoveGroupCommander("manipulator")

    # Get the current end-effector (TCP) pose
    current_tcp_pose = move_group.get_current_pose().pose
    # Increase the planning time (adjust as needed)
    move_group.set_planning_time(10)  # Set the planning time to 10 seconds

    # Set the target end-effector (TCP) pose 1 cm in the x-direction
    target_tcp_pose = current_tcp_pose
    target_tcp_pose.position.x += 0.1  # 1 cm in the x-direction

    # Move the robot to the target pose
    move_group.set_pose_target(target_tcp_pose)
    plan = move_group.go(wait=True)

    # If the plan is successful, print a message, else print an error
    if plan:
        rospy.loginfo("Robot moved successfully!")
    else:
        rospy.logerr("Failed to move the robot to the target pose.")




