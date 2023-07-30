#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('get_robot_state_node')

    # Initialize MoveIt Commander
    move_group = MoveGroupCommander("manipulator")

    # Get the current joint values
    current_joint_values = move_group.get_current_joint_values()
    print("Current Joint Values:", current_joint_values)

    # Get the current end-effector (TCP) pose
    current_tcp_pose = move_group.get_current_pose().pose
    print("Current TCP Pose:")
    print("Position (x, y, z):", current_tcp_pose.position.x, current_tcp_pose.position.y, current_tcp_pose.position.z)
    print("Orientation (x, y, z, w):", current_tcp_pose.orientation.x, current_tcp_pose.orientation.y,
          current_tcp_pose.orientation.z, current_tcp_pose.orientation.w)
