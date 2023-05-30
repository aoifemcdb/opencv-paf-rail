#!/usr/bin/env python

import numpy as np
# # import rospy
# from geometry_msgs.msg import PointStamped, PoseStamped
# from nav_msgs.msg import Path
from colour_shape_sensing.approximate_spline import *

def generate_trajectory(coords):
    # Generate a 2D trajectory from the input coordinates
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Fit a cubic spline to the x and y coordinates separately
    x_spline = UnivariateSpline(np.arange(len(x_coords)), x_coords, k=3, s=0)
    y_spline = UnivariateSpline(np.arange(len(y_coords)), y_coords, k=3, s=0)

    # Generate a new set of x and y coordinates at 1cm intervals
    t_new = np.arange(0, len(x_coords) - 1, 0.01)
    x_new = x_spline(t_new)
    y_new = y_spline(t_new)

    # Combine the x and y coordinates into a single array
    traj = np.column_stack((x_new, y_new))

    return traj


def publish_trajectory(traj):
    # Initialize a new ROS node
    rospy.init_node('trajectory_publisher')

    # Create a ROS publisher for the trajectory
    traj_pub = rospy.Publisher('/trajectory', Path, queue_size=10)

    # Define the frame ID for the trajectory
    frame_id = 'map'

    # Create a new ROS message for the trajectory
    traj_msg = Path()
    traj_msg.header.frame_id = frame_id

    # Add the trajectory points to the message
    for point in traj:
        pose = PoseStamped()
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = 0.0
        pose.header.frame_id = frame_id
        pose.header.stamp = rospy.Time.now()
        traj_msg.poses.append(pose)

    # Publish the trajectory message
    traj_pub.publish(traj_msg)

def process_image_traj(filepath):
    file_path = ''
    color_name = 'red'
    lower_color, upper_color = get_color_boundaries(color_name)
    img, x_new, y_new_smooth = process_image(file_path, lower_color, upper_color)
    return x_new, y_new_smooth


if __name__ == '__main__':
    # Define the input coordinates
    # coords = np.array([[0, 0], [1, 1], [2, 1], [3, 2]])
    process_image_traj()
    coords = get_spline_curve(x_new, y_new_smooth, 10)
    # Generate the trajectory
    traj = generate_trajectory(coords)

    # Publish the trajectory as a ROS message
    # publish_trajectory(traj)
