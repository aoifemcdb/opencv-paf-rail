#!/usr/bin/env python3

import rospy
import rosbag
import csv
from geometry_msgs.msg import Pose

# Callback function to handle incoming pose messages
def pose_callback(msg):
    global pose_data
    pose_data.append(msg)

if __name__ == '__main__':
    rospy.init_node('pose_subscriber')

    # Initialize the list to store pose data
    pose_data = []

    # Subscribe to the /robot_pose topic
    rospy.Subscriber('/robot_pose', Pose, pose_callback)

    # Run the subscriber for a specified duration (in seconds)
    duration = 30  # Change this to the desired duration for recording data
    rate = rospy.Rate(10)  # Set the rate at which the subscriber will check for new messages

    rospy.loginfo("Recording pose data for {} seconds...".format(duration))

    start_time = rospy.Time.now()

    while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
        rate.sleep()

    # Save pose data to a ROS bag file
    bag_filename = "pose_data.bag"
    with rosbag.Bag(bag_filename, 'w') as bag:
        for pose_msg in pose_data:
            bag.write('/robot_pose', pose_msg, pose_msg.header.stamp)

    rospy.loginfo("Pose data saved to {}.".format(bag_filename))

    # Convert the ROS bag file to CSV
    csv_filename = "pose_data.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['timestamp', 'position_x', 'position_y', 'position_z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])

        for pose_msg in pose_data:
            timestamp = pose_msg.header.stamp.to_sec()
            position_x = pose_msg.position.x
            position_y = pose_msg.position.y
            position_z = pose_msg.position.z
            orientation_x = pose_msg.orientation.x
            orientation_y = pose_msg.orientation.y
            orientation_z = pose_msg.orientation.z
            orientation_w = pose_msg.orientation.w

            csv_writer.writerow([timestamp, position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w])

    rospy.loginfo("Pose data saved to {}.".format(csv_filename))




