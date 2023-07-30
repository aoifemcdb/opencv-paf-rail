#!/usr/bin/env python3
import sys

import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension

from common import str2array


def msg_layout():
    dim_vect = MultiArrayDimension('vect', 25, 75)
    dim_xyz = MultiArrayDimension('xyz', 3, 3)
    dim = [dim_vect, dim_xyz]
    return MultiArrayLayout(dim=dim)


def trajectory_publisher(trajectory_str: str):
    print('Printing dVRK trajectory')
    print(trajectory_str)

    pub = rospy.Publisher('/case_study/dvrk_trajectory', Float32MultiArray, queue_size=10)
    rospy.init_node('trajectory_publisher')
    r = rospy.Rate(0.5)

    trajectory_msg = Float32MultiArray()
    trajectory_msg.layout = msg_layout()
    trajectory_msg.data = str2array(trajectory_str)

    while not rospy.is_shutdown():
        pub.publish(trajectory_msg)
        r.sleep()
    print("Rospy shutdown")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: trajectory_publisher.py trajectory_array")
    else:
        try:
            trajectory_publisher(sys.argv[1])
        except rospy.ROSInterruptException:
            pass