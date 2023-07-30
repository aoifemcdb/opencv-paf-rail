#!/usr/bin/env python3

import rospy
from std_msgs.msg import Char

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('keypress_publisher')

    # Create a publisher for the 'keypress' topic with message type Char
    pub = rospy.Publisher('keypress', Char, queue_size=1)

    # Wait for a short duration to allow the publisher to initialize
    rospy.sleep(1)

    # Publish 'r' to the 'keypress' topic
    pub.publish('r')

    # Sleep for a short duration to allow the message to be sent
    rospy.sleep(1)
