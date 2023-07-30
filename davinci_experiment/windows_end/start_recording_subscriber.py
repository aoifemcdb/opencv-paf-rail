#!/usr/bin/env python3
import sys

import rospy
from std_msgs.msg import String
from dataclasses import dataclass, field

from scripts.thread_streaming import thread_bk_and_webcam


@dataclass
class StartRecordingSubscriber:
    recording_folder: str
    start_recording_sub: rospy.Subscriber = field(init=False)
    start_time: float = 0.0
    stop_time: float = 0.0

    def __post_init__(self):
        self.start_recording_sub = rospy.Subscriber("/case_study/start_recording", String, self.callback)
        rospy.init_node('case_study')

    def callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + " I heard %s", data.data)
        if data.data == 'Start':
            self.start_time = rospy.get_time()
            rospy.loginfo('Starting recording : %s', self.start_time)
            thread_bk_and_webcam(self.recording_folder)
        elif data.data == 'Stop':
            self.start_time = rospy.get_time()
            rospy.loginfo('Stop recording : %s', self.stop_time)
        else:
            rospy.loginfo('Something is wrong :')
        rospy.loginfo('Something is wrong twice :')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Usage: trajectory_publisher.py trajectory_array")
    else:
        print('trajectory publisher ', sys.argv[1])
        start_recording_subscriber = StartRecordingSubscriber(sys.argv[1])

    rospy.spin()