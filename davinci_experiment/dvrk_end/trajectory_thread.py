import threading
import time
from threading import Thread

import rospy
from tf_conversions import posemath
from std_msgs.msg import Int16


class TrajectoryThread(threading.Thread):
    """
    Class that wait to be stopped
    """

    def __init__(self, *args, **kwargs):
        super(TrajectoryThread, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self.dvrk_status_publisher = rospy.Publisher('/case_study/dvrk_status', Int16, queue_size=10)
        self.move = False

    def start_thread(self, arm, poses):
        self._stop = threading.Event()
        Thread(target=self.follow_trajectory, args=(arm, poses)).start()
        return self

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def follow_trajectory(self, arm, poses):
        # Move the dVRK and update status step by step
        for i_pose in range(len(poses)):

            # If the user did not stop the dVRK keep on moving the arm
            if not self.stopped():
                # if i_pose % 5 == 0:
                #     print('Continue?')
                #     self.move = False
                #
                # while not self.move:
                #     time.sleep(0.1)
                #     if self.stopped():
                #         break
                #
                # if not self.stopped():
                arm.move_cp(posemath.fromMsg(poses[i_pose].pose)).wait()
                # Publish the status of the dVRK
                status_msg = Int16(i_pose)
                self.dvrk_status_publisher.publish(status_msg)
            else:
                break
