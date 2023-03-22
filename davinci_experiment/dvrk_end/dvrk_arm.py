#!/usr/bin/env python3
import time

import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R

import PyKDL
import rospy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf_conversions import posemath
import dvrk

from trajectory_thread import TrajectoryThread

ORI = np.resize(R.from_quat([0.7983, 0.59416, 0.09651, -0.0173]).as_matrix(), 9)
POSE_INI = PyKDL.Frame(PyKDL.Rotation(ORI[0], ORI[1], ORI[2], ORI[3], ORI[4], ORI[5], ORI[6], ORI[7], ORI[8]),
                       PyKDL.Vector(-0.0212, 0.0936703, -0.08433))
TRAJECTORY = np.round([[0, 0, 0.01 * i] for i in range(10)], 3)
nb_positions = 100


@dataclass
class Arm:
    name: str = 'PSM1'
    arm: dvrk.arm = dvrk.arm(name)
    init_arm: bool = True
    current_frame: PyKDL.Frame = PyKDL.Frame()
    trajectory: np.ndarray = TRAJECTORY
    path_pub: rospy.Publisher = field(init=False)
    poses: list = None
    trajectory_thread: TrajectoryThread = field(init=False)

    def __post_init__(self):
        print('Init arm')
        self.arm.enable()
        print('Homing')
        self.arm.home()

        # Go to init pause if required
        if self.init_arm:
            # Going to initial pose
            print('Going to initial pose ...')
            self.go_to_init_pose()

        # Get the current frame
        print('Getting the current frame')
        self.current_frame = self.arm.setpoint_cp()

        # Publishers
        self.path_pub = rospy.Publisher('/case_study/trajectory_path', Path, queue_size=10)

        # Initializing the trajectory thread
        self.trajectory_thread = TrajectoryThread()

    def go_to_init_pose(self):
        print("PSM1 - Set to initial position")
        # Move to the initial position
        goal = POSE_INI
        # Rotate tool tip frame by 45 degrees
        goal.M.DoRotX(np.pi * 0.5)
        self.arm.move_cp(goal).wait()

    def update_current_frame(self):
        # Get the current orientation
        try:
            self.current_frame = self.arm.setpoint_cp()
        except RuntimeWarning:
            print("Could not get the current cartesian position in time")
        return

    def create_poses_from_trajectory(self):
        poses = []

        self.update_current_frame()
        init_frame = self.current_frame

        for p in self.trajectory:
            new_vect = PyKDL.Vector(init_frame.p[0] + p[0], init_frame.p[1] + p[1], init_frame.p[2] + p[2])
            new_frame = PyKDL.Frame(init_frame.M, new_vect)

            # Convert it to a ROS message
            new_pose = PoseStamped()
            new_pose.header.frame_id = 'PSM1_psm_base_link'
            new_pose.header.stamp = rospy.Time.now()
            new_pose.pose = posemath.toMsg(new_frame)

            # Append at the end
            poses.append(new_pose)

        self.poses = poses
        return poses

    def update_trajectory(self, axis, orientation, values):
        # Get the current frame
        self.update_current_frame()

        # Set the trajectory to follow a line
        try:
            displacement_mm = int(values['-MOVE-'])
        except ValueError:
            displacement_mm = 10
        nb_poses = displacement_mm  # mm
        self.trajectory = np.zeros((nb_poses, 3))
        trajectory_along_axis = np.round([0.001 * i * orientation for i in range(nb_poses)], 3)
        self.trajectory[:, axis] = trajectory_along_axis

        self.create_poses_from_trajectory()

    def publish_trajectory(self):
        # Initialize a ROS Path message
        path = Path()
        path.header.frame_id = 'PSM1_psm_base_link'
        path.header.stamp = rospy.Time.now()

        if self.poses:
            path.poses = self.poses
        else:
            path.poses = self.create_poses_from_trajectory()

        # Publish the path to the dVRK
        self.path_pub.publish(path)

        return path.poses

    def follow_trajectory(self):
        self.trajectory_thread.start_thread(self.arm, self.poses)