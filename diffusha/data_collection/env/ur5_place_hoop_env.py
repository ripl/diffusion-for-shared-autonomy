#!/usr/bin/env python3

import numpy as np
import time
import roslibpy


def build_twist_msg(linear_vel, angular_vel):
    x, y, z = linear_vel
    ax, ay, az = angular_vel
    msg = roslibpy.Message({'linear': {'x': x, 'y': y, 'z': z},
                            'angular': {'x': ax, 'y': ay, 'z': az}})
    return msg


def unpack_pos(rosmsg_pos):
    """Convert geometry_msgs/Point to a list"""
    x, y, z = rosmsg_pos['x'], rosmsg_pos['y'], rosmsg_pos['z']
    return [x, y, z]


def unpack_quat(rosmsg_quat):
    """Convert geometry_msgs/Quaternion to a list"""
    x, y, z, w = rosmsg_quat['x'], rosmsg_quat['y'], rosmsg_quat['z'], rosmsg_quat['w']
    return [x, y, z, w]


class UR5PlaceHoopEnv:
    _tf_request = roslibpy.ServiceRequest(
            values={
                'source_frame': {'data': '/hand_finger_tip_link'},
                'target_frame': {'data': '/ur_arm_base'}
            }
        )
    _goals = [np.array((0.213, 0.42, 0.12)), np.array((0.426, 0.42, 0.12))]
    _eps = 0.02

    def __init__(self):
        self.client = roslibpy.Ros(host='localhost', port=9090)
        self.client.run()
        print(f'connecting to rosbridge server...done -- is_connected: {self.client.is_connected}')

        self.command_tp = roslibpy.Topic(self.client, '/twist_control/command', 'geometry_msgs/Twist')
        self.service_reset = roslibpy.Service(self.client, '/twist_control/reset', 'ur5_twist_control/Reset')

        self._tip_pose = None
        self.tf_subscriber = roslibpy.tf.TFClient(self.client, fixed_frame='/ur_arm_base')
        self.tf_subscriber.subscribe('/hand_finger_tip_link', self._save_tip_pose)

    def _save_tip_pose(self, message: dict):
        print('saving tip pose...')
        trans = unpack_pos(message['translation'])
        rot = unpack_quat(message['rotation'])
        self._tip_pose = np.array([*trans, *rot])

    # def get_tip_pose(self):
    #     # get pose for the realsense camera

    #     started = time.perf_counter()
    #     response = self.service_tf.call(self._tf_request)
    #     trans = unpack_pos(response['transform']['position'])
    #     rot = unpack_quat(response['transform']['orientation'])
    #     # trans, rot = response['pose']['position'], response['pose']['orientation']
    #     pose = np.array([*trans, *rot])

    #     elapsed = time.perf_counter() - started
    #     print(f'lookup transform took {elapsed:.2f} sec')

    #     return pose

    def reset(self):
        print('Resetting the environment...')
        self.service_reset.call()

        # Get the position
        pose = self.get_tip_pose()
        curr_pos = pose[:3]  # Only need x, y, z

        return curr_pos

    def step(self, action: np.ndarray):
        # Move the arm
        linear_vel = action
        angular_vel = np.array([0, 0, 0.])
        self.command_tp.publish(build_twist_msg(linear_vel, angular_vel))

        # Get the position
        pose = self._tip_pose
        curr_pos = pose[:3]  # Only need x, y, z

        reward = 0.
        done = False
        info = {}
        for goal in self._goals:
            if np.linalg.norm(curr_pos - goal) < self._eps:
                # Goal is reached!
                print(f'goal: {goal} is reached!')
                reward = 1.0
                done = True
                break

        return curr_pos, reward, done, info


if __name__ == '__main__':
    env = UR5PlaceHoopEnv()
    env.reset()

    small_vel = np.array([0.1, 0, 0])
    for i in range(10):
        obs, rew, done, info = env.step(small_vel)
        print('obs', obs)
