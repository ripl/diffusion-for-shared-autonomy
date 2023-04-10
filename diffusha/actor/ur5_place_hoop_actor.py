#!/usr/bin/env python3

"""
This script provides a scripted policy to achieve the following task
The task: Place a hoop in one of the sticks (there are two sticks in the environment)
Action space: Twist (linear and angular velocity)


The scripted policy:
- If under the height of a stick, go straight up to `stick_height + margin`
- If above the height of a stick, go stragit to `preplace` location
  - Preplace location: `stick x, y loc` and `stick_height + margin` height
- If at the epsilon ball within `preplace coord`, go straight down
- If at the epsilon ball within the goal, ring a bell

NOTE: All the coordinates and action values are **not** normalized in this script.
This should make it easier to adjust values.
"""

import numpy as np
from diffusha.actor import Actor

stick_height = 0.12
max_vel = 0.2
eps = 0.02

# NOTE: original coordinates
# goal locations: (0.213, 0.42), (0.426, 0.42)
# limits: (0,0) -> (0.64, 0.84)

# NOTE: normalized obs space
# goal locations: (0.333, 0.5, 0.2), (0.666, 0.5, 0.2)
# limits: (-0.5, -0.5, -0.5) -> (0.5, 0.5, 0.5)

class UR5PlaceHoopController(Actor):
    def __init__(self, obs_space, act_space) -> None:
        self.obs_space = obs_space
        self.act_space = act_space

        self._goal_pos = np.array([0.213, 0.42, 0.12])
        self._curr_vel = None

    def set_goal(self, goal_pos):
        self._goal_pos = goal_pos

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Args:
            - obs: (x, y, z) position of the gripper tip

        Returns:
            - vel (vx, vy, vz)
        """

        # TODO: Normalize the height so that z=0 is the table

        curr_pos = obs
        x, y, z = curr_pos

        if np.linalg.norm(curr_pos - self._goal_pos) < eps:
            print('Success!!')
            vx, vy, vz = 0, 0, 0

        elif z <= stick_height:
            # If the gripper is below the height of sticks, go straight up
            vx, vy, vz = 0, 0, 0.2

        elif np.linalg.norm(curr_pos[:2] - self.preplace_pos[:2]) <= eps:
            # The gripper is right above the goal, let's just move down.
            vx, vy, vz = 0, 0, 0.08
        else:
            # Move toward the preplace pos
            vx, vy, vz = self.get_velocity(curr_pos, self.preplace_pos)

        vel = np.array([vx, vy, vz])
        self._curr_vel = vel
        return vel

    def get_velocity(self, curr_pos, target_pos, p_gain=10.0, d_gain=-1.0):
        """PD controller to reach the target pos."""
        # Compute control
        prop = target_pos - curr_pos
        action = p_gain * prop + d_gain * self._curr_vel

        action = np.clip(action, -max_vel, max_vel)
        return action

    @property
    def preplace_pos(self):
        preplace_margin = 0.05
        x, y, z = self._goal_pos
        return np.array([x, y, z + preplace_margin])

    def get_action(self, location, velocity, target):
        """Copied from d4rl pointmaze waypoint controller"""
        if np.linalg.norm(self._target - np.array(self.gridify_state(target))) > 1e-3:
            #print('New target!', target, 'old:', self._target)
            self._new_target(location, target)

        dist = np.linalg.norm(location - self._target)
        vel = self._waypoint_prev_loc - location
        vel_norm = np.linalg.norm(vel)
        task_not_solved = (dist >= self.solve_thresh) or (vel_norm >= self.vel_thresh)

        if task_not_solved:
            next_wpnt = self._waypoints[self._waypoint_idx]
        else:
            next_wpnt = self._target

        # Compute control
        prop = next_wpnt - location
        action = self.p_gain * prop + self.d_gain * velocity

        dist_next_wpnt = np.linalg.norm(location - next_wpnt)
        if task_not_solved and (dist_next_wpnt < self.solve_thresh) and (vel_norm<self.vel_thresh):
            self._waypoint_idx += 1
            if self._waypoint_idx == len(self._waypoints)-1:
                assert np.linalg.norm(self._waypoints[self._waypoint_idx] - self._target) <= self.solve_thresh

        self._waypoint_prev_loc = location
        action = np.clip(action, -1.0, 1.0)
        return action, (not task_not_solved)
