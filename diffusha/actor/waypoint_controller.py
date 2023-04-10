#!/usr/bin/env python3
"""D4RL Waypoint Controller actor"""
import numpy as np

from .base import Actor
from d4rl.pointmaze.waypoint_controller import WaypointController

class WaypointActor(Actor):
    def __init__(self, obs_space, act_space, env) -> None:
        super().__init__(obs_space, act_space)
        self.controller = WaypointController(env.str_maze_spec)
        self._env = env

    def act(self, obs):
        position = obs[0:2]
        velocity = obs[2:4]
        act, done = self.controller.get_action(position, velocity, self._env.unwrapped._target)

        act= act.astype(np.float32)
        return act
