#!/usr/bin/env python3
from gym import Wrapper
from gym.core import ActType, Tuple, ObsType
from math import sqrt

class LunarLanderRewardWrapper(Wrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._counter = 0

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self._counter = 0
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps through the environment with action."""
        obs, rew, done, info = self.env.step(action)
        x, y, vx, vy, rx, ry, rvx, rvy, *_ = obs


        vel = sqrt(vx ** 2 + vy ** 2)
        overspeed_penalty = -5 * max(vel - 0.8, 0.)
        oov_penalty = -5 * max(y - 1.4, 0.)
        alive = 0.1
        # Just 0.1 as long as the agent keeps alive
        new_rew = alive + overspeed_penalty + oov_penalty
        return obs, new_rew, done, info
