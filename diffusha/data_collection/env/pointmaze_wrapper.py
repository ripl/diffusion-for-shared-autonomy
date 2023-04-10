#!/usr/bin/env python3

from gym import Wrapper
from gym.core import ActType, Tuple, ObsType

class PointMazeTerminationWrapper(Wrapper):
    """Simply set done to True when (sparse) reward is non-zero
    This prevents the weird fact that those agents that can stay close to the goal longer gets better rewards.
    And only encourages the agent to reach the goal
    """
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps through the environment with action."""
        obs, rew, done, info = self.env.step(action)

        if rew > 0:
            # Terminate if (sparse) reward is non-zero (i.e., the agent reached the goal)
            done = True

        # Add penatly for each step
        rew = rew - 0.002

        return obs, rew, done, info
