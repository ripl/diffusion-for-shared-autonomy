#!/usr/bin/env python3
"""Adopted from https://github.com/cbschaff/rsa/blob/master/lunar_lander/joystick_agent.py"""

import time
import pygame
import numpy as np
from diffusha.actor import Actor

#####################################
# Change these to match your joystick
UP_AXIS = 3  # AKA ；up(negative) and down(positive)
SIDE_AXIS = 2  # AKA ；left and right
#####################################


class LunarLanderJoystickActor(Actor):
    """Joystick Controller for Lunar Lander."""

    def __init__(self, env, fps=50):
        """Init."""
        self.env = env
        self.human_agent_action = np.array([0., 0.], dtype=np.float32)
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x)
                     for x in range(pygame.joystick.get_count())]
        # print(joysticks)
        # if len(joysticks) != 1:
        #     raise ValueError("There must be exactly 1 joystick connected."
        #                      f"Found {len(joysticks)}")
        self.joy = joysticks[-1]  # TEMP
        self.joy.init()
        pygame.init()
        self.t = None
        self.fps = fps

    def _get_human_action(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == UP_AXIS:
                    # for up and down, right handel
                    self.human_agent_action[0] = -1 * event.value
                elif event.axis == SIDE_AXIS:
                    # for left and right
                    self.human_agent_action[1] = event.value
        if abs(self.human_agent_action[1]) < 0.1:
            self.human_agent_action[0] = 0.0
        return self.human_agent_action

    def act(self, ob):
        """Act."""
        # self.env.render()
        action = self._get_human_action()
        return action

    def reset(self):
        self.human_agent_action[:] = 0.


if __name__ == '__main__':
    from diffusha.data_collection.env import make_env

    env = make_env(
        "LunarLander-v3",
        seed=1,
        test=False
    )

    actor = LunarLanderJoystickActor(env)

    for _ in range(10):
        ob = env.reset()
        env.render()
        done = False
        reward = 0.0

        while not done:
            env.render()
            ob, r, done, _ = env.step(actor.act(ob))
            reward += r
        print(reward)
