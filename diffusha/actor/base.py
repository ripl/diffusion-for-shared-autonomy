#!/usr/bin/env python3
"""Partially adopted from https://github.com/cbschaff/rsa"""

from typing import List
import numpy as np
import pfrl
from diffusha.config.default_args import Args


def choose_obs_if_necessary(obs, actor='pilot'):
    assert actor in ['pilot', 'copilot']

    if isinstance(obs, dict):
        return obs[actor]
    # elif isinstance(obs, list):
    #     # Assuming VectorEnv
    #     return [_obs[actor] for _obs in obs]
    else:
        return obs




class Actor:
    def __init__(self, obs_space, act_space) -> None:
        self.obs_space = obs_space
        self.act_space = act_space

    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return NotImplemented

    def batch_act(self, obss: np.ndarray) -> List[np.ndarray]:
        actions = [self.act(obs) for obs in obss]
        return actions

    def random_action(self, generator: np.random.Generator = None):
        if generator:
            return generator.uniform(self.act_space.low, self.act_space.high, size=self.act_space.low.size)
        else:
            return np.random.uniform(self.act_space.low, self.act_space.high, size=self.act_space.low.size)


class ZeroActor(Actor):
    """Output random actions."""
    def __init__(self, obs_space, act_space):
        """Init."""
        super().__init__(obs_space, act_space)

    def act(self, ob):
        """Act."""
        zero_act = self.act_space.low * 0
        return zero_act


class RandomActor(Actor):
    """Output random actions."""
    def __init__(self, obs_space, act_space, seed: int = 0):
        """Init."""
        super().__init__(obs_space, act_space)
        self.np_random = np.random.default_rng(seed=seed)

    def act(self, ob):
        """Act."""
        return self.random_action(generator=self.np_random)


class ExpertActor(Actor):
    """output expert actions."""

    def __init__(self, obs_space, act_space, expert_agent):
        super().__init__(obs_space, act_space)
        self.agent = expert_agent
        if isinstance(expert_agent, pfrl.agents.SoftActorCritic):
            self.agent.training = False

    def act(self, ob):
        """Act."""
        ob = choose_obs_if_necessary(ob)
        return self.agent.act(ob)


class LaggyActor(Actor):
    """Laggy actor"""
    def __init__(self, obs_space, act_space, actor: Actor, repeat_prob: float, seed: int = 0):
        super().__init__(obs_space, act_space)
        self.actor = actor
        self.repeat_prob = repeat_prob
        self.actions = None
        self.np_random = np.random.default_rng(seed=seed)

        # TODO: Maze needs to maintain self.repeat (previously we repeated the action for 5 times)

    def act(self, ob, index=-1):
        """Act."""
        if index == -1:
            self.maybe_init_actions(num_envs=1)

        ob = choose_obs_if_necessary(ob)
        if self.np_random.random() < self.repeat_prob:
            return self.actions[index]
        else:
            self.actions[index] = self.actor.act(ob)
            return self.actions[index]

    def maybe_init_actions(self, num_envs):
        if self.actions is None:
            self.actions = [self.random_action(generator=self.np_random) for _ in range(num_envs)]

    # need to overwrite base class
    def batch_act(self, obss):
        self.maybe_init_actions(len(obss))
        actions = [self.act(obss[index], index) for index in range(len(obss))]
        return actions


class NoisyActor(Actor):
    """Noisy actor"""
    def __init__(self, obs_space, act_space, actor: Actor, eps: float, preserve_norm: bool = False, seed: int = 0):
        super().__init__(obs_space, act_space)
        self.actor = actor
        self.eps = eps
        self._preserve_norm = preserve_norm
        self.repeat = 0
        self.np_random = np.random.default_rng(seed=seed)
        self.action = self.random_action(generator=self.np_random)

    def act(self, ob):
        """Act."""
        ob = choose_obs_if_necessary(ob)
        if self.repeat:
            self.repeat -= 1
            return self.action
        elif self.np_random.random() < self.eps:
        # elif np.random.rand() < self.eps:
            if 'maze' in Args.env_name or 'Maze' in Args.env_name:
                self.repeat = 4
            else:
                self.repeat = 0
            self.action = self.get_random(self.action)
            return self.action
        else:
            return self.actor.act(ob)

    def get_random(self, action):
        if self._preserve_norm:
            action0 = self.np_random.uniform(0.9, 1)
            action1 = self.np_random.uniform(0.9, 1)
            if self.np_random.random() < 0.5:
                action0 = -action0
            if self.np_random.random() < 0.5:
                action1 = -action1
            rand_action = np.array([action0, action1], dtype=action.dtype)

            """
            # theta = np.random.rand() * 2 * np.pi
            action0 = np.random.uniform(0.9, 1)
            action1 = np.random.uniform(0.9, 1)
            if np.random.rand() < 0.5:
                action0 = -action0
            if np.random.rand() < 0.5:
                action1 = -action1
            rand_action = np.array([action0, action1], dtype=action.dtype)
            # action = 0.0001 * np.array([np.cos(theta), np.sin(theta)], dtype=action.dtype)
            # print("use norm")
            # action = np.linalg.norm(action) * np.array([np.cos(theta), np.sin(theta)], dtype=action.dtype)
            """
            return rand_action
        else:
            rand_action = self.random_action(generator=self.np_random)
            # rand_action = self.act_space.sample()
            return rand_action
