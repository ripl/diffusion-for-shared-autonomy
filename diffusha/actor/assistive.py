#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import torch

from .base import Actor
from typing import Callable
from functools import partial
device='cuda'

# Magic to avoid circular import due to type hint annotation
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from diffusha.diffusion.ddpm import DiffusionModel


class DiffusionAssistedActor(Actor):
    def __init__(self, obs_space, act_space, diffusion: DiffusionModel, behavioral_actor: Actor = None, fwd_diff_ratio: float = 0.45) -> None:
        super().__init__(obs_space, act_space)
        self.diffusion = diffusion
        self.behavioral_actor = behavioral_actor
        self.fwd_diff_ratio = fwd_diff_ratio

        # self.obs_size = obs_space.low.size
        self.act_size = act_space.low.size

        assert 0 <= fwd_diff_ratio <= 1
        self._k = int((self.diffusion.num_diffusion_steps - 1) * self.fwd_diff_ratio)
        print(f'forward diffusion steps for action: {self._k} / {self.diffusion.num_diffusion_steps}')

    def _diffusion_cond_sample(self, obs, user_act, run_in_batch=False):
        """Conditional sampling"""

        if user_act is None:
            user_act = torch.randn((self.act_size,))

        # HACK
        if not run_in_batch:
            obs_size = obs.size
        else:
            obs_size = obs.shape[1]

        # import pdb; pdb.set_trace()

        # Concat obs and user action
        if torch.is_tensor(obs):
            state = torch.cat((obs, user_act), axis=1)
        else:
            # Luzhe: TEMP!
            # This if else condition is specific for play.py I am not sure whether this would cause a problem for eval
            state = torch.as_tensor(np.concatenate((obs, user_act), axis=0))

        # NOTE: Currently only support hard conditioning (replacing a part of the input / output)

        # Forward diffuse user_act for k steps
        if not run_in_batch:
            x_k, e = self.diffusion.diffuse(state.unsqueeze(0), torch.as_tensor([self._k]))
        else:
            x_k, e = self.diffusion.diffuse(state, torch.as_tensor([self._k]))

        # Reverse diffuse Tensor([*crisp_obs, *noisy_user_act]) for (diffusion.num_diffusion_steps - k) steps
        obs = torch.as_tensor(obs, dtype=torch.float32)
        x_k[:, :obs_size] = obs  # Add condition
        x_i = x_k
        for i in reversed(range(self._k)):
            x_i = self.diffusion.p_sample(x_i, i)
            x_i[:, :obs_size] = obs  # Add condition

        if not run_in_batch:
            out = x_i.squeeze()  # Remove batch dim
            return out[obs_size:].cpu().numpy()
        else:
            out = x_i
            return out[..., obs_size:].cpu().numpy()


    def act(self, obs: np.ndarray, report_diff: bool = False, return_original: bool = False):
        if isinstance(obs, dict):
            obs_pilot = obs['pilot']
            obs_copilot = obs['copilot']
        else:
            obs_pilot = obs_copilot = obs

        # Get user input
        # import pdb; pdb.set_trace()
        user_act = self.behavioral_actor.act(obs_pilot)
        # print('user act', user_act)

        # action = user_act
        if self.fwd_diff_ratio != 0:
            action = self._diffusion_cond_sample(obs_copilot, user_act)
        else:
            action = user_act

        if return_original:
            return action, user_act

        if report_diff:
            diff = np.linalg.norm(user_act - action)
            return action, diff
        else:
            return action

    def batch_act(self, obss: np.ndarray, report_diff: bool = False, return_original: bool = False):
        user_actions = []
        obs_copilots = []
        obs_pilots = []
        for obs in obss:
            if isinstance(obs, dict):
                obs_pilot = obs['pilot']
                obs_copilot = obs['copilot']
            else:
                obs_pilot = obs_copilot = obs
            obs_copilots.append(obs_copilot)
            obs_pilots.append(obs_pilot)

            # Get user input
            # import pdb; pdb.set_trace()
            # user_act = self.behavioral_actor.act(obs_pilot)
            # user_actions.append(user_act)
        user_actions = self.behavioral_actor.batch_act(obs_pilots)
        # print('user act', user_act)


        # action = user_act
        if self.fwd_diff_ratio != 0:
            user_actions = torch.as_tensor(np.array(user_actions))
            obs_copilots = torch.as_tensor(np.array(obs_copilots))
            actions = self._diffusion_cond_sample(obs_copilots, user_actions, run_in_batch=True)
        else:
            actions = user_actions

        if return_original:
            return actions, user_actions

        if report_diff:
            # TEMP: LUZHE suppose they are in shape (n * (action_dim+obs_dim)) , maybe flipped, take care about axis
            diffs = np.linalg.norm(np.array(user_actions) - np.array(actions), axis=1)
            return actions, diffs
        else:
            return actions

    def act_without_env(self, obs: np.ndarray, act: np.ndarray, report_diff: bool = False):
        if isinstance(obs, dict):
            obs_pilot = obs['pilot']
            obs_copilot = obs['copilot']
        else:
            obs_pilot = obs_copilot = obs
        # Get user input
        user_act = act

        if self.fwd_diff_ratio != 0:
            action = self._diffusion_cond_sample(obs_copilot, user_act)
        else:
            action = user_act

        if report_diff:
            diff = np.linalg.norm(user_act - action)
            return action, diff
        else:
            return action




