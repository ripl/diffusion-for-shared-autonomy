#!/usr/bin/env python3
"""Remove goal info from state"""
from typing import List, Any
import numpy as np
from diffusha.actor.base import Actor
from copy import deepcopy
from gym.core import ActType, Wrapper, ObservationWrapper, Tuple, ObsType
from gym import error, spaces

class LunarLanderSplitObsWrapper(ObservationWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert 'LunarLander' in self.env.unwrapped.spec.name

        self.pilot_observation_space = self.observation_space
        lvl = self.env.unwrapped.spec.name.split('-')[-1]
        if lvl == 'v1':  # Reach
            _state_size = self.observation_space.low.size - 2
            self.copilot_observation_space = spaces.Box(
                -np.inf, np.inf, shape=(_state_size,), dtype=np.float32
            )
        elif lvl == 'v5':  # Land with randomlized helipad
            _state_size = self.observation_space.low.size - 1
            self.copilot_observation_space = spaces.Box(
                -np.inf, np.inf, shape=(_state_size,), dtype=np.float32
            )
        else:
            raise RuntimeError(f'Env {self.env} is not supported by SplitObsWrapper')

    def observation(self, obs):
        """
        observation:  [x, y, vx, vy, r, rv, cl, cr]
        if v1:
          obs += [tgt_x, tgt_y]
        elif v5:
          obs += [helipad_x]
        """
        lvl = self.env.unwrapped.spec.name.split('-')[-1]
        pilot_obs = obs
        if lvl == 'v1':  # Reach
            copilot_obs = obs[:-2]
        elif lvl == 'v5':  # Land (helipad randomized)
            copilot_obs = obs[:-1]
        else:
            raise RuntimeError(f'Unknown env: {self.env.spec.name}')

        return {'pilot': pilot_obs, 'copilot': copilot_obs}


class BlockPushExpandObsWrapper(ObservationWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert 'Push' in self.env.unwrapped.spec.name

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

    def observation(self, obs):
        """
        observation:  obs space:  Dict(block_translation:Box(-5.0, 5.0, (2,), float32),
        block_orientation:Box(-6.2831855, 6.2831855, (1,), float32),
        effector_translation:Box([ 0.05 -0.6 ], [0.8 0.6], (2,), float32),
        effector_target_translation:Box([ 0.05 -0.6 ], [0.8 0.6], (2,), float32),
        target_translation:Box(-5.0, 5.0, (2,), float32),
        target_orientation:Box(-6.2831855, 6.2831855, (1,), float32),
        target2_translation:Box(-5.0, 5.0, (2,), float32),
        target2_orientation:Box(-6.2831855, 6.2831855, (1,), float32))
        """
        new_obs = []
        for key in ['block_translation', 'block_orientation', 'effector_translation', 'effector_target_translation']:
            new_obs.extend(obs[key])
        return np.array(new_obs)


class BlockPushSplitObsWrapper(ObservationWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert 'Push' in self.env.unwrapped.spec.name

        self.pilot_observation_space = self.observation_space
        # import pdb; pdb.set_trace()
        _state_size = self.observation_space.low.size - 8
        self.copilot_observation_space = spaces.Box(-np.inf, np.inf, shape=(_state_size,), dtype=np.float32)

    def observation(self, obs):
        pilot_obs = obs
        copilot_obs = obs[:-8]
        return {'pilot': pilot_obs, 'copilot': copilot_obs}


class BlockPushMirrorObsWrapper(Wrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert 'Push' in self.env.unwrapped.spec.name

    def step(self, act):
        new_act = self.action(act)
        obs, rew, done, info = super().step(new_act)
        new_obs = self.observation(obs)

        return new_obs, rew, done, info

    def action(self, action):
        """Mirror action along with x = 0 (i.e., just flip x axis values!!)"""
        new_action = deepcopy(action)
        new_action[0] = -action[0]
        return new_action

    def observation(self, obs):
        """Mirror observation along with x = 0 (i.e., just flip x axis values!!)"""
        new_obs = deepcopy(obs)  # Let's try not to update the values in-place
        # Flip x values of translations:
        for key in ['block_translation', 'effector_translation', 'effector_target_translation']:
            new_obs[key][0] = -obs[key][0]

        block_ori = obs['block_orientation']
        if - np.pi < block_ori + np.pi < np.pi:
            new_obs['block_orientation'] = block_ori + np.pi
        else:
            new_obs['block_orientation'] = block_ori - np.pi

        assert - np.pi < new_obs['block_orientation'] < np.pi, f'Hmmm the observation is {obs}'

        return new_obs

class BlockPushMirrorObsActorWrapper:
    def __init__(self, actor: Actor) -> None:
        self.actor = actor

    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        obs = self._flip_obs(obs)
        act = self.actor.act(obs)
        return self._flip_act(act)

    def batch_act(self, obss: np.ndarray) -> List[np.ndarray]:
        obss = np.array([self._flip_obs(obs) for obs in obss])
        acts = self.actor.batch_act(obss)
        return [self._flip_act(act) for act in acts]

    def _flip_obs(self, obs: np.ndarray):
        """
        obs:
        - obs[0:2] -- block_translation
        - obs[2:3] -- block_orientation
        - obs[3:5] -- ee_translation
        - obs[5:7] -- ee_target_translation
        """
        new_obs = deepcopy(obs)
        flip_dims = [0, 3, 5]
        new_obs[flip_dims] = -obs[flip_dims]

        return new_obs


    def _flip_act(self, act: np.ndarray):
        new_act = deepcopy(act)
        new_act[0] = - act[0]
        return new_act

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.actor, __name)
