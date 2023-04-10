#!/usr/bin/env python3

import pfrl
import gym

from diffusha.data_collection.env.assistance_wrappers import BlockPushMirrorObsWrapper, LunarLanderSplitObsWrapper, BlockPushSplitObsWrapper, BlockPushExpandObsWrapper
from diffusha.data_collection.env.multiprocess_vector_env import MultiprocessVectorEnv
from diffusha.data_collection.env.block_pushing.block_pushing_multimodal_1block import BlockPushMultimodal


class Spec:
    def __init__(self, name) -> None:
        self.name = name


def make_env(env_name, test, seed=0, terminate_at_any_goal=False, split_obs=False, user_goal='target', **kwargs):
    if split_obs:
        assert "maze" not in env_name

    if "LunarLander" in env_name:
        """
        - LunarLander-v1: Reach task
        - LunarLander-v2: Landing task
        - LunarLander-v3: Deprecated (hand-crafted floating reward)
        - LunarLander-v4: Floating task
        - LunarLander-v5: Landing task, helipad is randomized every time
        """
        if kwargs == {}:
            pass
        else:
            print('extra kwargs to make_env:', kwargs)
        from .lunar_lander import LunarLander
        version = env_name.split('-')[-1]
        if version == 'v1':
            # Reach task with Chip's reward
            env = LunarLander(continuous=True, task='reach', spec=Spec(f'LunarLander-{version}'), **kwargs)
        elif version == 'v4':
            # Float task
            env = LunarLander(continuous=True, task='float', fuel_penalty=False, spec=Spec(f'LunarLander-{version}'), **kwargs)
        elif version == 'v5':
            env = LunarLander(continuous=True, randomize_helipad=True, spec=Spec(f'LunarLander-{version}'), **kwargs)
        else:
            # Landing task
            env = LunarLander(continuous=True, randomize_helipad=False, spec=Spec(f'LunarLander-{version}'), **kwargs)
        time_limit = 1000

    elif 'maze' in env_name:
        import d4rl
        if terminate_at_any_goal:
            env = gym.make(env_name, reward_type='sparse', terminate_at_any_goal=True, **kwargs)
        else:
            env = gym.make(env_name, reward_type='sparse', terminate_at_goal=True, **kwargs)
        time_limit = 300

    else:
        # for block pushing task
        env = gym.make('BlockPushMultimodal-v1', user_goal=user_goal)
        # import pdb; pdb.set_trace()
        time_limit = 200

    env_seed = 2**32 - 1 - seed if test else seed
    env.seed(env_seed)

    if "LunarLander" in env_name:
        if env_name == "LunarLander-v3":
            from .reward_wrapper import LunarLanderRewardWrapper
            # Simple reward
            env = LunarLanderRewardWrapper(env)

    elif 'maze' in env_name:
        from .pointmaze_wrapper import PointMazeTerminationWrapper
        env = PointMazeTerminationWrapper(env)

    elif 'Push' in env_name:
        # NOTE: Wrapper should be applied to the actor, rather than environment.
        # if user_goal == 'target2':
        #     # Mirror the observation so that the scene looks identical to user_goal=='target' for the policy
        #     env = BlockPushMirrorObsWrapper(env)

        env = BlockPushExpandObsWrapper(env)
    else:
        raise "Unexpected Env Name"

    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)
    # Normalize action space to [-1, 1]^n
    env = pfrl.wrappers.NormalizeActionSpace(env)

    # Split observation into pilot and copilot
    if 'LunarLander' in env_name and split_obs:
        env = LunarLanderSplitObsWrapper(env)

    if 'Push' in env_name and split_obs:
        env = BlockPushSplitObsWrapper(env)

    if 'Push' not in env_name:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)

    return env


def is_maze2d(env):
    return 'maze2d' in env.unwrapped.spec.name

def is_lunarlander(env):
    return 'LunarLander' in env.unwrapped.spec.name


def is_blockpush(env):
    return 'BlockPush' in env.unwrapped.spec.name

