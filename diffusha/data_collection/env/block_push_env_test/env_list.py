#!/usr/bin/env python3

import os
from pathlib import Path

import gym
import numpy as np
from diffusha.data_collection.env.block_push_env_test import eval_env as eval_env_module
from diffusha.data_collection.env.block_push_env_test.tasks import IBC_TASKS
from diffusha.data_collection.env import make_env
from diffusha.data_collection.env.block_pushing import block_pushing_multimodal_1block
from diffusha.data_collection.env.block_pushing.block_pushing_multimodal_1block import BlockPushMultimodal


def save_video(frame_stack, path, fps=20, **imageio_kwargs):
    """Save a vidoe from a list of frames.

    Correspondence: https://github.com/geyang/ml_logger
    """
    import os
    import tempfile, imageio  # , logging as py_logging
    import shutil
    # py_logging.getLogger("imageio").setLevel(py_logging.WARNING)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    format = 'mp4'
    with tempfile.NamedTemporaryFile(suffix=f'.{format}') as ntp:
        from skimage import img_as_ubyte
        try:
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        except imageio.core.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        ntp.seek(0)
        shutil.copy(ntp.name, path)

def save_pdf(frame, path):
    """Save a vidoe from a list of frames.

    Correspondence: https://github.com/geyang/ml_logger
    """
    import os
    import tempfile, imageio  # , logging as py_logging
    import shutil
    # py_logging.getLogger("imageio").setLevel(py_logging.WARNING)
    # print(frame)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    format = 'jpeg'
    # import pdb;pdb.set_trace()
    imageio.plugins.freeimage.download()
    imageio.imsave("tmp.jpeg", frame, format=format)
    shutil.copy("tmp.jpeg", path)


image_obs = False
shared_memory_eval = False
sequence_length = 2
goal_tolerance = 0.02
num_envs = 1
frames = []
env_name = 'LunarLander-v5'

if 'maze2d' in env_name and 'goal' in env_name:
    # Fix the goal to bottom left if it is maze2d env
    make_eval_env = lambda **kwargs: make_env(env_name, test=True, terminate_at_any_goal=True, bigger_goal=True,
                                              goal='left', **kwargs)
elif 'LunarLander' in env_name:
    make_eval_env = lambda **kwargs: make_env(env_name, test=True, split_obs=True, **kwargs)
elif 'Push' in env_name:
    make_eval_env = lambda **kwargs: make_env(env_name, test=True, user_goal='target', **kwargs)
else:
    raise RuntimeError()

eval_env = make_eval_env()
eval_env.reset()
for i in range(30):
    action = eval_env.action_space.sample()
    obs, reward, done, info = eval_env.step(action)

frames.append(eval_env.render())
frames = np.asarray(frames)
# save_video(frames, path=Path(os.environ['RMX_OUTPUT_DIR']) / 'video.mp4', fps=3)
save_pdf(frames[0], path=Path(os.environ['RMX_OUTPUT_DIR']) / 'frame2.jpeg')

# for task in IBC_TASKS:
#     if 'PARTICLE' in task:
#         continue
#
#     env_name = eval_env_module.get_env_name(task, shared_memory_eval, image_obs)
#     print('Got env name:', env_name)
#     #
#     # eval_env = eval_env_module.get_eval_env(env_name, sequence_length, goal_tolerance, num_envs)
#
#     eval_env = make_env("Push", test=False, seed=0)
#     # eval_env = suite_gym.load(env_name, max_episode_steps=100)
#
#     eval_env.reset()
#     frames.append(eval_env.render())
#     # done = False
#     # print("act space: ", eval_env.action_space, "obs space: ", eval_env.observation_space)
#     # step = 0
#     # while not done:
#     #     action = eval_env.action_space.sample()
#     #     # import pdb; pdb.set_trace()
#     #     obs, reward, done, info = eval_env.step(action)
#     #     # eval_env.reset()
#     #     # (240, 320, 3)
#     #     frames.append(eval_env.render())
#     #     step += 1
#     #
#     # print(action, step, reward, done, info)
#
# # Save video to a file
# frames = np.asarray(frames)
# # save_video(frames, path=Path(os.environ['RMX_OUTPUT_DIR']) / 'video.mp4', fps=3)
# save_pdf(frames[0], path=Path(os.environ['RMX_OUTPUT_DIR']) / 'frame.jpeg')
