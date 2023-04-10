#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Callable
import time
import numpy as np
import torch

import wandb
# from tqdm import tqdm

from diffusha.actor import Actor
from diffusha.actor.assistive import DiffusionAssistedActor
from diffusha.utils.renderer import Maze2dRenderer


def record_traj(make_env: Callable, actor: Actor, save_path: str | Path):
    env = make_env()
    assert 'maze2d' in env.unwrapped.spec.name, 'This function only supports maze2d env'


    trajectories = []
    print(f'reset locations: {env.reset_locations}')
    for loc in env.reset_locations:
        print(f'trying start loc: {loc}')
        traj = []
        done = False
        obs = env.reset()

        # Set initial state
        qvel = env.init_qvel * 0.
        qpos = np.array(loc)
        env.set_state(qpos, qvel)

        # Rollout the episode
        while not done:
            act = actor.act(obs)
            next_obs, rew, done, info = env.step(act)

            traj.append(
                {'obs': obs, 'act': act, 'next_obs': next_obs, 'start_loc': loc, 'done': done}
            )
            obs = next_obs
        trajectories.append(traj)

    # Save trajectories
    torch.save(trajectories, save_path)

    # Render trajectories overlayed on the same image
    renderer = Maze2dRenderer(env)
    obs_trajs = [np.asarray([trans['obs'] for trans in traj]) for traj in trajectories]
    img = renderer.render_multiple(obs_trajs, alpha=0.2)  # shape: (500, 500, 4)
    return img
