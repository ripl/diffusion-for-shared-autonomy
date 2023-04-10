#!/usr/bin/env python3
"""
This is initially copied from https://github.com/jannerm/diffuser
The adopted version also lives in https://github.com/takuma-yoneda/d4rl@rmx
"""

import os
from typing import Callable
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
import gym
import warnings

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

MAZE_BOUNDS = {
    'maze2d-umaze2-v0': (0, 5, 0, 5),
    'maze2d-umaze2-mirror-v0': (0, 5, 0, 5),
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-funnel-v0': (0, 7, 0, 7),
    'maze2d-funnel-v1': (0, 8, 0, 7),
    'maze2d-funnel-goal-v0': (0, 9, 0, 9),
    'maze2d-large-v1': (0, 9, 0, 12),
    'maze2d-simple-multi-goal-v0': (0, 9, 0, 9),
    'maze2d-simple-two-goals-v0': (0, 9, 0, 9),
    'maze2d-simple-two-goals-v1': (0, 9, 0, 9),
    'maze2d-simple-two-goals-v2': (0, 9, 0, 9),
}

class MazeRenderer:
    def __init__(self, env):
        self._config = env._config
        self._background = self._config != ' '
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, title=None, updated_background=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        if updated_background is not None:
            raise ValueError('currently updated_background is not supported...')
            _background = updated_background
        else:
            _background = self._background
        plt.imshow(_background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)


        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], s=8, c=colors, zorder=20, linewidths=0.)

        # TEMP: draw rectangle manually ;P
        # from matplotlib.patches import Rectangle
        # plt.gca().add_patch(
        #     Rectangle((1, 1), 6, 5, fill=False, edgecolor='black', lw=4)
        # )

        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def render_multiple(self, trajectories, title=None, updated_background=None, alpha=0.2, color=None):
        """Render multiple trajectories on the same image"""
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        if updated_background is not None:
            raise ValueError('currently setting updated_background is not supported...')
            _background = updated_background
        else:
            _background = self._background

        # Fuck it, I don't know, but without this line, the entire plot goes upside down
        if self.env.unwrapped.spec.id == 'maze2d-funnel-v0' or self.env.unwrapped.spec.id == 'maze2d-simple-two-goals-v0':
            _background[:] = 0  # HACK
        plt.imshow(_background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        colormaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        for idx, observations in enumerate(trajectories):
            path_length = len(observations)
            # colors = plt.cm.jet(np.linspace(0,1,path_length))
            if color:
                colormap = getattr(plt.cm, color[idx])
            else:
                colormap = getattr(plt.cm, colormaps[idx % len(colormaps)])
            colors = colormap(np.linspace(0.5, 1, path_length))  # Let's try Greens. Other colors: (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            # plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
            plt.scatter(observations[:,1], observations[:,0], s=30, c=colors, zorder=20 + idx, alpha=alpha, label=idx, linewidths=0.)

        # TEMP: Draw bunch of lines instead of background
        if self.env.unwrapped.spec.id == 'maze2d-funnel-v0':
            pad = 0.0  # Somehow the trajectory goes into wall a bit without this
            shift_x = - 0.03
            shift_y = - 0.03
            linewidth = 4
            tl = (1/7 - pad + shift_x, 1/7 - pad + shift_y)
            tr = (6/7 + pad + shift_x, 1/7 - pad + shift_y)
            bl = (1/7 - pad + shift_x, 5/7 + pad + shift_y)
            br = (6/7 + pad + shift_x, 5/7 + pad + shift_y)
            mbl = (3/7 - pad + shift_x, 5/7 + pad + shift_y)
            mbr = (4/7 + pad + shift_x, 5/7 + pad + shift_y)
            plt.plot((tl[0], bl[0]), (tl[1], bl[1]), 'k', linewidth=linewidth)
            plt.plot((tr[0], br[0]), (tr[1], br[1]), 'k', linewidth=linewidth)
            plt.plot((tl[0], tr[0]), (tl[1], tr[1]), 'k', linewidth=linewidth)
            plt.plot((bl[0], mbl[0]), (bl[1], mbl[1]), 'k', linewidth=linewidth)
            plt.plot((mbr[0], br[0]), (mbr[1], br[1]), 'k', linewidth=linewidth)
            plt.plot((mbl[0], mbl[0]), (mbl[1], mbl[1] + 0.02), 'k', linewidth=linewidth)
            plt.plot((mbr[0], mbr[0]), (mbr[1], mbr[1] + 0.02), 'k', linewidth=linewidth)
            # TEMP: draw rectangle manually ;P
            # pad = 0.02  # Somehow the trajectory goes into wall a bit without this
            # from matplotlib.patches import Rectangle
            # plt.gca().add_patch(
            #     Rectangle((1/7 - pad, 1/7 - pad), 5/7 + pad*2, 4/7 + pad*2, fill=False, edgecolor='black', lw=4)
            # )

        if self.env.unwrapped.spec.id == 'maze2d-simple-two-goals-v0':
             pad = 0.0  # Somehow the trajectory goes into wall a bit without this
             shift_x = - 0.03
             shift_y = - 0.03
             linewidth = 6
             a = 2.24
             y = 2.25
             b = 7.32
             c = 7.8
             k = 1.35
             t = 7.33
             tl = (a / 9 - pad + shift_x, y / 9 - pad + shift_y)
             tr = (b / 9 + pad + shift_x, y / 9 - pad + shift_y)
             bl = (a / 9 - pad + shift_x, c / 9 + pad + shift_y)
             br = (b / 9 + pad + shift_x, c / 9 + pad + shift_y)
             blm = ((a+k) / 9 - pad + shift_x, c / 9 + pad + shift_y)
             brm = ((b-k) / 9 + pad + shift_x, c / 9 + pad + shift_y)
             mlm = ((a+k) / 9 - pad + shift_x, t / 9 + pad + shift_y)
             mrm = ((b-k) / 9 - pad + shift_x, t / 9 + pad + shift_y)
             plt.plot((tl[0], bl[0]), (tl[1], bl[1]), 'k', linewidth=linewidth)
             plt.plot((tr[0], br[0]), (tr[1], br[1]), 'k', linewidth=linewidth)
             plt.plot((tl[0], tr[0]), (tl[1], tr[1]), 'k', linewidth=linewidth)
             # plt.plot((bl[0], blm[0]), (bl[1], blm[1]), 'k', linewidth=linewidth)
             # plt.plot((brm[0], br[0]), (brm[1], br[1]), 'k', linewidth=linewidth)
             plt.plot((mlm[0], blm[0]), (mlm[1], blm[1]), 'k', linewidth=linewidth)
             plt.plot((mrm[0], brm[0]), (mrm[1], brm[1]), 'k', linewidth=linewidth)
             plt.plot((mlm[0], mrm[0]), (mlm[1], mrm[1]), 'k', linewidth=linewidth)
             # plt.plot((mbl[0], mbl[0]), (mbl[1], mbl[1] + 0.02), 'k', linewidth=linewidth)
             # plt.plot((mbr[0], mbr[0]), (mbr[1], mbr[1] + 0.02), 'k', linewidth=linewidth)
             plt.annotate(title, xy=((a+0.5) / 9 - pad + shift_x, (y-0.3) / 9 - pad + shift_y))

        plt.axis('off')
        # plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img


    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def single_composite(self, savepath, obs, ncol=1, **kwargs):
        '''
                    savepath : str
                    observations : [ n_paths x horizon x 2 ]
                '''
        assert len(obs) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = self.renders(obs)
        imageio.imsave(savepath, images)
        print(f'Saved {len(obs)} samples to: {savepath}')


class Maze2dRenderer(MazeRenderer):

    def __init__(self, env):
        self.env = env
        self.bounds = MAZE_BOUNDS[env.unwrapped.spec.id]

        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        observations = observations + .7
        if len(self.bounds) == 2:
            _, scale = self.bounds
            observations /= scale
        elif len(self.bounds) == 4:
            _, iscale, _, jscale = self.bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env}: {self.bounds}')

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)

    def render_multiple(self, trajectories, title=None, updated_background=None, alpha=0.2, color=None):
        for idx in range(len(trajectories)):
            observations = trajectories[idx]
            observations = observations + .7
            if len(self.bounds) == 2:
                _, scale = self.bounds
                observations /= scale
            elif len(self.bounds) == 4:
                _, iscale, _, jscale = self.bounds
                observations[:, 0] /= iscale
                observations[:, 1] /= jscale
            else:
                raise RuntimeError(f'Unrecognized bounds for {self.env}: {self.bounds}')
            trajectories[idx] = observations

        return super().render_multiple(trajectories, title=title, updated_background=updated_background, alpha=alpha, color=color)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)


def render_recorded_traj(env_name, traj_path, outdir):
    """
    Typically these are predictions by diffusion models, not experts.
    How these trajectories are recorded? --> diffusha/utils/eval.py record_traj fn
    """

    import d4rl
    from pathlib import Path
    from PIL import Image
    import torch
    env = gym.make(env_name)
    renderer = Maze2dRenderer(env)

    trajectories = torch.load(traj_path)
    obs_trajs = [np.asarray([trans['obs'] for trans in traj]) for traj in trajectories]
    img = renderer.render_multiple(obs_trajs, alpha=0.1)  # shape: (500, 500, 4)

    basename = Path(traj_path).stem
    Image.fromarray(img).save(Path(outdir) / f'recorded_{basename}.png')


def render_expert_traj(env_name, h5path, outdir, num_episodes, min_traj_len=10):
    from d4rl.pointmaze.maze_model import U_MAZE, U_MAZE2, OPEN, MazeEnv
    from PIL import Image
    import os
    from pathlib import Path

    # env = MazeEnv(U_MAZE2)
    env = gym.make(env_name)
    # maze = env.str_maze_spec
    # max_episode_steps = env._max_episode_steps

    # NOTE: dataset is a dict. Keys: dict_keys(['actions', 'infos/goal', 'infos/qpos', 'infos/qvel', 'observations', 'rewards', 'terminals'])
    # dataset['observations'].shape: (500000, 4)
    dataset = env.get_dataset(h5path=h5path)
    renderer = Maze2dRenderer(env)

    terminal_steps = np.where(dataset['terminals'])[0]
    print(f'There are {len(terminal_steps)} episodes in the dataset!')
    prev_term_step = 0
    counter = 0

    # Extract valid episodes
    episodes = []
    for term_step in terminal_steps:
        term_step = term_step + 1  # Why??
        if term_step - prev_term_step < min_traj_len:
            print(f'A short trajectory with len {term_step - prev_term_step} < {min_traj_len} is rejected.')
            prev_term_step = term_step
            continue

        ep = dataset['observations'][prev_term_step:term_step]
        episodes.append(ep)

        prev_term_step = term_step
        counter += 1

        if counter == num_episodes:
            break

    # Render each episode
    for idx, ep in enumerate(episodes):
        img = renderer.renders(ep)  # shape: (500 , 500, 4) (RGBA?)
        Image.fromarray(img).save(Path(outdir) / f'ep_{idx:04d}.png')

    # Render all episodes on a single image
    img = renderer.render_multiple(episodes[:10])
    Image.fromarray(img).save(Path(outdir) / f'ep_all.png')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-funnel-v0', help='Maze type')
    parser.add_argument('--h5path', type=str, default='/d4rl/maze2d/maze2d-funnel-v0.hdf5', help='Path to h5py file')
    parser.add_argument('--outdir', type=str, default=os.environ['RMX_OUTPUT_DIR'], help='Where to save the generated figures')
    parser.add_argument('--num-episodes', type=int, default=100, help='number of episodes to visualize')
    parser.add_argument('--mode', type=str, default='expert', choices=['expert', 'recorded'], help='specify mode')
    parser.add_argument('--traj-path', type=str, default='/data/ddpm/diffusha-diffusion-sweep_pointmaze/2x3u6aox/traj_0000000.pkl', help='specify mode')
    args = parser.parse_args()

    if args.mode == 'expert':
        render_expert_traj(args.env_name, args.h5path, args.outdir, args.num_episodes)
    elif args.mode == 'recorded':
        render_recorded_traj(args.env_name, args.traj_path, outdir=args.outdir)
