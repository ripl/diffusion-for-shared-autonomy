#!/usr/bin/env python3
"""This script performs evaluation on how much return each policy can achieve
with and without assistance, and saves the results to a file.

Diffusion Models trained with Floating task is loaded,
Then surrogate policy (some of which is trained for landing) is evaluated for landing task.
"""

import json
from diffusha.utils import patch
from os import PathLike
from functools import partial
from typing import Callable, Sequence

import os
import torch
import datetime


from diffusha.config.default_args import Args
from diffusha.data_collection.env import is_maze2d
from diffusha.diffusion.evaluation.helper import prepare_diffusha
from diffusha.diffusion.evaluation.eval import evaluate, evaluate_vector_env
from diffusha.diffusion.evaluation.eval import DiffusionActor

from diffusha.diffusion.ddpm import DiffusionModel
from diffusha.actor.assistive import DiffusionAssistedActor

from diffusha.diffusion.utils import initial_expert_agent
from diffusha.utils import reproducibility
from diffusha.actor import Actor, ZeroActor, RandomActor, LaggyActor, ExpertActor, NoisyActor, WaypointActor
import diffusha.utils
# from diffusha.utils.tticslurm import report_cuda_error, upload_slurm_logs


def get_actors(expert_agent, obs_space, act_space,
               laggy_actor_repeat_prob: float, noisy_actor_eps: float, actor_list=None):
    # agent = initial_expert_agent()
    expert_actor = ExpertActor(obs_space, act_space, expert_agent)
    zero_actor = ZeroActor(obs_space, act_space)
    laggy_expert_actor = LaggyActor(obs_space, act_space, expert_actor, repeat_prob=laggy_actor_repeat_prob, seed=123)
    noisy_expert_actor = NoisyActor(obs_space, act_space, expert_actor, eps=noisy_actor_eps, preserve_norm=False, seed=123)
    # random actor
    random_actor = RandomActor(obs_space, act_space, seed=123)

    actors = {'_base_agent': expert_agent, 'expert': expert_actor, 'zero': zero_actor, 'laggy': laggy_expert_actor, 'noisy': noisy_expert_actor, 'random': random_actor}

    # Filter if actor_list is set
    if actor_list is not None:
        actors = {key: val for key, val in actors.items() if (key in actor_list or key.startswith('_'))}

    return actors


def eval_assisted_actors(diffusion: DiffusionModel, make_env: Callable,
                         expert_agent, fwd_diff_ratio: float,
                         laggy_actor_repeat_prob: float, noisy_actor_eps: float,
                         num_episodes: int = 10, save_video: bool = False, histogram: bool = False, actor_list=None, use_vector_env=False, num_envs=1):
    """Create DiffusionAssistedActor for each expert obtained from get_actors func, and evaluate them with evaluate function."""

    sample_env = make_env()
    if hasattr(sample_env, 'copilot_observation_space'):
        obs_space = sample_env.copilot_observation_space
    else:
        obs_space = sample_env.observation_space
    act_space = sample_env.action_space

    actors = {
        name: DiffusionAssistedActor(obs_space, act_space, diffusion, actor, fwd_diff_ratio)
        for name, actor in get_actors(
                expert_agent, obs_space, act_space,
                laggy_actor_repeat_prob, noisy_actor_eps,
                actor_list=actor_list
        ).items() if not name.startswith('_')
    }

    log_entry = {}
    for name, actor in actors.items():
        print("-------------------------------------------------------------------")
        print(f"Running policy '{name}' with assistance")
        if use_vector_env:
            assert not save_video, 'saving video is not supported for vector env'
            log_entry[name] = evaluate_vector_env(make_env, actor, num_episodes, num_envs=num_envs, histogram=histogram)
        else:
            log_entry[name] = evaluate(make_env, actor, num_episodes, save_video=save_video, histogram=histogram)
    return log_entry


def eval_original_actors(make_env: Callable, expert_agent, num_episodes: int = 10,
                         save_video: bool = False, histogram: bool = False, diffusion_actor: bool = False, use_vector_env=False, num_envs=1):
    sample_env = make_env()
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space
    name = sample_env.unwrapped.spec.name

    # NOTE: for laggy_actor, look at diffusha/actor/base.py

    actors = {
        name: actor
        for name, actor in get_actors(
                expert_agent, obs_space, act_space,
                laggy_actor_repeat_prob, noisy_actor_eps,
        ).items()
        if not name.startswith("_")
    }

    log_entry = {}
    for name, actor in actors.items():
        print("-------------------------------------------------------------------")
        print(f"Running policy '{name}' without assistance")
        if use_vector_env:
            assert not save_video, 'saving video is not supported for vector env'
            log_entry[name] = evaluate_vector_env(make_env, actor, num_episodes, num_envs=num_envs, histogram=histogram)
        else:
            log_entry[name] = evaluate(make_env, actor, num_episodes, save_video=save_video, histogram=histogram)

    return log_entry


if __name__ == '__main__':
    import wandb
    from pathlib import Path
    from diffusha.data_collection.env import make_env
    from params_proto.hyper import Sweep
    import argparse

    from diffusha.diffusion.evaluation.eval import eval_diffusion

    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", help="sweep file")
    parser.add_argument("-l", "--line-number",
                        type=int, help="line number of the sweep-file")

    parser.add_argument("--env-name", default="LunarLander-v1", choices=['Maze2d-simple-two-goals-v0', 'LunarLander-v1', 'LunarLander-v5', 'BlockPushMultimodal-v1'], help="what env to use")
    parser.add_argument("--out-dir", help="Output directory")

    parser.add_argument("--num-episodes", type=int, default=300, help="number of evaluation episodes")
    parser.add_argument("--num-envs", type=int, default=30, help="number of environments to run in parallel")
    parser.add_argument("--save-video", action='store_true', help="Save videos of the episodes")
    parser.add_argument("--force", action='store_true', help="force to overwrite the existing file")

    args = parser.parse_args()

    if args.save_video:
        if args.num_envs > 1:
            args.num_envs = 1
            print('WARN: num_envs and num_episodes are overwritten to save the video.')
        args.num_episodes = min(args.num_episodes, 20)

    # Obtain kwargs from Sweep
    if args.sweep is not None and args.line_number is not None:
        sweep = Sweep(Args).load(args.sweep)
        kwargs = list(sweep)[args.line_number]

        if 'env_name' in kwargs:
            print('WARN: overwriting args.env_name with Args.env_name')
            args.env_name = kwargs['env_name']
    else:
        kwargs = {}

    # Set correct probabilities
    if args.env_name.split('-')[0] == 'LunarLander':
        if 'v1' in args.env_name:
            Args.laggy_actor_repeat_prob = kwargs.get('laggy_actor_repeat_prob', 0.85)
            Args.noisy_actor_eps = kwargs.get('noisy_actor_eps', 0.6)
        if 'v5' in args.env_name:
            Args.laggy_actor_repeat_prob = kwargs.get('laggy_actor_repeat_prob', 0.85)
            Args.noisy_actor_eps = kwargs.get('noisy_actor_eps', 0.3)

    # Update Args
    Args._update(kwargs)

    noisy_actor_eps = Args.noisy_actor_eps
    laggy_actor_repeat_prob = Args.laggy_actor_repeat_prob
    fwd_diff_ratio = Args.fwd_diff_ratio

    # timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    directory = Path(Args.results_dir) / 'assistance' / args.env_name.lower()
    directory.mkdir(mode=0o775, parents=True, exist_ok=True)

    # Read config (Args) from config.json
    with open(Path(__file__).parent / 'configs.json', 'r') as f:
        env2config = json.load(f)

    config = env2config[args.env_name]
    env2step = {'maze2d-simple-two-goals': 9999,
                'LunarLander-v1': 29999,
                'LunarLander-v5': 24000,
                'BlockPushMultimodal-v1': 29999}

    # Retrieve Args from wandb; NOTE: This updates Args internally!!
    diffusion = prepare_diffusha(
        env2config[args.env_name],
        Path(Args.ddpm_model_path) / args.env_name.lower(),
        env2step[args.env_name],
        args.env_name,
        fwd_diff_ratio,
        laggy_actor_repeat_prob,
        noisy_actor_eps
    )

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project='shared-autonomy-via-diffusion--eval',
        # project=f'test',
        group=f'{args.env_name}',
        config={'runtime': vars(args), 'Args': vars(Args)}
    )

    saved_keys = ['fwd_diff_ratio', 'laggy_actor_repeat_prob', 'noisy_actor_eps', 'obs_noise_level', 'obs_noise_cfg_prob']
    config = {}
    config['args'] = {key: val for key, val in vars(args).items() if (key not in ['sweep_file', 'line_number', 'force', 'dir_name', 'num_env'] and not key.startswith('_'))}
    config['Args'] = {key: val for key, val in vars(Args).items() if (key in saved_keys and not key.startswith('_'))}

    env_name = args.env_name
    if 'maze2d' in env_name and 'goal' in env_name:
        # Fix the goal to bottom left if it is maze2d env
        make_eval_env = lambda **kwargs: make_env(env_name, test=True, terminate_at_any_goal=True, bigger_goal=True, goal='left', **kwargs)
    elif 'LunarLander' in env_name:
        make_eval_env = lambda **kwargs: make_env(env_name, test=True, split_obs=True, **kwargs)
    elif 'Push' in env_name:
        make_eval_env = lambda **kwargs: make_env(env_name, test=True, user_goal='target', **kwargs)
    else:
        raise RuntimeError()

    sample_env = make_eval_env()
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space
    pilot_obs_space = sample_env.observation_space
    if 'LunarLander' in Args.env_name:
        copilot_obs_space = sample_env.copilot_observation_space.low.size
    else:
        copilot_obs_size = pilot_obs_space

    # name = sample_env.unwrapped.spec.name
    if 'maze2d' in env_name and 'goal' in env_name:
        # Maze2d goal
        _env = make_eval_env()
        expert_agent = WaypointActor(obs_space, act_space, _env)
    elif 'LunarLander' in env_name:
        from diffusha.data_collection.config.default_args import DCArgs
        # LunarLander
        lvl = env_name.split('-')[-1]
        modeldir = f'{DCArgs.lunarlander_sac_model_dir}/{lvl}'
        expert_agent = initial_expert_agent(make_eval_env, modeldir)

    elif 'Push' in env_name:
        from diffusha.data_collection.config.default_args import DCArgs
        push_model_dir = Path(DCArgs.blockpush_sac_model_dir)
        expert_agent = initial_expert_agent(make_eval_env, push_model_dir)

    if args.save_video:
        actor2log_entry = eval_original_actors(make_eval_env, expert_agent, num_episodes=args.num_episodes,
                                               save_video=True)
        actor2assist_log_entry = eval_assisted_actors(diffusion, make_eval_env, expert_agent,
                                                      fwd_diff_ratio=fwd_diff_ratio,
                                                      laggy_actor_repeat_prob=laggy_actor_repeat_prob,
                                                      noisy_actor_eps=noisy_actor_eps,
                                                      num_episodes=args.num_episodes, save_video=True)

    else:
        actor2log_entry = eval_original_actors(make_eval_env, expert_agent, num_episodes=args.num_episodes,
                                               save_video=False, use_vector_env=True, num_envs=args.num_envs)
        actor2assist_log_entry = eval_assisted_actors(diffusion, make_eval_env, expert_agent,
                                                      fwd_diff_ratio=fwd_diff_ratio,
                                                      laggy_actor_repeat_prob=laggy_actor_repeat_prob,
                                                      noisy_actor_eps=noisy_actor_eps,
                                                      num_episodes=args.num_episodes,
                                                      save_video=False, use_vector_env=True, num_envs=args.num_envs)

    if is_maze2d(sample_env):
        # Render visualized trajectories
        wandb.log({f'traj/assisted-{actor_name}': entry['maze-traj'] for actor_name, entry in actor2assist_log_entry.items()})
        wandb.log({f'traj/{actor_name}': entry['maze-traj'] for actor_name, entry in actor2log_entry.items()})

        # Remove all entries that are wandb objects, as they cause issues at loading time
        for entry in actor2assist_log_entry.values():
            del entry['maze-traj']
        for entry in actor2log_entry.values():
            del entry['maze-traj']

    if args.save_video:

        wandb.log({f'traj/assisted-{actor_name}': entry['video'] for actor_name, entry in actor2assist_log_entry.items()})
        wandb.log({f'traj/{actor_name}': entry['video'] for actor_name, entry in actor2log_entry.items()})

        # Remove all entries that are wandb objects, as they cause issues at loading time
        for entry in actor2assist_log_entry.values():
            del entry['video']
        for entry in actor2log_entry.values():
            del entry['video']

        # wandb.log(
        #     {
        #         f"{actor_name}/with-assist-{key}": val
        #         for actor_name, entry in actor2assist_log_entry.items()
        #         for key, val in entry.items()
        #     }
        # )
        # wandb.log(
        #     {
        #         f"{actor_name}/without-assist-{key}": val
        #         for actor_name, entry in actor2log_entry.items()
        #         for key, val in entry.items()
        #     }
        # )

    # Save the evaluation resutls to a file
    fname = directory / 'eval.pt'
    torch.save({'assisted': actor2assist_log_entry, 'original': actor2log_entry, 'hyperparams': config}, fname)
    print(f'The evaluation results are stored at: {fname}')
