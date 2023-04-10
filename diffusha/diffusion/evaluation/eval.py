from __future__ import annotations

import functools
import os
import pdb
import time
from pathlib import Path
from typing import Callable, List, Tuple
import math

import numpy as np
import torch
from diffusha.actor.assistive import DiffusionAssistedActor
import wandb
from diffusha.actor.base import Actor, choose_obs_if_necessary
from diffusha.data_collection.env import is_lunarlander, is_maze2d, is_blockpush
import pfrl

from diffusha.utils.eval import record_traj

# Magic to avoid circular import due to type hint annotation
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusha.diffusion.ddpm import DiffusionModel


def sample(
    diffusion: DiffusionModel,
    start_obs,
    k: int | None = None,
    fwd_diff_ratio: float | None = None,
):
    """Conditional sampling"""
    assert (k is not None) or (
        fwd_diff_ratio is not None
    ), "You must specify either k or fwd_diff_ratio."

    if k is None and fwd_diff_ratio is not None:
        k = int((diffusion.num_diffusion_steps - 1) * fwd_diff_ratio)
        print(
            f"forward diffusion steps for action: {k} / {diffusion.num_diffusion_steps}"
        )

    state = torch.as_tensor(start_obs, dtype=torch.float32)

    # Forward diffuse user_act for k steps
    x_k, e = diffusion.diffuse(state.unsqueeze(0), torch.as_tensor([k]))

    # Reverse diffuse for k steps
    x_i = x_k
    for i in reversed(range(k)):
        x_i = diffusion.p_sample(x_i, i)

    out = x_i.squeeze()  # Remove batch dim
    return out


class DiffusionActor(Actor):
    def __init__(
        self, diffusion: DiffusionModel, obs_size, act_size, naive_cond=False
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.diffusion = diffusion
        self.naive_cond = naive_cond
        # self.device = 'cuda'  # TEMP

    def _cond_sample(self, state: np.ndarray, run_in_batch=False):
        """Conditional sampling

        Args:
            - state: (state_dim, ) or (batch_size, state_dim)
        """
        assert (
            state.shape[-1] == self.obs_size
        ), f"obs dimension mismatch: actual {state.shape[-1]} vs expected {self.obs_size}"

        # Get the state shape the diffusion model should work on
        if len(state.shape) == 2:  # (batch_size, state_dim)
            shape = (state.shape[0], self.obs_size + self.act_size)
        elif len(state.shape) == 1:  # (state_dim, )
            shape = self.obs_size + self.act_size
        else:
            raise ValueError(f"Unsupported shape in state: {state.shape}")

        # xt = torch.cat((state, rand_act)).to(self.device)
        out, _ = self.diffusion.p_sample_loop(
            shape, cond=torch.as_tensor(state), naive_cond=self.naive_cond
        )
        if not run_in_batch:
            out = out.squeeze()  # Remove batch dim
            return out[self.obs_size :].cpu().numpy()
        else:
            return out[..., self.obs_size :].cpu().numpy()

    def act(self, state):
        state = choose_obs_if_necessary(state, actor="copilot")  # Check with Luzhe!!
        state = state[None, :]  # Add batch dim
        act = self._cond_sample(state)
        return act  # Batch dim is already squeezed in _cond_sample !!

    def batch_act(self, states):
        states = np.asarray(
            [choose_obs_if_necessary(state, actor="copilot") for state in states]
        )
        actions = self._cond_sample(states, run_in_batch=True)
        return actions


def evaluate_vector_env(
    make_env: Callable,
    actor: Actor,
    num_episodes: int = 100,
    num_envs: int = 1,
    histogram=True,
) -> dict:
    """Evaluate actor on the env created with make_vector_env(), and returns a dictionary of evaluation results."""
    from diffusha.data_collection.env import MultiprocessVectorEnv

    vector_env = MultiprocessVectorEnv(
        [functools.partial(make_env, seed=i) for i in range(num_envs)]
    )

    sample_env = make_env()
    is_ll = is_lunarlander(sample_env)
    is_maze = is_maze2d(sample_env)
    is_bp = is_blockpush(sample_env)

    timer = {}
    num_envs = vector_env.num_envs
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")
    batch_curr_traj = [[] for _ in range(num_envs)]
    batch_trajs = [[] for _ in range(num_envs)]

    assert (
        num_episodes % num_envs == 0
    ), f"num_episodes ({num_episodes}) must be divisible by num_envs ({num_envs})"
    num_episodes_per_env = num_episodes // num_envs

    init_seeds = np.arange(num_envs) + 1

    # import pdb;pdb.set_trace()
    if is_ll:
        obss = vector_env.reset(seeds=init_seeds)
    else:
        obss = vector_env.reset()
    # print("reset test successful")
    # print("obss:",obss)

    step = 0
    print(f"Running {num_envs} envs in parallel...")
    while episode_idx.min() < num_episodes_per_env:
        started = time.perf_counter()
        if isinstance(actor, DiffusionAssistedActor):
            actions, diffs = actor.batch_act(obss, report_diff=True)
        else:
            actions = actor.batch_act(obss)
            diffs = [0.0 for _ in range(num_envs)]
        timer["predict-action"] = time.perf_counter() - started

        next_obss, rs, dones, infos = vector_env.step(actions)

        # Store transitions to trajectories
        for idx, (obs, act, r, next_obs, done, info, diff) in enumerate(
            zip(obss, actions, rs, next_obss, dones, infos, diffs)
        ):
            info["action_diff"] = diff
            batch_curr_traj[idx].append(
                {
                    "obs": obs,
                    "act": act,
                    "rew": r,
                    "next_obs": next_obs,
                    "done": done,
                    "info": info,
                }
            )

            if done:
                # Store the trajectory to batch_trajs, and
                # remove the corresponding batch_curr_traj
                batch_trajs[idx].append(batch_curr_traj[idx].copy())
                batch_curr_traj[idx].clear()
                # print("num completed episodes (per env)", episode_idx)

        # Compute mask for done and reset
        max_episode_steps = getattr(vector_env.spec, "max_episode_steps", None)
        if max_episode_steps is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = episode_len == max_episode_steps
        resets = np.logical_or(
            resets, [info.get("needs_reset", False) for info in infos]
        )

        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)
        episode_idx += end

        seeds = init_seeds + episode_idx * num_envs
        # print("seeds:", seeds)

        # Reset only those env that are `end`
        if is_ll:
            obss = vector_env.reset(mask=not_end, seeds=seeds)
        else:
            obss = vector_env.reset(mask=not_end)

        # Overwrite only those observation that are not end yet
        for index in range(num_envs):
            if not_end[index]:
                obss[index] = next_obss[index]

        step += 1

    trajectories = []
    for idx in range(len(batch_trajs)):
        trajectories.extend(batch_trajs[idx][:num_episodes_per_env])

    # TODO: Retrieve env-specific termination info da ta(goal, reach, land etc)
    sum_diffs = []

    returns = []
    ep_lengths = []
    if is_maze:
        maze_correct_goal = []
        maze_wrong_goal = []
        maze_time_out = []
    if is_ll:
        success = []
        game_over = []
    if is_bp:
        push_correct_goal = []
        push_wrong_goal = []
        push_time_out = []

    for index, traj in enumerate(trajectories):
        ep_lengths.append(len(traj))
        sum_reward = 0.0
        sum_diff = 0.0
        for curr_dict in traj:
            # {'obs': obs, 'act': act, 'rew': r, 'next_obs': next_obs, 'done': done, 'info': info}
            sum_reward += curr_dict["rew"]
            sum_diff += curr_dict["info"]["action_diff"]
        returns.append(sum_reward)
        sum_diffs.append(sum_diff)

        # Env-specific metrics
        if is_ll:
            info = traj[-1]["info"]
            success.append(
                "goal" in info.keys()
                and (info["goal"] == "landed" or info["goal"] == "target-reached")
            )
            game_over.append("crashed" in info.keys() and info["crashed"])

        elif is_maze:
            info = traj[-1]["info"]
            if "TimeLimit.truncated" in info.keys():
                # reach nowhere
                maze_correct_goal.append(0)
                maze_wrong_goal.append(0)
                maze_time_out.append(1)
            elif "target_reached" in info.keys():
                # reach the correct goal
                if info["target_reached"] == "left":
                    # reach correct goal
                    maze_correct_goal.append(1)
                    maze_wrong_goal.append(0)
                    maze_time_out.append(0)
                else:
                    # reach wrong goal
                    maze_correct_goal.append(0)
                    maze_wrong_goal.append(1)
                    maze_time_out.append(0)

        elif is_bp:
            info = traj[-1]["info"]
            push_correct_goal.append(info["finished"])
            # (info['state'] != 'NotDone') already in target
            # (not info['finished']) but not in the correct target
            push_wrong_goal.append(
                (info["state"] != "NotDone") and (not info["finished"])
            )
            push_time_out.append(info["state"] == "NotDone")
        else:
            raise RuntimeError(f"unknown env: {vector_env}")

        # print(f'ep: {index}\tsum_rewards: {sum_reward}\tep_len: {len(traj)}')

    wandb_log_entry = {
        "return": returns,
        "ep_lengths": ep_lengths,
        "return_mean": np.array(returns).mean(),
        "ep_len_mean": np.array(ep_lengths).mean(),
        # frames: (t, h, w, c) -> (t, c, h, w)
        # 'diffusion_elapsed_mean': np.array(diffusion_elapsed).mean(),
        # 'trajectories': trajectories,
    }

    # Env-specific metrics
    if is_ll:
        wandb_log_entry = {
            "ll-success": success,
            "ll-game_over": game_over,
            **wandb_log_entry,
        }
        print(f"Success: {sum(success)} / {len(success)}")
    elif is_maze:
        # Visualize the trajectories
        from diffusha.utils.renderer import Maze2dRenderer

        renderer = Maze2dRenderer(sample_env)
        obs_trajs = [
            np.asarray([trans["obs"] for trans in traj]) for traj in trajectories
        ]
        # set diff color to diff trajectories
        color = []
        for i in range(len(maze_correct_goal)):
            if maze_correct_goal[i]:
                color.append("Greens")
            elif maze_wrong_goal[i]:
                color.append("Blues")
            else:
                color.append("Reds")
        title = f"left:{int(np.array(maze_correct_goal).sum())} right:{int(np.array(maze_wrong_goal).sum())} time out:{int(np.array(maze_time_out).sum())}"
        img = renderer.render_multiple(
            obs_trajs, alpha=0.2, color=color, title=title
        )  # shape: (500, 500, 4)
        wandb_log_entry = {
            "maze-traj": wandb.Image(img, mode="RGBA"),
            "maze-correct-goal": maze_correct_goal,
            "maze-wrong-goal": maze_wrong_goal,
            "maze-time-out": maze_time_out,
            **wandb_log_entry,
        }
        sample_env.close()

    elif is_bp:
        wandb_log_entry = {
            "bp-correct-goal": push_correct_goal,
            "bp-wrong-goal": push_wrong_goal,
            "bp-time-out": push_time_out,
            **wandb_log_entry,
        }
        print(f"Correct: {sum(push_correct_goal)} / {len(push_correct_goal)}")
    else:
        raise RuntimeError(f"unknown env: {vector_env}")

    if histogram:
        for entry in ["return", "ep_lengths"]:
            if entry in wandb_log_entry:
                wandb_log_entry[entry] = wandb.Histogram(wandb_log_entry[entry])
    vector_env.close()
    return wandb_log_entry


def evaluate(
    make_env: Callable,
    actor: Actor,
    num_episodes: int = 10,
    save_video=True,
    histogram=True,
    render_scale_img=1.0,
) -> dict:
    """Evaluate actor on the env created with make_env(), and returns a dictionary of evaluation results."""
    from diffusha.data_collection.env.eval_hook import get_eval_frame
    from tqdm import tqdm

    # NOTE: It's better to have an identical set of episodes for every evaluation, thus creating it every time.
    # env = make_env(test=True, seed=0, time_limit=TIME_LIMIT)
    # import pdb; pdb.set_trace()
    sample_env = make_env(seed=0)
    is_ll = is_lunarlander(sample_env)
    is_maze = is_maze2d(sample_env)
    is_bp = is_blockpush(sample_env)

    assisted = isinstance(actor, DiffusionAssistedActor)
    if assisted:
        diffs = []
        sum_diffs = []

    returns = []
    ep_lengths = []
    frames = []
    diffusion_elapsed = []
    trajectories = []
    if is_maze:
        maze_correct_goal = []
        maze_wrong_goal = []
        maze_time_out = []
    if is_ll:
        success = []
        game_over = []
    if is_bp:
        push_correct_goal = []
        push_wrong_goal = []
        push_time_out = []

    # ll_seed = [3, 4, 6] # Lander
    # ll_seed = [4,6,8] # Reacher
    for ep in tqdm(range(num_episodes)):
        done = False
        if is_ll:
            obs = sample_env.reset(seed=ep)
            # obs = sample_env.reset(seed=(ep*3))
        else:
            obs = sample_env.reset()
        sum_rewards = 0
        sum_diff = 0
        step = 0
        traj = []

        # Original resolution is (400 x 600)
        if save_video:
            frames.append(get_eval_frame(sample_env, ep, 0, scale=render_scale_img))

        while not done:
            started = time.time()
            if assisted:
                action, diff = actor.act(obs, report_diff=True)
                diffs.append(diff)
                sum_diff += diff
            else:
                action = actor.act(obs)
            diffusion_elapsed.append(time.time() - started)
            next_obs, rew, done, info = sample_env.step(action)
            # next_obs, rew, done, info = env.step(env.action_space.naive_sample())
            sum_rewards += rew
            step += 1
            traj.append(
                {
                    "obs": obs,
                    "act": action,
                    "next_obs": next_obs,
                    "done": done,
                    "rew": rew,
                    "info": info,
                }
            )
            obs = next_obs
            if save_video:
                if not done:
                    # frames.append(sample_env.render())
                    frames.append(get_eval_frame(sample_env, ep, step, scale=render_scale_img))
                else:
                    last_frame = get_eval_frame(
                        sample_env, ep, step, info=info, scale=render_scale_img
                    )
                    for _ in range(10):
                        frames.append(last_frame)

        trajectories.append(traj)

        # Env-specific metrics
        if is_ll:
            info = traj[-1]["info"]
            success.append(
                "goal" in info.keys()
                and (info["goal"] == "landed" or info["goal"] == "target-reached")
            )
            game_over.append("crashed" in info.keys() and info["crashed"])

        elif is_maze:
            info = traj[-1]["info"]
            if "TimeLimit.truncated" in info.keys():
                # reach nowhere
                maze_correct_goal.append(0)
                maze_wrong_goal.append(0)
                maze_time_out.append(1)
            elif "target_reached" in info.keys():
                # reach the correct goal
                if info["target_reached"] == "left":
                    # reach correct goal
                    maze_correct_goal.append(1)
                    maze_wrong_goal.append(0)
                    maze_time_out.append(0)
                else:
                    # reach wrong goal
                    maze_correct_goal.append(0)
                    maze_wrong_goal.append(1)
                    maze_time_out.append(0)

        elif is_bp:
            info = traj[-1]["info"]
            push_correct_goal.append(info["finished"])
            # (info['state'] != 'NotDone') already in target
            # (not info['finished']) but not in the correct target
            push_wrong_goal.append(
                (info["state"] != "NotDone") and (not info["finished"])
            )
            push_time_out.append(info["state"] == "NotDone")
        else:
            raise RuntimeError(f"unknown env: {sample_env}")

        # print(f'ep: {ep}\tsum_rewards: {sum_rewards}\tep_len: {step}\t info:{info}')
        returns.append(sum_rewards)
        ep_lengths.append(step)
        if assisted:
            sum_diffs.append(sum_diff)

    # NOTE: How large can a video be?
    # num_ep: 10, num_frames: 1000, width: 600, height: 400, channels: 3
    # 10 * 1000 * (600 * 400 * 3) * 1 (byte) = 7.2 GB !!
    wandb_log_entry = {
        "return": returns,
        "ep_lengths": ep_lengths,
        "return_mean": np.array(returns).mean(),
        "ep_len_mean": np.array(ep_lengths).mean(),
        # frames: (t, h, w, c) -> (t, c, h, w)
        # 'diffusion_elapsed_mean': np.array(diffusion_elapsed).mean(),
        # 'trajectories': trajectories,
    }

    # Env-specific metrics
    if is_ll:
        wandb_log_entry = {
            "ll-success": success,
            "ll-game_over": game_over,
            **wandb_log_entry,
        }

        # print("success:", success)
        # print("game over:", game_over)
        # print("sum_success:", np.array(success).sum())
        # print("sum_game_over:", np.array(game_over).sum())
        print("success_rate", np.sum(success) / len(success))

    elif is_maze:
        # Visualize the trajectories
        from diffusha.utils.renderer import Maze2dRenderer

        renderer = Maze2dRenderer(sample_env)
        obs_trajs = [
            np.asarray([trans["obs"] for trans in traj]) for traj in trajectories
        ]
        # set diff color to diff trajectories
        color = []
        for i in range(len(maze_correct_goal)):
            if maze_correct_goal[i]:
                color.append("Greens")
            elif maze_wrong_goal[i]:
                color.append("Blues")
            else:
                color.append("Reds")
        title = f"left:{int(np.array(maze_correct_goal).sum())} right:{int(np.array(maze_wrong_goal).sum())} time out:{int(np.array(maze_time_out).sum())}"
        img = renderer.render_multiple(
            obs_trajs, alpha=0.2, color=color, title=title
        )  # shape: (500, 500, 4)
        wandb_log_entry = {
            "maze-traj": wandb.Image(img, mode="RGBA"),
            "maze-correct-goal": maze_correct_goal,
            "maze-wrong-goal": maze_wrong_goal,
            "maze-time-out": maze_time_out,
            **wandb_log_entry,
        }
    elif is_bp:
        wandb_log_entry = {
            "bp-correct-goal": push_correct_goal,
            "bp-wrong-goal": push_wrong_goal,
            "bp-time-out": push_time_out,
            **wandb_log_entry,
        }
    else:
        raise RuntimeError(f"unknown env: {sample_env}")

    if assisted:
        wandb_log_entry["action_diffs"] = diffs
        wandb_log_entry["action_correction_level"] = np.array(diffs).mean()
        wandb_log_entry["sum_diffs"] = sum_diffs

    if save_video:
        if is_bp:
            wandb_log_entry["video"] = wandb.Video(
                np.asarray(frames).transpose(0, 3, 1, 2), fps=15, format="mp4"
            )
        if is_ll:
            wandb_log_entry["video"] = wandb.Video(
                np.asarray(frames).transpose(0, 3, 1, 2), fps=50, format="mp4"
            )

    if histogram:
        for entry in ["return", "ep_lengths", "action_diffs", "total_diffs"]:
            if entry in wandb_log_entry:
                wandb_log_entry[entry] = wandb.Histogram(wandb_log_entry[entry])

    return wandb_log_entry


def eval_diffusion(
    diffusion: DiffusionModel,
    make_env: Callable,
    num_episodes: int = 20,
    save_video: bool = True,
    use_vector_env: bool = False,
) -> dict:
    """Evaluate diffusion model given by `diffusion` on an env, using evaluate() function."""
    print("Running evaluation...")
    # env = make_env(test=True, seed=0, time_limit=TIME_LIMIT)
    sample_env = make_env()
    act_space = sample_env.action_space
    act_size = act_space.low.size
    copilot_obs_space = getattr(
        sample_env, "copilot_observation_space", sample_env.observation_space
    )
    copilot_obs_size = copilot_obs_space.low.size

    # NOTE: for laggy_actor, look at diffusha/actor/base.py
    actors = {
        "diff_naive_actor": DiffusionActor(
            diffusion, obs_size=copilot_obs_size, act_size=act_size, naive_cond=True
        ),
        "diff_actor": DiffusionActor(
            diffusion, obs_size=copilot_obs_size, act_size=act_size, naive_cond=False
        ),
    }
    if use_vector_env:
        if save_video:
            print("warn: save_video is not supported in vector env")
        num_envs = 10
        log_entry = {
            name: evaluate_vector_env(make_env, actor, num_episodes, num_envs=num_envs)
            for name, actor in actors.items()
        }
    else:
        log_entry = {
            name: evaluate(make_env, actor, num_episodes, save_video=save_video)
            for name, actor in actors.items()
        }

    print("Running evaluation...done!")
    return log_entry


def generate_state_traj(env, state_diffusion: DiffusionModel, save_path: str | Path):
    """Given a diffusion model trained for state marginal,
    This function performs forward and reverse diffusion with different magnitude, starting from current state,
    and records that trajectories.
    """
    from d4rl.pointmaze import MazeEnv

    assert isinstance(env, MazeEnv)
    print(
        "Generating state trajectories by varying number of diffusion timesteps for forward/backward..."
    )
    from diffusha.utils.renderer import Maze2dRenderer

    renderer = Maze2dRenderer(env)
    trajectories = []
    for loc in env.reset_locations:
        print(f"trying start loc: {loc}")

        # Set initial state
        qvel = env.init_qvel * 0.0
        qpos = np.array(loc)
        start_obs = np.concatenate([qpos, qvel])

        states = np.array(
            [
                sample(state_diffusion, start_obs, k=k).cpu().numpy()
                for k in range(state_diffusion.num_diffusion_steps)
            ]
        )
        trajectories.append(states)

    torch.save(trajectories, save_path)

    # Render trajectories overlayed on the same image
    img = renderer.render_multiple(trajectories, alpha=0.2)  # shape: (500, 500, 4)
    return {"state_traj": wandb.Image(img, mode="RGBA")}


def pointmaze_record_traj(
    diffusion: DiffusionModel, make_env: Callable, traj_save_dir: str | Path, step: int
):
    print("recording trajectories...")
    sample_env = make_env()
    obs_size = sample_env.observation_space.low.size
    act_size = sample_env.action_space.low.size
    diff_naive_actor = DiffusionActor(
        diffusion, obs_size=obs_size, act_size=act_size, naive_cond=True
    )
    diff_actor = DiffusionActor(
        diffusion, obs_size=obs_size, act_size=act_size, naive_cond=False
    )
    naive_img = record_traj(
        make_env,
        diff_naive_actor,
        save_path=Path(traj_save_dir) / f"naive_traj_{step:07d}.pkl",
    )
    img = record_traj(
        make_env, diff_actor, save_path=Path(traj_save_dir) / f"traj_{step:07d}.pkl"
    )
    print("recording trajectories...done")

    return {
        "naive_traj": wandb.Image(naive_img, mode="RGBA"),
        "traj": wandb.Image(img, mode="RGBA"),
    }
