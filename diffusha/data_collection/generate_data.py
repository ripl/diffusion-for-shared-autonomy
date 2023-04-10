#!/usr/bin/env python3

"""Script to generate trajectories from pretrained expert, only for LunarLander envs"""

from __future__ import annotations
from diffusha.actor import ExpertActor
import os
from datetime import datetime
import random
from pathlib import Path
import argparse
from typing import Optional
import numpy as np
import torch
from pfrl import utils
import wandb

from diffusha.data_collection.env.assistance_wrappers import (
    BlockPushMirrorObsActorWrapper,
)

from .utils.agent import get_agent
from .env import make_env
from .train_sac import TIME_LIMIT
from diffusha.data_collection.config.default_args import DCArgs
from diffusha.config.default_args import Args


class ReplayBuffer:
    def __init__(self, directory, state_dim, action_dim, chunk_size=10_000) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o775, exist_ok=True, parents=True)

        # +1 is to store q_val
        self._current_buff = np.zeros(
            (chunk_size, state_dim + action_dim + 1), dtype=np.float32
        )
        self.pointer = 0
        self.chunk_size = chunk_size

        self._file_cache = {}

        # If there are files in the directory, lets read all files first
        for fname in self.directory.iterdir():
            print(f"loading {fname} from {self.directory}...")
            self._file_cache[fname] = torch.load(fname)

    def store(
        self, state: np.ndarray, action: np.ndarray, q_val: Optional[np.ndarray] = None
    ) -> None:
        if q_val is None:
            q_val = np.zeros((1,), dtype=np.float32)
        entry = np.concatenate((state, action, q_val), dtype=np.float32)
        self._current_buff[self.pointer] = entry
        self.pointer += 1

        if self.pointer == self.chunk_size:
            self.dump_buffer()
            self.pointer = 0

    def dump_buffer(self):
        """Save the content of current buffer, and empty it."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        fname = self.directory / (timestamp + ".pkl")
        print(f"saving chunk to {str(fname)}")
        torch.save(self._current_buff, fname)

    def sample(self):
        import random

        assert len(self._file_cache) > 0
        fname = random.choice(list(self._file_cache.keys()))
        buff = self._file_cache[fname]

        # Sample a random idx
        idx = random.randint(0, buff.shape[0] - 1)
        return buff[idx]


def main(save_video):
    from diffusha.data_collection.env.eval_hook import get_frame

    # Set a random seed used in PFRL
    utils.set_random_seed(DCArgs.seed)

    sample_env = make_env(
        DCArgs.env_name,
        test=True,
        seed=DCArgs.seed,
        user_goal=DCArgs.blockpush_user_goal,
    )
    obs_size = sample_env.observation_space.low.size
    act_size = sample_env.action_space.low.size

    if "LunarLander" in DCArgs.env_name:
        lvl = DCArgs.env_name.split("-")[-1]
        modeldir = f"{DCArgs.lunar_sac_model_dir}/{lvl}"
        datadir = f"{DCArgs.lunar_data_dir}/{lvl}/randp_{DCArgs.randp}"
    elif "Push" in DCArgs.env_name:
        modeldir = Path(DCArgs.blockpush_sac_model_dir)
        datadir = (
            Path(DCArgs.blockpush_data_dir)
            / DCArgs.blockpush_user_goal
            / f"randp_{DCArgs.randp}"
        )
    else:
        raise ValueError(f"{DCArgs.env_name} is not supported")

    replay_buffer = ReplayBuffer(datadir, obs_size, act_size)

    # NOTE: batch_size and replay_start_size shouldn't matter
    agent = get_agent(
        sample_env, policy_output_scale=1.0, batch_size=256, replay_start_size=10_000
    )
    agent.load(modeldir)
    actor = ExpertActor(sample_env.observation_space, sample_env.action_space, agent)

    if "Push" in DCArgs.env_name and DCArgs.blockpush_user_goal == "target2":
        # Mirror observation
        print("Using BlockPushMirrorObsActorWrapper")
        actor = BlockPushMirrorObsActorWrapper(actor)

    # Collect data
    env = make_env(
        DCArgs.env_name,
        test=False,
        seed=DCArgs.seed,
        user_goal=DCArgs.blockpush_user_goal,
    )
    step = 0

    scale = 0.5
    ep = 0
    frames = []
    num_vis_episodes = 50

    with actor.agent.eval_mode():
        while step < DCArgs.num_transitions:
            candidate_entries = []
            obs = env.reset()
            done = False
            sum_rewards = 0
            last_ep_step = 0

            # Original resolution is (400 x 600)
            if save_video and ep < num_vis_episodes:
                frames.append(get_frame(env, ep, step, obs, scale=scale))
            while not done:
                exp_act = actor.act(obs)

                if random.random() < DCArgs.randp:
                    action = env.action_space.sample()
                else:
                    action = exp_act

                next_obs, rew, done, _ = env.step(action)
                candidate_entries.append((obs.copy(), exp_act))
                obs = next_obs

                step += 1
                last_ep_step += 1
                sum_rewards += rew

                if save_video and ep < num_vis_episodes:
                    frames.append(
                        get_frame(
                            env,
                            ep,
                            step,
                            obs,
                            rew,
                            sum_rewards,
                            action=action,
                            scale=scale,
                        )
                    )

            if save_video and ep == num_vis_episodes:
                # Log video
                wandb.log(
                    {
                        "video": wandb.Video(
                            np.asarray(frames).transpose(0, 3, 1, 2),
                            fps=30,
                            format="mp4",
                        )
                    }
                )

            # NOTE: only store trajectories that exceed a threshold
            if sum_rewards >= DCArgs.valid_return_threshold:
                # Store data only if the episode achieved sum_rewards >= threshold
                qvals = [None] * len(candidate_entries)  # Backward compatibility
                for entry, qval in zip(candidate_entries, qvals):
                    obs, act = entry
                    replay_buffer.store(obs, act, qval)

                print(
                    f"step: {step} / {DCArgs.num_transitions}\tsum_rewards: {sum_rewards}\tep_len: {last_ep_step}"
                )
                if ep < num_vis_episodes:
                    wandb.log(
                        {
                            "step": step,
                            "sum_rewards": sum_rewards,
                            "ep_len": last_ep_step,
                        }
                    )
            else:
                # Take away steps
                step -= last_ep_step

            ep += 1


if __name__ == "__main__":
    from params_proto.hyper import Sweep

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-file", type=str, default=None, help="sweep file")
    parser.add_argument(
        "-l", "--line-number", type=int, default=0, help="line number of the sweep-file"
    )
    args = parser.parse_args()

    # Set CVD
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        avail_gpus = [0]
        cvd = avail_gpus[args.line_number % len(avail_gpus)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    if args.sweep_file is not None:
        # Obtain kwargs from Sweep
        sweep = Sweep(DCArgs).load(args.sweep_file)
        kwargs = list(sweep)[args.line_number]
        DCArgs._update(kwargs)

    # wandb config, run main function
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="diffusha",
        group="generate_data",
        config={"Args": vars(Args), "DCArgs": vars(DCArgs)},
    )
    main(save_video=True)
    wandb.finish()
