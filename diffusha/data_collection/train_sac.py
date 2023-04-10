#!/usr/bin/env python3
"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.
This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
"""
from __future__ import annotations
import os
import argparse
import logging
import sys

import gym
import gym.wrappers
from diffusha.config.default_args import Args

import pfrl
from pathlib import Path
from pfrl import experiments, utils
import wandb
from .env import make_env
from .env.eval_hook import WandBLogger
from .utils.agent import get_agent


TIME_LIMIT = 1000


def main(args, outdir):
    logging.basicConfig(level=args.log_level)

    outdir = experiments.prepare_output_dir(args, outdir, argv=sys.argv)
    print("Output files are saved in {}".format(outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    sample_env = make_env(args.env_name, test=False, seed=args.seed)
    # timestep_limit = sample_env.spec.max_episode_steps
    timestep_limit = TIME_LIMIT

    agent = get_agent(
        sample_env, args.policy_output_scale, args.batch_size, args.replay_start_size
    )

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_env(args.env_name, test=True, seed=args.seed),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
        import json

        with open(os.path.join(outdir, "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(args.env_name, test=False, seed=args.seed),
            eval_env=make_env(args.env_name, test=True, seed=args.seed),
            outdir=outdir,
            checkpoint_freq=args.checkpoint_freq,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            evaluation_hooks=[WandBLogger()],
        )


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        type=str,
        default="LunarLander-v3",
        help="LunarLander-v3 for floating around reward shaping, v2 for original.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--policy-output-scale",
        type=float,
        default=1.0,
        help="Weight initialization scale of policy output.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20_000,
        help="Weight initialization scale of policy output.",
    )
    args = parser.parse_args()

    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project=f"pfrl-{args.env_name}",
        config=vars(args),
    )

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        avail_gpus = [0]
        # gpu_id = 0 if args.line_number is None else args.line_number % len(avail_gpus)
        gpu_id = 0
        cvd = avail_gpus[gpu_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    outdir = (
        Path(Args.sac_model_dir)
        / args.env_name.lower()
        / wandb.run.project
        / wandb.run.id
    )
    outdir.mkdir(mode=0o775, parents=True, exist_ok=True)

    main(args, outdir)
    wandb.finish()
