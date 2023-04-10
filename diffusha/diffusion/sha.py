#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import torch
import argparse
from diffusha.data_collection.env import make_env
from diffusha.data_collection.train_sac import TIME_LIMIT
from .ddpm import Diffusion
import wandb
from diffusha.config.default_args import Args
from diffusha.diffusion.eval import eval_diffusion


def main(**kwargs):
    Args._update(kwargs)
    sample_env = make_env(test=True, seed=0, time_limit=TIME_LIMIT)
    obs_size = sample_env.observation_space.low.size
    act_size = sample_env.action_space.low.size

    diffusion = Diffusion(
        num_diffusion_steps=Args.num_diffusion_steps, state_dim=obs_size + act_size
    )

    # TODO: Use a shared function that maps Args to a unique path
    model_path = (
        Path(Args.ddpm_model_path)
        / f"df-{Args.num_diffusion_steps}-train-{Args.num_training_steps}-model.pt"
    )

    checkpoint = torch.load(model_path)
    diffusion.model.load_state_dict(checkpoint["model"])

    log_entry = eval_diffusion(diffusion)
    wandb.log(log_entry)


if __name__ == "__main__":
    from params_proto.hyper import Sweep

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", type=str, help="sweep file")
    parser.add_argument(
        "-l", "--line-number", type=int, help="line number of the sweep-file"
    )
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    sweep = Sweep(Args).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    num_gpus = 1
    cvd = args.line_number % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    Args._update(kwargs)

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="diffusha-diffusion-eval",
        config=vars(Args),
    )

    main()

    wandb.finish()
