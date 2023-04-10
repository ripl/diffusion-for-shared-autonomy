#!/usr/bin/env python3
import os
from pathlib import Path
import wandb
import torch

from diffusha.data_collection.env import make_env

from diffusha.config.default_args import Args


def prepare_diffusha(
    config,
    model_dir,
    step,
    env_name,
    fwd_diff_ratio,
    laggy_actor_repeat_prob,
    noisy_actor_eps,
):
    """Load hyperparmeters from wandb, and load a pytorch model from birch/elm, and return an instantiated diffusion model."""
    from diffusha.diffusion.ddpm import DiffusionModel, DiffusionCore

    Args._update(config)

    # Overwrite these parameters
    Args.fwd_diff_ratio = fwd_diff_ratio
    Args.laggy_actor_repeat_prob = laggy_actor_repeat_prob
    Args.noisy_actor_eps = noisy_actor_eps

    if "LunarLander" in env_name:
        sample_env = make_env(env_name, seed=0, split_obs=True, test=True)
        copilot_obs_space = sample_env.copilot_observation_space
        act_space = sample_env.action_space
        obs_size = copilot_obs_space.low.size
        act_size = act_space.low.size
    else:
        sample_env = make_env(env_name, seed=0, test=True)
        obs_space = sample_env.observation_space
        act_space = sample_env.action_space
        obs_size = obs_space.low.size
        act_size = act_space.low.size

    # Load diffusion model
    diffusion = DiffusionModel(
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=Args.num_diffusion_steps,
        input_size=(obs_size + act_size),
        beta_schedule=Args.beta_schedule,
        beta_min=Args.beta_min,
        beta_max=Args.beta_max,
        cond_dim=obs_size,
    )

    # load diffusion model from Arguments
    # job_name = wandb_proj_name.split("/")[-1]
    # model_path = (
    #     Path(Args.ddpm_model_path) / job_name / wandb_run_id / f"step_{step:08d}.pt"
    # )
    model_path = model_dir / f"step_{step:08d}.pt"
    print(f"Loading a model from {str(model_path)}")
    checkpoint = torch.load(model_path)
    diffusion.model.load_state_dict(checkpoint["model"])
    diffusion.model.eval()

    return diffusion
