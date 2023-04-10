#!/usr/bin/env python3
import torch
import time
from torch.distributions import Normal
from diffusha.config.default_args import Args
from contextlib import contextmanager

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def initial_expert_agent(make_env, model_dir):
    from diffusha.data_collection.utils.agent import get_agent
    # TEMP: COPY FROM TAKUMA: NOTE: Hardcoded!!
    policy_output_scale = 1.0
    batch_size = 256
    replay_start_size = 10_000

    sample_env = make_env()
    agent = get_agent(sample_env, policy_output_scale, batch_size, replay_start_size)
    agent.load(model_dir)  # Trained for landing
    agent.eval_mode()

    return agent
