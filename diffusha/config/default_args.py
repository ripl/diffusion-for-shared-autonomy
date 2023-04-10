#!/usr/bin/env python3
from params_proto import ParamsProto
import os
proj_root = os.environ.get('RMX_CODE_DIR', '')

class Args(ParamsProto):
    env_name = 'LunarLander-v1'
    dataset_envs = None

    # DDPM configuration
    num_diffusion_steps = 50
    beta_min = 1e-4
    beta_max = 0.26
    beta_schedule = 'sigmoid'
    ddpm_model_path = '/data/ddpm'

    num_training_steps = 100_000
    eval_every = 2000
    save_every = 2000

    # Data directories
    lunarlander_data_dir = '/data/replay/lunarlander'
    pointmaze_data_dir = '/data/replay/pointmaze'
    blockpush_data_dir = '/data/replay/blockpush'

    # Stores evaluation results
    results_dir = '/data/results'
    pt_dir = '/data'

    randp = 0.
    seed = 0

    # Used in evaluation
    fwd_diff_ratio = 0.4
    laggy_actor_repeat_prob = 0.8
    noisy_actor_eps = 0.8

    batch_size = 4096

    # Temporary directory that stores SAC model checkpoints
    sac_model_dir = '/data/sac'
