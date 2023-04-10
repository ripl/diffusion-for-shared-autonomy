#!/usr/bin/env python3

import torch
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent.parent.resolve()))

from diffusha.diffusion.utils import make_beta_schedule

timesteps = [10, 20, 30, 50, 100]
beta_mins = [1e-4]
beta_maxs = [0.02, 0.1, 0.2, .4, .51]

for beta_min in beta_mins:
    for beta_max in beta_maxs:
        for T in timesteps:
            betas = make_beta_schedule(schedule='sigmoid', n_timesteps=T, start=1e-4, end=beta_max)
            alphas = 1 - betas
            alphas_prod = torch.cumprod(alphas, 0)
            alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
            alphas_bar_sqrt = torch.sqrt(alphas_prod)
            one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
            one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

            sigmas = (one_minus_alphas_bar_sqrt / alphas_bar_sqrt)
            print(f'timesteps: {T}\tbeta_min: {beta_min}\tbeta_max: {beta_max}\tsigma_min: {sigmas[0]}\tsigma_max: {sigmas[-1]}')
