from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import random
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time


from .utils import make_beta_schedule, extract
from .models import ConditionalModel, EMA
from diffusha.config.default_args import Args
import wandb


class DiffusionCore:
    def __init__(self) -> None:
        from pfrl import utils
        from diffusha.data_collection.config.default_args import DCArgs

        utils.set_random_seed(DCArgs.seed)

    def noise_estimation_loss(
        self, diffusion: DiffusionModel, x_0: torch.Tensor, cond_dim: int = 0
    ) -> torch.Tensor:
        batch_size = x_0.shape[0]
        # Select a random step for each example
        t = torch.randint(0, diffusion.num_diffusion_steps, size=(batch_size // 2 + 1,))
        # Sample timestemps in a symmetric way!
        t = torch.cat([t, diffusion.num_diffusion_steps - t - 1], dim=0)[
            :batch_size
        ].long()
        t = t.to(diffusion.device)

        x_t, e = self.diffuse(diffusion, x_0, t, cond_dim=cond_dim)

        output = diffusion.model(x_t, t)
        err = e - output

        return err.square().mean()

    @torch.no_grad()
    def diffuse(
        self,
        diffusion: DiffusionModel,
        x_0: torch.Tensor,
        t: torch.Tensor,
        cond_dim: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion

        Args:
          - x_0 (torch.Tensor): shape: (batch_size, **remaining-dimensions)
          - t (torch.Tensor): shape: (times, )
        """
        assert len(x_0.shape) > 1, "x_0 must have batch dimension"
        assert len(t.shape) == 1, f"Wrong shape for t: {t.shape}"

        x_0 = x_0.to(diffusion.device)
        t = t.to(diffusion.device)

        # x0 multiplier
        a = extract(diffusion.alphas_bar_sqrt, t, x_0)
        # eps multiplier
        am1 = extract(diffusion.one_minus_alphas_bar_sqrt, t, x_0)
        e = torch.randn_like(x_0)

        x_t = x_0 * a + e * am1

        # Optionally remove noise on conditional dimensions
        if cond_dim > 0:
            x_t[..., :cond_dim] = x_0[..., :cond_dim]

            # NOTE: This is interesting
            # If we do not remove the noise, the model needs to predict noises for the conditional dimensions as well.
            # This may sound nonsense, but this could encourage the model to learn distribution of the conditional dimensions.
            # Also, effectively the support of input distribution enlarges (i.e., individual data sample + noise)
            # e[
            #     ..., :cond_dim
            # ] = 0.0  # BUG: This was not specified on the original version

        return x_t, e


class DiffusionModel:
    def __init__(
        self,
        diffusion_core: DiffusionCore,
        num_diffusion_steps: int,
        input_size: int,
        beta_schedule: str,
        beta_min: float,
        beta_max: float,
        cond_dim: int = 0,
    ) -> None:
        self.diffusion_core = diffusion_core
        self.cond_dim = cond_dim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_diffusion_steps, start=1e-5, end=1e-2)
        betas = make_beta_schedule(
            schedule=beta_schedule,
            n_timesteps=num_diffusion_steps,
            start=beta_min,
            end=beta_max,
        )
        self.betas = betas.to(device)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0).to(device)
        self.alphas_prod_p = torch.cat(
            [torch.tensor([1], device=device).float(), self.alphas_prod[:-1]], 0
        )
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        print("sigma_max", (self.one_minus_alphas_bar_sqrt / self.alphas_bar_sqrt)[-1])

        self.model = ConditionalModel(num_diffusion_steps, input_size=input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.num_diffusion_steps = num_diffusion_steps
        self.predict_epsilon = True

        self.model.to(device)

        # Create EMA model
        ema = EMA(0.9)
        ema.register(self.model)
        self.ema = ema

        # configurations
        self.device = device
        self.step = 0

    @torch.no_grad()
    def p_sample(self, x, t):
        x = x.float()
        x = x.to(self.device)
        t = torch.tensor([t]).to(self.device)
        # Factor to the model output
        eps_factor = (1 - extract(self.alphas, t, x)) / extract(
            self.one_minus_alphas_bar_sqrt, t, x
        )

        # Model output
        if self.predict_epsilon:
            eps_theta = self.model(x, t)
            # Final values
            mean = (1 / extract(self.alphas, t, x).sqrt()) * (
                x - (eps_factor * eps_theta)
            )
        else:
            # NOTE: Takuma -- not confident at all if this is correct
            mean = self.model(x, t)

        # Generate z
        z = torch.randn_like(x)
        # Fixed sigma
        sigma_t = extract(self.betas, t, x).sqrt()
        sample = mean + sigma_t * z

        return sample

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        start_x: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        naive_cond: bool = False,
    ):
        """Peforms conditional sampling (if cond is not None)

        This assumes cond Tensor corresponds to the first cond.shape[0] dimension of diffusion state space.
        """

        def apply_naive_condition(x: torch.Tensor, cond: torch.Tensor, timestep: int):
            """Simply replace a part of x with cond"""
            assert len(cond.shape) in [1, 2], f"Wrong shape on cond: {cond.shape}"
            assert len(cond.shape) == len(
                x.shape
            ), "Condition shape does not match sample shape"
            cond_dim = cond.shape[-1]
            x[..., :cond_dim] = cond
            return x

        def apply_condition(x: torch.Tensor, cond: torch.Tensor, timestep: int):
            """A better way: Replace a part of x with noisy cond tensor"""
            assert len(cond.shape) in [1, 2], f"Wrong shape on cond: {cond.shape}"
            assert len(cond.shape) == len(
                x.shape
            ), f"Condition shape does not match sample shape"
            cond_dim = cond.shape[-1]

            if len(cond.shape) == 1:
                naive_x = torch.zeros_like(x).unsqueeze(0)  # Add batch dim
            else:
                naive_x = torch.zeros_like(x)  # Add batch dim

            naive_x[
                ..., :cond_dim
            ] = cond  # TODO: The values of the rest of dimensions shouldn't matter, but should double check as Luzhe mentionied
            naive_x = naive_x.to(self.device)
            k = torch.as_tensor([timestep]).to(self.device)
            noisy_x, _ = self.diffuse(naive_x, k)
            noisy_x = noisy_x.squeeze(0)  # Remove batch dim
            _cond = noisy_x[..., :cond_dim]

            x[..., :cond_dim] = _cond
            return x

        _apply_cond = apply_naive_condition if naive_cond else apply_condition

        # Use start_x if specified
        if start_x is not None:
            assert shape == start_x.shape
            x = start_x
        else:
            x = torch.randn(shape)

        x_seq = []
        for k in reversed(range(self.num_diffusion_steps)):
            if cond is not None:
                x = _apply_cond(x, cond, k)
            x_seq.append(x.detach().cpu())
            x = self.p_sample(x, k)

        # Don't forget to append the last one
        x_seq.append(x.detach().cpu())

        return x, x_seq

    def noise_estimation_loss(self, x_0: torch.Tensor):
        return self.diffusion_core.noise_estimation_loss(
            self, x_0, cond_dim=self.cond_dim
        )

    @torch.no_grad()
    def diffuse(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.diffusion_core.diffuse(self, x_0, t, cond_dim=self.cond_dim)

    def train_step(self, batch: torch.Tensor, step: int):
        """Run one step of training."""

        loss = self.noise_estimation_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.ema.update(self.model)

        return loss

    def get_jobdir(self):
        model_dir = Path(Args.ddpm_model_path) / wandb.run.project / wandb.run.id
        model_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
        return model_dir

    def save_model(self, step):
        model_path = self.get_jobdir() / f"step_{step:08d}.pt"
        print(f"Saving model to {model_path}... ")
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optim": self.optimizer.state_dict(),
                "step": step,
            },
            model_path,
        )
        print(f"Saving model to {model_path}... done.")


class Trainer:
    def __init__(
        self,
        diffusion: DiffusionModel,
        obs_size: int,
        act_size: int,
        save_every: int = 10_000,
        eval_every: int = 5000,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.diffusion = diffusion

        self.log_freq = 100
        self.eval_every = eval_every
        self.save_every = save_every

        self.obs_size = obs_size
        self.act_size = act_size

        # configurations
        self.device = device

        if self.eval_every <= 0:
            print("eval_every is set to <= 0. Models will not be saved!!")

    def train(
        self,
        data_iter,
        make_eval_env: Optional[Callable],
        num_training_steps=100_000,
        eval_assistance: bool = False,
    ):
        sample_env = make_eval_env() if make_eval_env is not None else None

        losses = []
        for step in range(num_training_steps):
            batch = next(data_iter).to(self.device)
            loss = self.diffusion.train_step(batch, step)

            if self.save_every > 0 and step % self.save_every == 0:
                self.diffusion.save_model(step)

            losses.append(loss.item())

            # Print loss
            if step % self.log_freq == 0:
                logs = {"loss": np.mean(losses), "step": step}
                wandb.log({**logs})
                losses = []

            # Evaluation
            if sample_env is not None and step % self.eval_every == 0:
                from diffusha.diffusion.evaluation.eval_assistance import (
                    eval_assisted_actors,
                )
                from diffusha.diffusion.evaluation.eval import (
                    eval_diffusion,
                    pointmaze_record_traj,
                )

                def get_expert_agent(envname):
                    from diffusha.diffusion.utils import initial_expert_agent
                    from diffusha.actor.waypoint_controller import WaypointActor

                    if "maze2d" in envname:
                        # Maze2d goal
                        return WaypointActor(obs_space, act_space, sample_env)
                    elif "LunarLander" in envname:
                        from diffusha.data_collection.config.default_args import DCArgs

                        # LunarLander
                        lvl = envname.split("-")[-1]
                        lvl2modeldir = {
                            "v1": DCArgs.sac_v1_model_dir,
                            "v2": DCArgs.sac_v2_model_dir,
                            "v3": DCArgs.sac_v3_model_dir,
                            "v4": DCArgs.sac_v4_model_dir,
                            "v5": DCArgs.sac_v5_model_dir,
                        }
                        return initial_expert_agent(make_eval_env, lvl2modeldir[lvl])
                    elif "Push" in envname:
                        modeldir = Path(Args.sac_model_dir) / Args.pushenv_model_dir
                        return initial_expert_agent(make_eval_env, modeldir)
                    else:
                        raise ValueError(f"Unknown env name: {envname}\t{sample_env}")

                obs_space = sample_env.observation_space
                act_space = sample_env.action_space
                name = sample_env.unwrapped.spec.name

                expert_agent = get_expert_agent(name)

                # Only for the first time
                if step == 0:
                    from diffusha.diffusion.evaluation.eval_assistance import (
                        eval_original_actors,
                    )

                    print("Evaluating original actors...")
                    log_entry = eval_original_actors(
                        make_eval_env, expert_agent, num_episodes=20, histogram=True
                    )
                    # Dirty but not sure how to use wandb properly in this case...
                    wandb.log(
                        {
                            **{
                                f"{actor}/{key}": val
                                for actor in log_entry.keys()
                                for key, val in log_entry[actor].items()
                                if (
                                    key not in ["trajectories"]
                                    and not isinstance(val, list)
                                )
                            },
                            "step": step,
                        }
                    )

                print("Evaluating diffusion actor...")
                log_entry = eval_diffusion(
                    self.diffusion, make_eval_env, save_video=False, use_vector_env=True
                )
                print("Evaluating diffusion actor...done")

                if eval_assistance:
                    print("Evaluating assisted actors...")
                    _log_entry = eval_assisted_actors(
                        self.diffusion,
                        make_eval_env,
                        expert_agent,
                        fwd_diff_ratio=Args.fwd_diff_ratio,
                        num_episodes=20,
                        histogram=True,
                        actor_list=["expert", "noisy"],
                        use_vector_env=True,
                    )

                    # Prefix with "assisted" if necessary
                    _log_entry = {
                        f"assisted-{actor}": val for actor, val in _log_entry.items()
                    }
                    log_entry = {**log_entry, **_log_entry}
                    print("Evaluating assisted actors...done")

                # Dirty but not sure how to use wandb properly in this case...
                wandb.log(
                    {
                        **{
                            f"{actor}/{key}": val
                            for actor in log_entry.keys()
                            for key, val in log_entry[actor].items()
                            if (
                                key not in ["trajectories"]
                                and not isinstance(val, list)
                            )
                        },
                        "step": step,
                    }
                )

                # Run preset episodes (specific start positions), and record its trajectories in csv
                if "maze2d" in Args.env_name:
                    traj_entry = pointmaze_record_traj(
                        self.diffusion,
                        make_eval_env,
                        traj_save_dir=self.diffusion.get_jobdir(),
                        step=step,
                    )

                    wandb.log(
                        {
                            "eval_naive/traj": traj_entry["naive_traj"],
                            "eval/traj": traj_entry["traj"],
                            "step": step,
                        }
                    )

        # Save the final model
        if self.save_every > 0:
            self.diffusion.save_model(step)
