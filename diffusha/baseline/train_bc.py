"""MLP for naive Behavior Cloning"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb
from diffusha.data_collection.env import make_env
from diffusha.data_collection.train_sac import TIME_LIMIT
from diffusha.config.default_args import Args
from diffusha.diffusion.train import ExpertTransitionDataset, MultiExpertTransitionDataset
from diffusha.actor.base import Actor
from diffusha.diffusion.evaluation.eval import evaluate

device='cuda'

# HARDCODED
bc_policy_dir = '/data/bc_policy'
data_root_dir = '/data/hosted_data/replay'


class BCActor(Actor):
    def __init__(self, model, action_space):
        self.model = model
        self.action_space = action_space

    def act(self, obs):
        obs = torch.as_tensor(obs).unsqueeze(0).to(device)  # Add batch dim
        with torch.no_grad():
            action = self.model(obs)
        action = action.squeeze().cpu().numpy()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action


def get_model(obs_size, action_size, policy_output_scale=1.):
    policy = nn.Sequential(
        nn.Linear(obs_size, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_size),
    )
    nn.init.xavier_uniform_(policy[0].weight)
    nn.init.xavier_uniform_(policy[2].weight)
    nn.init.xavier_uniform_(policy[4].weight, gain=policy_output_scale)

    return policy


def main(**kwargs):
    Args._update(kwargs)

    batch_size = 8192
    log_freq = 50
    # eval_freq = 1000

    sample_env = make_env(Args.env_name, test=False, split_obs=True, seed=Args.seed)
    if hasattr(sample_env, 'copilot_observation_space'):
        obs_space = sample_env.copilot_observation_space
    else:
        obs_space = sample_env.observation_space
    act_space = sample_env.action_space
    obs_size = obs_space.low.size
    act_size = act_space.low.size
    model = get_model(obs_size, act_size)
    model.to(device)

    # actor = BCActor(model, sample_env.action_space)

    if 'LunarLander' in Args.env_name:
        simple_envname = 'lunarlander'
        level = Args.env_name.split('-')[-1]
        data_dir = Path(data_root_dir) / simple_envname / level / "randp_0.0"
        model_path = Path(bc_policy_dir) / simple_envname / level / 'bc_model.pt'
    elif 'BlockPush' in Args.env_name:
        simple_envname = 'blockpush'
        data_dir = Path(data_root_dir) / simple_envname / "randp_0.0"
        model_path = Path(bc_policy_dir) / simple_envname / 'bc_model.pt'
    else:
        raise ValueError()


    dataset = ExpertTransitionDataset(data_dir, obs_size, act_size)
    # dataset = MultiExpertTransitionDataset(data_dir, obs_size, act_size)
    loader = iter(DataLoader(dataset, batch_size=batch_size, num_workers=0))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    make_eval_env = lambda: make_env(Args.env_name, test=True, seed=Args.seed, time_limit=TIME_LIMIT)

    losses = []
    for step in range(Args.num_training_steps):
        batch = next(loader).to(device)
        # Separate observation and action
        # obs_batch = batch[..., :obs_size]
        # act_batch = batch[..., obs_size:]

        # Luzhe: Remove target dimension [obs,target,act]
        obs_batch = batch[..., :obs_size]
        act_batch = batch[..., -act_size:]

        # Calculate loss
        pred_act = model(obs_batch)

        loss = F.mse_loss(pred_act, act_batch)
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optimizer.step()

        losses.append(loss.item())
        if step > 0 and step % log_freq == 0:
            wandb.log({'train/loss': sum(losses) / len(losses),
                       'step': step})
            losses = []

        # if step % eval_freq == 0:
        #     model.eval()
        #     log_entry = evaluate(make_eval_env, actor)
        #     model.train()
        #     wandb.log({
        #         **{f'eval/{key}': val for key, val in log_entry.items()},
        #         'step': step,
        #     })

    simple_envname = Args.env_name.split(':')[-1].lower()
    # fpath = f'/data/bc_policy/{simple_envname}'
    print(f"Saving the BC model to {model_path}")
    model_path.parent.mkdir(mode=0o775, exist_ok=True, parents=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)
    print(f"model saved at {model_path}")


if __name__ == '__main__':
    import os
    import argparse
    from params_proto.hyper import Sweep
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-file", type=str,
                        help="sweep file")
    parser.add_argument("-l", "--line-number",
                        type=int, help="line number of the sweep-file")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    sweep = Sweep(Args).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        num_gpus = 1
        cvd = args.line_number % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    Args._update(kwargs)

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project='diffusha-diffusion-bc-randp',
        config=vars(Args)
    )

    main()
    wandb.finish()
