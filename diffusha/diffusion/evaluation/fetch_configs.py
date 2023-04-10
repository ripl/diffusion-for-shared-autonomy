#!/usr/bin/env python3
from copy import deepcopy
import wandb
import json

if __name__ == '__main__':
    env2wandb = {
        "maze2d-simple-two-goals-v0": "takuma-yoneda/diffusha/ulh2ovtj",
        "LunarLander-v1": "takuma-yoneda/diffusha/24gbtia4",
        "LunarLander-v5": "takuma-yoneda/diffusha/qa3br4pa",
        "BlockPushMultimodal-v1": "takuma-yoneda/diffusha/a6tc22h8"
    }

    # Locally save a mapping from env to configurations
    env2config = {}
    for env, wandb_path in env2wandb.items():
        wandb_api = wandb.Api()
        run = wandb_api.run(wandb_path)
        env2config[env] = deepcopy(run.config)

    # Writing to sample.json
    with open("configs.json", "w") as f:
        json.dump(env2config, f)
