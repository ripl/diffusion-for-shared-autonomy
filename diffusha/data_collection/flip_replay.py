#!/usr/bin/env python3

"""This script reads from replay buffer of block push env, and flip obs and action."""

import numpy as np
import torch
from pathlib import Path
from diffusha.data_collection.generate_data import ReplayBuffer


if __name__ == "__main__":
    import argparse
    from diffusha.data_collection.env import make_env

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="source replay directory")
    parser.add_argument("tgt_dir", help="target replay directory")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    args = parser.parse_args()

    assert Path(args.src_dir).is_dir()
    assert not Path(args.tgt_dir).is_dir()

    env_name = "BlockPushMultimodal-v1"
    sample_env = make_env(env_name, test=True)
    obs_size = sample_env.observation_space.low.size
    act_size = sample_env.action_space.low.size
    replay = ReplayBuffer(args.src_dir, obs_size, act_size)

    # Create target directory
    Path(args.tgt_dir).mkdir(mode=0o775, parents=True, exist_ok=False)

    # Convert files one by one from replay
    for fname in replay.directory.iterdir():
        print(f"loading {fname} from {replay.directory}...")
        buff = replay._file_cache[fname]

        """
        obs:
        - obs[0:2] -- block_translation
        - obs[2:3] -- block_orientation
        - obs[3:5] -- ee_translation
        - obs[5:7] -- ee_target_translation

        act:
        - act[0:2] -- x-y ee diff
        """

        # Flip obs and action x axis
        flip_dims = [0, 3, 5] + [7]
        buff[:, flip_dims] = -buff[:, flip_dims]

        # Flip block orientation
        block_ori = buff[:, 2]
        new_block_ori = block_ori + np.pi
        new_block_ori[np.pi <= new_block_ori] -= 2 * np.pi
        buff[:, 2] = new_block_ori

        tgt_fname = Path(args.tgt_dir) / (fname.stem + ".pkl")
        print(f"saving chunk to {str(tgt_fname)}")
        torch.save(buff, tgt_fname)
