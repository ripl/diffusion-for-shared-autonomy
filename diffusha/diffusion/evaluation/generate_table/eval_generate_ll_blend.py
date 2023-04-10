#!/usr/bin/env python3
"""This file computes statistics from evaluation trajectories saved in files.

Before running this file, trajectories must have been generated with `eval_assistance.py`
"""
import functools
from typing import Sequence, Tuple
import numpy as np
import torch
from diffusha.config.default_args import Args
import csv
import os


@functools.cache
def torch_load(fname):
    return torch.load(fname)


def sort_val_by_return(val):
    new_val = {}
    reward = np.array(val["return"])
    avg_reward = reward.copy()
    ep_len = np.array(val["ep_lengths"])
    if "sum_diffs" in val.keys():
        sum_diffs = np.array(val["sum_diffs"])
    else:
        sum_diffs = np.zeros_like(reward)

    avg_diffs = np.zeros_like(sum_diffs)
    for i in range(reward.shape[0]):
        avg_reward[i] = 100 * reward[i] / ep_len[i]
        avg_diffs[i] = sum_diffs[i] / ep_len[i]

    game_over = np.array(val["ll-game_over"])
    success = np.array(val["ll-success"])
    matrix = np.vstack(
        (reward, avg_reward, ep_len, avg_diffs, sum_diffs, success, game_over)
    )

    # Sort by reward
    matrix = matrix[:, matrix[0].argsort()]

    new_val["return"] = matrix[0, :].tolist()
    new_val["avg_reward"] = matrix[1, :].tolist()
    new_val["ep_lengths"] = matrix[2, :].tolist()
    new_val["avg_diffs"] = matrix[3, :].tolist()
    new_val["sum_diffs"] = matrix[4, :].tolist()
    new_val["success"] = matrix[5, :].tolist()
    new_val["game_over"] = matrix[6, :].tolist()

    return new_val


def get_stats(sequence: Sequence):
    sequence = np.asarray(sequence)
    return sequence.mean(), sequence.std()


if __name__ == "__main__":
    from pathlib import Path
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-proj-name",
        default="takuma-yoneda/diffusha",
        help="Wandb project name to load from",
    )
    parser.add_argument(
        "--wandb-run-id", default="qa3br4pa", help="Wandb run id to load from"
    )
    parser.add_argument("--dir-name", default="0215", help="which exp config")
    parser.add_argument("--env-name", default="LunarLander-v1", help="what env to use")
    args = parser.parse_args()

    if "v5" in args.env_name:
        task = "Landing"
    elif "v1" in args.env_name:
        task = "Reaching"

    # directory = (
    #     Path(Args.results_dir)
    #     / "assistance"
    #     / args.wandb_proj_name
    #     / args.wandb_run_id
    #     / args.dir_name
    # )

    user_config = 'naive_blending'
    directory = Path(Args.pt_dir) / 'results-20230405' / 'assistance' / args.env_name.lower() / user_config

    n = 3
    group_size = 100
    for group in range(n):
        # file_path = (
        #     Path(Args.results_dir)
        #     / "assistance"
        #     / args.wandb_proj_name
        #     / args.wandb_run_id
        #     / f"{task}-{args.dir_name}-{group}.csv"
        # )
        file_path = Path(Args.pt_dir) / 'results-20230405' / 'assistance' / args.env_name.lower() / f'{user_config}-{task}-{group}.csv'
        csvfile = open(file_path, "w")
        writer = csv.writer(csvfile)

        # Traverse the directory and read files one by one
        for f_index, fname in enumerate(directory.iterdir()):
            start_time = time.time()
            print(f"Processing {fname.resolve()}...")
            obj = torch_load(fname)
            actor2assist_log_entry = obj["assisted"]
            actor2log_entry = obj["original"]
            config = obj["hyperparams"]
            print("Hyperaparams:\n", config)
            if config["args"]["num_episodes"] < 100:
                continue

            # fwr = config["Args"]["fwd_diff_ratio"]
            fwr = config["Args"]["blend_alpha"]
            lag_p = config["Args"]["laggy_actor_repeat_prob"]
            noise_p = config["Args"]["noisy_actor_eps"]
            writer.writerow([f"FWR={fwr}P={lag_p}"])
            # import pdb; pdb.set_trace()

            template = (
                "{actor}\t{return_mean:.2f} +/- {return_std:.2f}\t{avg_return_mean:.2f} +/- {avg_return_std:.2f}\t{ep_len_mean:.1f} +/- {ep_len_std:.1f}\t{action_diffs_mean:.1f} +/- {action_diffs_std:.1f}\t"
                "{success:.0f}/{total:.0f}\t{game_over:.0f}/{total:.0f}\t"
            )

            print(
                "Actor\treturn   \tavg_return   \tep_len   \taction_diffs\tsuccess\tgame over (with assistance)"
            )
            writer.writerow(
                [
                    "actor",
                    "reward",
                    "avg_reward",
                    "ep length",
                    "action diffs",
                    "success",
                    "game over",
                    "floating",
                    "Assisted",
                ]
            )
            for actor_name, log_entry in sorted(actor2assist_log_entry.items()):
                # log_entry.keys:
                # ['ll-success', 'll-game_over', 'return', 'ep_lengths', 'return_mean', 'ep_len_mean',
                #  'diffusion_elapsed_mean', 'trajectories', 'action_diffs', 'action_correction_level', 'sum_diffs']
                my_val = {}
                for key, value in log_entry.items():
                    if isinstance(value, list):
                        my_val[key] = value[
                            group * group_size : (group + 1) * group_size
                        ]
                new_val = sort_val_by_return(my_val)
                print(
                    template.format(
                        actor=actor_name,
                        return_mean=get_stats(new_val["return"])[0],
                        return_std=get_stats(new_val["return"])[1],
                        avg_return_mean=get_stats(new_val["avg_reward"])[0],
                        avg_return_std=get_stats(new_val["avg_reward"])[1],
                        ep_len_mean=get_stats(new_val["ep_lengths"])[0],
                        ep_len_std=get_stats(new_val["ep_lengths"])[1],
                        action_diffs_mean=get_stats(new_val["sum_diffs"])[0],
                        action_diffs_std=get_stats(new_val["sum_diffs"])[1],
                        success=int(np.array(new_val["success"]).sum()),
                        total=len(new_val["success"]),
                        game_over=int(np.array(new_val["game_over"]).sum()),
                    )
                )

                writer.writerow(
                    [
                        actor_name,
                        "%.3f" % get_stats(new_val["return"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["return"])[1],
                        "%.3f" % get_stats(new_val["avg_reward"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["avg_reward"])[1],
                        "%.3f" % get_stats(new_val["ep_lengths"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["ep_lengths"])[1],
                        "%.3f" % get_stats(new_val["sum_diffs"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["sum_diffs"])[1],
                        f"{int(np.array(new_val['success']).sum())}/{len(new_val['success'])}",
                        f"{int(np.array(new_val['game_over']).sum())}/{len(new_val['success'])}",
                        f"{len(new_val['success'])-int(np.array(new_val['game_over']).sum())-int(np.array(new_val['success']).sum())}/{len(new_val['success'])}",
                    ]
                )

            print(
                "Actor\treturn\tavg_return\tep_len\taction_diffs\tsuccess\tgame over (without assistance)"
            )
            writer.writerow(
                [
                    "actor",
                    "reward",
                    "avg_reward",
                    "ep length",
                    "action diffs",
                    "success",
                    "game over",
                    "floating",
                    "Not Assisted",
                ]
            )
            for actor_name, log_entry in sorted(actor2log_entry.items()):
                # log_entry.keys():
                # (['ll-success', 'll-game_over', 'return', 'ep_lengths', 'return_mean', 'ep_len_mean',
                #   'diffusion_elapsed_mean', 'trajectories'])
                my_val = {}
                for key, value in log_entry.items():
                    if isinstance(value, list):
                        my_val[key] = value[
                            group * group_size : (group + 1) * group_size
                        ]
                new_val = sort_val_by_return(my_val)
                print(
                    template.format(
                        actor=actor_name,
                        return_mean=get_stats(new_val["return"])[0],
                        return_std=get_stats(new_val["return"])[1],
                        avg_return_mean=get_stats(new_val["avg_reward"])[0],
                        avg_return_std=get_stats(new_val["avg_reward"])[1],
                        ep_len_mean=get_stats(new_val["ep_lengths"])[0],
                        ep_len_std=get_stats(new_val["ep_lengths"])[1],
                        action_diffs_mean=get_stats(new_val["sum_diffs"])[0],
                        action_diffs_std=get_stats(new_val["sum_diffs"])[1],
                        success=int(np.array(new_val["success"]).sum()),
                        total=len(new_val["success"]),
                        game_over=int(np.array(new_val["game_over"]).sum()),
                    )
                )

                writer.writerow(
                    [
                        actor_name,
                        "%.3f" % get_stats(new_val["return"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["return"])[1],
                        "%.3f" % get_stats(new_val["avg_reward"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["avg_reward"])[1],
                        "%.3f" % get_stats(new_val["ep_lengths"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["ep_lengths"])[1],
                        "%.3f" % get_stats(new_val["sum_diffs"])[0]
                        + "+-"
                        + "%.3f" % get_stats(new_val["sum_diffs"])[1],
                        f"{int(np.array(new_val['success']).sum())}/{len(new_val['success'])}",
                        f"{int(np.array(new_val['game_over']).sum())}/{len(new_val['success'])}",
                        f"{len(new_val['success']) - int(np.array(new_val['game_over']).sum()) - int(np.array(new_val['success']).sum())}/{len(new_val['success'])}",
                    ]
                )

            writer.writerow("\n")
            end_time = time.time()
            print("Running time for one file:", end_time - start_time)
            print(f"Process Finished: Group: {group + 1}/{n},File:{f_index + 1}")

        csvfile.close()
