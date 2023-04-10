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
import time


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

    correct = np.array(val["bp-correct-goal"])
    wrong = np.array(val["bp-wrong-goal"])
    timeout = np.array(val["bp-time-out"])

    matrix = np.vstack(
        (reward, avg_reward, ep_len, avg_diffs, sum_diffs, correct, wrong, timeout)
    )

    # Sort by reward
    matrix = matrix[:, matrix[0].argsort()]

    new_val["return"] = matrix[0, :].tolist()
    new_val["avg_reward"] = matrix[1, :].tolist()
    new_val["ep_lengths"] = matrix[2, :].tolist()
    new_val["avg_diffs"] = matrix[3, :].tolist()
    new_val["sum_diffs"] = matrix[4, :].tolist()
    new_val["correct"] = matrix[5, :].tolist()
    new_val["wrong"] = matrix[6, :].tolist()
    new_val["time_out"] = matrix[7, :].tolist()

    return new_val


def get_stats(sequence: Sequence):
    sequence = np.asarray(sequence)
    return sequence.mean(), sequence.std()


if __name__ == "__main__":
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-proj-name",
        default="takuma-yoneda/diffusha",
        help="Wandb project name to load from",
    )
    parser.add_argument(
        "--wandb-run-id", default="a6tc22h8", help="Wandb run id to load from"
    )
    parser.add_argument("--dir-name", default="0215", help="which exp config")
    parser.add_argument(
        "--env-name", default="BlockPushMultimodal-v1", help="what env to use"
    )
    args = parser.parse_args()

    if "Push" in args.env_name:
        task = "BP"
    else:
        raise f"Unknown Env: {args.env_name}"

    # This directory contains evaluation results across different gamma (forward diffusion ratio)
    directory = (
        Path(Args.results_dir)
        / "assistance"
        / args.wandb_proj_name
        / args.wandb_run_id
        / args.dir_name
    )

    n = 3
    group_size = 100
    for group in range(n):
        file_path = (
            Path(Args.results_dir)
            / "assistance"
            / args.wandb_proj_name
            / args.wandb_run_id
            / f"{task}-{args.dir_name}-{group}.csv"
        )
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
            print(f"Hyperaparams:\n", config)
            if config["args"]["num_episodes"] < 100:
                continue

            fwr = config["Args"]["fwd_diff_ratio"]
            lag_p = config["Args"]["laggy_actor_repeat_prob"]
            noise_p = config["Args"]["noisy_actor_eps"]
            writer.writerow([f"FWR={fwr}P={lag_p}"])

            template = (
                "{actor}\t{return_mean:.2f} +/- {return_std:.2f}\t{avg_return_mean:.2f} +/- {avg_return_std:.2f}\t{ep_len_mean:.1f} +/- {ep_len_std:.1f}\t{action_diffs_mean:.1f} +/- {action_diffs_std:.1f}\t"
                "{correct:.0f}/{total:.0f}\t{wrong:.0f}/{total:.0f}\t{timeout:.0f}/{total:.0f}\t"
            )

            print(
                "Actor\treturn\tavg_return\tep_len\taction_diffs\tcorrect\twrong\ttimeout (with assistance)"
            )
            writer.writerow(
                [
                    "actor",
                    "reward",
                    "avg_reward",
                    "ep length",
                    "action diffs",
                    "left",
                    "right",
                    "timeout",
                    "Assisted",
                ]
            )
            for actor_name, log_entry in sorted(actor2assist_log_entry.items()):
                # log_entry.keys:
                # ['bp-correct-goal', 'bp-wrong-goal', 'bp-time-out', 'return', 'ep_lengths', 'return_mean',
                #  'ep_len_mean', 'diffusion_elapsed_mean', 'trajectories', 'action_diffs', 'action_correction_level',
                #  'sum_diffs'])
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
                        correct=int(np.array(new_val["correct"]).sum()),
                        total=len(new_val["correct"]),
                        wrong=int(np.array(new_val["wrong"]).sum()),
                        timeout=int(np.array(new_val["time_out"]).sum()),
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
                        f"{int(np.array(new_val['correct']).sum())}/{len(new_val['correct'])}",
                        f"{int(np.array(new_val['wrong']).sum())}/{len(new_val['correct'])}",
                        f"{int(np.array(new_val['time_out']).sum())}/{len(new_val['correct'])}",
                    ]
                )

            print(
                "Actor\treturn\tavg_return\tep_len\taction_diffs\tcorrect\twrong\ttimeout (without assistance)"
            )
            writer.writerow(
                [
                    "actor",
                    "reward",
                    "avg_reward",
                    "ep length",
                    "action diffs",
                    "left",
                    "right",
                    "timeout",
                    "Not Assisted",
                ]
            )
            for actor_name, log_entry in sorted(actor2log_entry.items()):
                # log_entry.keys():
                # (['bp-correct-goal', 'bp-wrong-goal', 'bp-time-out', 'return', 'ep_lengths', 'return_mean',
                #   'ep_len_mean', 'diffusion_elapsed_mean', 'trajectories'])
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
                        correct=int(np.array(new_val["correct"]).sum()),
                        total=len(new_val["correct"]),
                        wrong=int(np.array(new_val["wrong"]).sum()),
                        timeout=int(np.array(new_val["time_out"]).sum()),
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
                        f"{int(np.array(new_val['correct']).sum())}/{len(new_val['correct'])}",
                        f"{int(np.array(new_val['wrong']).sum())}/{len(new_val['correct'])}",
                        f"{int(np.array(new_val['time_out']).sum())}/{len(new_val['correct'])}",
                    ]
                )

            writer.writerow("\n")
            end_time = time.time()
            print("Running time for one file:", end_time - start_time)
            print(f"Process Finished: Group: {group + 1}/{n},File:{f_index + 1}")

        csvfile.close()
