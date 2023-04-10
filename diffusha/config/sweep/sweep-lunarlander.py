#!/usr/bin/env python3
from params_proto.hyper import Sweep
import os
# from diffusha.config import RUN
from diffusha.config.default_args import Args
import sys
this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    with sweep.product:
        Args.env_name = ['LunarLander-v1', 'LunarLander-v5']

    Args.num_training_steps = 30_000  # Roughly 5 hours per 30k steps
    Args.randp = 0.0

sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')
