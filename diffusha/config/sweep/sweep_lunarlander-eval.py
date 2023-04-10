#!/usr/bin/env python3
from params_proto.hyper import Sweep
import os
# from diffusha.config import RUN
from diffusha.config.default_args import Args
import sys
this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    Args.naive_blending = True
    with sweep.product:
        Args.env_name = [f'LunarLander-{level}' for level in ['v1', 'v5']]
        Args.blend_alpha = [0.1 * i for i in range(11)]

sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')
