#!/usr/bin/env python3

import os
from params_proto.neo_hyper import Sweep
# from diffusha.config import RUN
from diffusha.data_collection.config.default_args import DCArgs
import sys
this_file_name = sys.argv[0]

with Sweep(DCArgs) as sweep:
    DCArgs.valid_return_threshold = 800
    DCArgs.env_name = 'LunarLander-v5'
    with sweep.product:
        DCArgs.randp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')
