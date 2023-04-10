#!/usr/bin/env python3

from pathlib import Path
import sys
this_file_name = sys.argv[0]

from params_proto.neo_hyper import Sweep
from diffusha.data_collection.config.default_args import DCArgs

with Sweep(DCArgs) as sweep:
    DCArgs.env_name = 'BlockPushMultimodal-v1'
    DCArgs.randp = 0.0
    with sweep.product:
        DCArgs.blockpush_user_goal = ['target', 'target2']

    DCArgs.num_transitions = 3_000_000
    DCArgs.valid_return_threshold = 200

sweep.save(Path(this_file_name).stem + '.jsonl')
