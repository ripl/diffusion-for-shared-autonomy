#!/usr/bin/env python3
from params_proto import PrefixProto


class DCArgs(PrefixProto):
    seed = 0
    env_name = 'LunarLander-v1'  # Custom Env
    num_transitions = 10_000_000

    # Lunar Lander
    lunarlander_sac_model_dir = '/data/experts/lunarlander'
    lunarlander_data_dir = '/data/replay/lunarlander'

    pointmaze_data_dir = '/data/replay/pointmaze'
    valid_return_threshold = -50
    randp = 0.

    # Block Push
    blockpush_sac_model_dir = '/data/experts/blockpush'
    blockpush_data_dir = '/data/replay/blockpush'
    blockpush_user_goal = 'target'  # 'target' or 'target2'
