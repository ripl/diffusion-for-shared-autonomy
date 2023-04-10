if __name__ == '__main__':
    from params_proto.neo_hyper import Sweep
    import os
    from diffusha.config.default_args import Args
    import sys

    print("python",sys.version)

    this_file_name = sys.argv[0]

    with Sweep(Args) as sweep:
        Args.naive_blending = True
        Args.num_training_steps = 5_000
        with sweep.product:
            Args.env_name = [f'LunarLander-{level}' for level in ['v1', 'v5']]

    sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')

