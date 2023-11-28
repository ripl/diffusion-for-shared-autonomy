# To the Noise and Back: Diffusion for Shared Autonomy

[Takuma Yoneda](https://takuma.yoneda.xyz), [Luzhe Sun](https://tllokn.github.io/), [Ge Yang](https://www.episodeyang.com), [Bradly C. Stadie](https://bstadie.github.io), [Matthew R. Walter](https://ttic.edu/walter)

[[Paper]](http://arxiv.org/abs/2302.12244) [[Website]](https://diffusion-for-shared-autonomy.github.io/)

![fwd-rev-diffusion](https://user-images.githubusercontent.com/28857806/228377612-154ff764-28b5-411d-9eed-49eff1942e3b.png)

## Prerequisite
Our script uses [Weights and Biases](https://wandb.ai) (WandB) to log metrics.
Please make sure to set the environment variable `WANDB_API_KEY`.
``` bash
export WANDB_API_KEY=<your WANDB key>
```

If you don't plan to use WandB, you can simply set the environment variable to `WANDB_MODE=disable`, which
makes all WandB functions behave as no-ops. You should then replace `wandb.log(...)` with your logging method.

<a name="quickstart"></a>
## Quickstart 🚀

``` bash
git clone https://github.com/ripl/diffusion-for-shared-autonomy.git
cd diffusion-for-shared-autonomy
mkdir output-dir
export CODE_DIR=`pwd`
export DATA_DIR=$CODE_DIR/data-dir  # Pretrained models are stored here
export OUT_DIR=$CODE_DIR/output-dir
```

You can pull our docker image and run a container via
``` bash
docker run -it --gpus all -e WANDB_API_KEY=$WANDB_API_KEY -v $CODE_DIR:/code -v $DATA_DIR:/data -v $OUT_DIR:/outdir --workdir /code ripl/diffusion-for-shared-autonomy bash
```
<details>
<summary>If you use Singularity</summary>

```bash
# Pull the image from dockerhub
singularity pull diffusion-for-shared-autonomy.sif docker://ripl/diffusion-for-shared-autonomy:latest
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY  # Singularity way to pass an envvar into a container
singularity run --nv --containall --writable-tmpfs -B $CODE_DIR:/code -B $DATA_DIR:/data -B $OUT_DIR:/outdir --pwd /code diffusion-for-shared-autonomy.sif bash
```
</details>


Now, inside of the container, you can run surrogate pilots with / without the assistance of our diffusion models.

``` bash
python -m diffusha.diffusion.evaluation.eval_assistance --env-name LunarLander-v1 --out-dir /outdir --save-video
```
This takes ~30 min.

`--env-name` can be one of the following
- `LunarLander-v1` (Lunar Reacher)
- `LunarLander-v5` (Lunar Lander)
- `BlockPushMultimodal-v1`

This will generate and log videos on WandB console, and saves evaluation statistics under `output-dir`.

NOTE: The following errors can be safely ignored
> ALSA lib confmisc.c:767:(parse_card) cannot find card '0'  
> ALSA lib conf.c:4732:(_snd_config_evaluate) function xxxxx returned error: No such file or directory

---

## Going through the entire pipeline 🐢

The following sections describe how to manually run all the steps of our pipeline. Specifically,
1. [Pretraining expert policies](#pretraining-expert-policies)
2. [Collecting demonstrations](#collecting-demonstrations)
3. [Training a diffusion model](#training-diffusion-model)

<a name="preparing-the-dataset"></a>
## Preparing the dataset for training a diffusion model
You can either follow the instruction below, or you can also download the resulting dataset:

<details>
<summary>You can also download the resulting dataset</summary>

``` bash
cd diffusion-for-shared-autonomy
wget https://dl.ttic.edu/diffusion-for-shared-autonomy.tar.gz
tar xfvz diffusion-for-shared-autonomy
mv hosted_data/* data-dir/
```

</details>


<a name="pretraining-expert-policies"></a>
### Pretraining expert policies
We first need to train an expert policy for each task (we use Soft Actor-Critic (SAC)).
(You can also download the pretrained SAC models from the link in the [Quickstart section](#quickstart) above.

**Lunar Reacher**
``` bash
python -m diffusha.data_collection.train_sac --env-name LunarLander-v1 --steps 3000000
```

**Lunar Lander**
``` bash
python -m diffusha.data_collection.train_sac --env-name LunarLander-v5 --steps 3000000
```

**Block Pushing**
``` bash
python -m diffusha.data_collection.train_sac --env-name BlockPushMultimodal-v1 --steps 1000000
```

The checkpoints will be saved under 
`Path(Args.sac_model_dir) / args.env_name.lower() / wandb.run.project / wandb.run.id`.
`Args.sac_model_dir` can be configured in `diffusha/config/default_args.py`.
> `wandb.run.project` and `wandb.run.id` are here only to make sure a unique id is given for each run. You may simply replace these with other strings if you don't use wandb.

<a name="collecting-demonstrations"></a>
### Rollout expert policies to collect demonstration
Once expert policies are obtained, we use them to collect expert demonstrations.


You can also download the generated demonstrations from the link (See [the previous section](#preparing-the-dataset)).

<!--
Originally:
"SAC_V1_MODEL_PATH": "/data/sac-v1/3300000_checkpoint",
"SAC_V5_MODEL_PATH": "/data/sac-v5/3000000_checkpoint",
blockpush: /data/sac/pfrl-BlockPushMultimodal-v1/o6rd3jun/20230128T145045.656374/1000000_finish
-->
1. Store the pretrained expert models in the following locations:
- Lunar Reacher: `$DATA_DIR/experts/lunarlander/v1`
- Lunar Lander: `$DATA_DIR/experts/lunarlander/v5`
- Block Pushing: `$DATA_DIR/experts/blockpush`

2. Run `generate_data.py` as follows:
**Lunar Reacher**
You can run the script with a sweep file:
`python -m diffusha.data_collection.generate_data -l 0 --sweep-file diffusha/data_collection/config/sweep/sweep_lander-v1.jsonl`.

Running this^ is equivalent to running:
`python -m diffusha.data_collection.generate_data`
after manually editing `diffusha/data_collection/config/default_args.py`:
``` python
class DCArgs(PrefixProto):
    ...
    env_name = 'LunarLander-v1'
    valid_return_threshold = 800
    randp = 0.0
    ...
```

**Lunar Lander Landing**
`python -m diffusha.data_collection.generate_data -l 0 --sweep-file diffusha/data_collection/config/sweep/sweep_lander-v5.jsonl`

**Block Pushing**
`python -m diffusha.data_collection.generate_data -l 0 --sweep-file diffusha/data_collection/config/sweep/sweep-blockpush.jsonl`

**Block Pushing (Running flip-replay)**  
For the Block Pushing task, we only collect demonstrations that push a block to one of the goals.
We obtain demonstrations that reach the other goal by "flipping" the collected trajectories.
```bash
python -m diffusha.data_collection.flip_replay /data/replay/blockpush/target/randp_0.0 /data/replay/blockpush/target-flipped/randp_0.0
```

<a name="training-diffusion-model"></a>
## Training a diffusion model
After obtaining demonstrations, we are finally ready to train a diffusion model.
Store the generated demonstrations in the correct locations:
- Lunar Reacher: `$DATA_DIR/replay/lunarlander/v1/randp_0.0` 
- Lunar Lander: `$DATA_DIR/replay/lunarlander/v5/randp_0.0`
- Block Pushing:
  - `$DATA_DIR/replay/blockpush/target/randp_0.0`
  - `$DATA_DIR/replay/blockpush/target-flipped/randp_0.0` (flipped replay)

Run the following command from the root of the project directory:

**Lunar Lander / Reacher**
- reacher task: `python -m diffusha.diffusion.train --sweep-file diffusha/config/sweep/sweep-lunarlander.jsonl -l 0`
- lander task: `python -m diffusha.diffusion.train --sweep-file diffusha/config/sweep/sweep-lunarlander.jsonl -l 1` 

**Block Pushing**
- `python -m diffusha.diffusion.train --sweep-file diffusha/config/sweep/sweep-blockpush.jsonl -l 0
`
---

If you find our work useful in your research, please consider citing the paper as follows:

``` bibtex
@inproceedings{yoneda2023diffusha,
    author = {Takuma Yoneda and
              Luzhe Sun and
              Ge Yang and
              Bradly C. Stadie and
              Matthew R. Walter},
    title = {To the Noise and Back: Diffusion for Shared Autonomy},
    booktitle = {Robotics: Science and Systems XIX, Daegu, Republic of Korea, July 10-14, 2023},
    year = {2023}
}
```
