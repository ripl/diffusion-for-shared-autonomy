# Evaluation Workflow

### Evaluate:

1. Change Argument in `eval_assistance.py`

   - change `--wandb-run-id` to match the diffusion model we need to use
   - change `--env-name` to the correct environment 
   - change `wandb.init()` project name to store each evaluation in a new project
   - change `directory` path last item to be current date '20230109' (Not necessary but would be good)

2. Run `launch_eval_intention.sh` 

   - sweep_maze_eval.jsonl

   - --num-episodes: 10, 50, 100 ...

   - -l: which config line you want to run

   - Single config:

     `rmx run elm --contain -f 'cd .. && python -m diffusha.diffusion.evaluation.eval_assistance diffusha/config/sweep/sweep_maze_eval.jsonl --force --num-episodes 10 -l 0'`

   - Sweep Run:

     `rmx run elm --contain -f -d --sweep 0-44 -- 'cd .. && python -m diffusha.diffusion.evaluation.eval_assistance diffusha/config/sweep/sweep_maze_eval.jsonl --force --num-episodes 100 -l $RMX_RUN_SWEEP_IDX'`

3. If you want to use `slurm`  you need to uncomment 

   ```python
   line 28 # from diffusha.utils.tticslurm import report_cuda_error, upload_slurm_logs
   line 160 # upload_slurm_logs()
   ```

   Then comment line 147-149

   ```python
   gpus = ['0','1','2','3']
   cvd = args.line_number % len(gpus)
   os.environ["CUDA_VISIBLE_DEVICES"] = gpus[cvd]

### Generate Plot:

#### 1. Generate quantitative table

##### For LunarLander:

1. Change params in `eval_generate_ll.py`

   - `--wandb-run-id`  to match the correct diffusion model

   - `directory` path last item to match the training date `"20230108"` If the date don't match, you would fail. 

     (This date is set because you may evaluate the same param several times in different days. If we diceide to evaluate only one time we may remove this date later, and just use the run-id. But let's keep it temporarily.)

   - `file_path` last item to be whatever you want. e.g. `"20230108-reaching-summary.csv"`

2. Run `launch_eval_gen_ll.sh`

3. Now you get the csv table, get it to your local computer or wherever you can reach the file

##### For Maze2D:

(Generate Table column name should be fix. So that next step can be done........TODO)



#### 2. Process Table to Get Plot

1. Change file name in `read_csv_and_plot.py` to the csv you need to evaluation

   `"summary/20230108-reaching-summary.csv"`

2. Parameters explanation

   - change the `Laggy` and `Noisy` is the probability of making mistake. They can be set differently

   - `actors ` is the actor you would like to evaluate

   - `assisted` can be True or False

   - `attr` only for plt name, if you want to measure different attribute you still need to change line 111 and 112 manually 

     ```python
     mean.append(entry.success[0])
     std.append(entry.success[1])
     ```

   Play around with the script, it's npt hard. Just like a big dict of the csv you input.

3. Above is for "LunarLander" Would write one for Maze soon...........TODO







