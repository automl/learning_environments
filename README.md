# Learning Synthetic Environments for RL with Evolution Strategies Models:

# SE Model Download
The models can be downloaded here: https://www.dropbox.com/sh/1najkivfzc6tskw/AAABsp_5rEwQchZLf1fJDwR7a?dl=0

# RN Model Download
The models can be downloaded here: <todo>

# Installation

To install the required packages, you need mujoco (http://mujoco.org/).
The python3 packages can be installed using the requirements.txt file:

```
pip install -r requirements.txt
```


# Learning Synthetic Environments

## Optimizing Hyperparameters for Learning Synthetic Environments (three-level optimization)

### Overall structure

Several scripts in the experiments folder make use of the three-level optimization approach. 
- outer loop: BOHB
- middle loop: NES (in scripts referred to as "GTN-RL")
- inner loop: RL 

During parallelization, the logic is as follows: A single BOHB master orchestrates several BOHB workers (done by the BOHB package).
Every BOHB worker corresponds to a NES master and every NES master (referred to as "GTN" -> see GTN_master.py) orchestrates several NES workers  
(GTN_worker.py) via file IO. In each NES outer loop the NES master writes an individual input file and an input check file 
(the latter just prevents the first file from being read too early) to all of its NES works. After they finished calculations,
they write a result file and a result check file, which is then read again by the NES master.

To ensure that the individual files can be distinguished, every NES master and worker can be uniquely identified by
a combination of its BOHB-ID (assigned to the NES master and all NES workers of a single BOHB worker) and its ID
(assigned individually to each NES worker). The general file types for file IO between NES master and NES workers thus are:
```
<BOHB-ID>_<ID>_input.pt 
<BOHB-ID>_<ID>_input_check.pt
<BOHB-ID>_<ID>_result.pt 
<BOHB-ID>_<ID>_result_check.pt
```
Many scripts allow the BOHB-ID to be chosen manually whereas the ID for the different NES workers must span the range [0,X] 
with X as the number of NES workers per NES master.

### Example

To start a BOHB script (in this example: experiments/GTNC_evaluate_cartpole_params.py) with two BOHB workers (i.e. two NES master) 
and 3 NES slaves per NES master, execute the following procedure:

- ensure that the values in the loaded config file (look up the .py file to see which yaml file is loaded) are set to proper values:
```
agents.gtn.num_workers: 3
```
- run the individual files in parallel.  After setting the PYTHONPATH environment variable to the repository base folder, run in the command line:
```
python3 GTN_Worker.py 0 0 &
python3 GTN_Worker.py 0 1 &
python3 GTN_Worker.py 0 2 &
python3 GTN_Worker.py 1 0 &
python3 GTN_Worker.py 1 1 &
python3 GTN_Worker.py 1 2 &
python3 GTNC_evaluate_cartpole_params.py 0 3 &
python3 GTNC_evaluate_cartpole_params.py 1 3 &
```

An easier way on the slurm cluster would be the use of two scripts, the first script to call the NES workers (change absolute paths where necessary):
```
#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 3-23:59 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -a 0-5 # array size
#SBATCH -D /home/nierhoff/master_thesis/learning_environments # Change working_dir
#SBATCH -o /home/nierhoff/master_thesis/scripts/log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /home/nierhoff/master_thesis/scripts/log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

echo "source activate"
source ~/master_thesis/mtenv/bin/activate
echo "run script"
export PYTHONPATH=$PYTHONPATH:/home/nierhoff/master_thesis/learning_environments
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nierhoff/.mujoco/mjpro150/bin
bohb_id=$(($SLURM_ARRAY_TASK_ID/3+0))
id=$(($SLURM_ARRAY_TASK_ID%3))
cd experiments
python3 -u GTN_Worker.py $bohb_id $id
echo "done"
```
and the second script to call BOHB respectively the NES master (change absolute paths where necessary):
```
#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 3-23:59 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -a 0-1 # array size
#SBATCH -D /home/nierhoff/master_thesis/learning_environments # Change working_dir
#SBATCH -o /home/nierhoff/master_thesis/scripts/log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /home/nierhoff/master_thesis/scripts/log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

echo "source activate"
source ~/master_thesis/mtenv/bin/activate
echo "run script"
export PYTHONPATH=$PYTHONPATH:/home/nierhoff/master_thesis/learning_environments
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nierhoff/.mujoco/mjpro150/bin
cd experiments
bohb_id=$(($SLURM_ARRAY_TASK_ID+0))
python3 -u GTNC_evaluate_cartpole.py $bohb_id 3
echo "done"
```

A third alternative is to run everything (single NES master and multiple NES workers) on a single PC, e.g for debug purposes
All the scripts in the experiments folder containing a "run_bohb_serial" method support this feature. Just call 
```
python3 GTN_Worker_single_pc.py &
python3 GTNC_evaluate_cartpole_params.py &
```

## Training Synthetic Environments after HPO

To train synthetic environments, run the corresponding "GTNC_evaluate_XXX.py" script, e.g.
```
GTNC_evaluate_gridworld.py
GTNC_evaluate_acrobot.py
GTNC_evaluate_cartpole.py
```
as described in the "Example" section before. It might be necessary to modify some parameters in the corresponding .yaml file.
The trained synthetic environments are then called within other functions to generate the plots as described in the "Visualizations scripts" section

## Training Reward Networks after HPO

To train reward environments, run the corresponding "GTNC_evaluate_XXX_reward_env.py" script if there exists a "GTNC_evaluate_XXX.py" script as well.
In all other cases the script is called "GTNC_evaluate_XXX.py", e.g.
```
GTNC_evaluate_gridworld_reward_env.py
GTNC_evaluate_cartpole_reward_env.py
GTNC_evaluate_cmc.py
GTNC_evaluate_halfcheetah.py
```
The detailed procedure how to run these scripts is described in the "Example" section before. 
It might also be necessary to modify some parameters in the corresponding .yaml file.
The only difference is that these scripts feature another input parameter specifying the "mode", i.e. type of used RL agent / reward environment:

- -1: ICM
- 0: native environment (no reward env)
- 1: potential function (exclusive)
- 2: potential function (additive)
- 3: potential function with additional info vector (exclusive)
- 4: potential function with additional info vector (additive)
- 5: non-potential function (exclusive)
- 6: non-potential function (additive)
- 7: non-potential function with additional info vector (exclusive)
- 8: non-potential function with additional info vector (additive)
- 101: weighted info vector as baseline (exclusive)
- 102: weighted info vector as baseline (additive)

The trained reward environments are then called within other functions to generate the plots as described in the "Visualizations scripts" section

## Visualizations scripts

Please note that form many scripts (in general all starting with "GTNC_visualize") first the corresponding environments must
be trained as described in the previous section.

Fig. 2 (standalone):
```
python3 experiments/demo_score_transform.py
```

Fig. 6: First run "GTNC_evaluate_gridworld_step_size.py" as described in the "Example" section. Then run
```
python3 experiments/GTNC_visualize_gridworld_step_size.py
```
with appropriate "log_dir" variable.

Fig. 7 / Fig. 8: First run "GTNC_evaluate_gridworld.py" as described in the "Example" section. Then run
```
python3 experiments/GTNC_visualize_gridworld.py
```
with appropriate "dir" and "file_name" variable.

Fig. 9: First run "GTNC_evaluate_cartpole.py" as described in the "Example" section. Then run
```
python3 experiments/GTNC_visualize_cartpole_histogram.py
```
with appropriate "dir" and "file_name" variable.

Fig. 10: First run "GTNC_evaluate_cartpole.py" as described in the "Example" section. Then run
```
python3 experiments/GTNC_visualize_cartpole_threshold.py
```
with appropriate "dir" and "model_file_name" variable.

Fig. 12: First run "GTNC_evaluate_cartpole.py" with optimized parameters (default_config_cartpole_syn_env.yaml) and 
"GTNC_evaluate_acrobot.py" with optimized parameters (default_config_acrobot_syn_env.yaml) 
as described in the "Example" section. Then run
```
python3 experiments/GTNC_visualize_cartpole_threshold.py
```
with appropriate "LOG_DIRS" variable.

Fig. 13: First run "GTNC_evaluate_gridworld_reward_env.py" 
as described in the "Example" section with the different modes as input. Then run
```
python3 experiments/GTN_visualize_gridworld_learned_reward_env.py
```
with appropriate "LOG_DIR"/"SIMPLIFY" variable.

Fig. 14: First run 
"GTNC_evaluate_gridworld_reward_env.py",
"GTNC_evaluate_cartpole_reward_env.py",
"GTNC_evaluate_cmc.py",
"GTNC_evaluate_halfcheetah.py",
as described in the "Example" section with the different modes as input. Then run
```
python3 experiments/GTNC_evaluate_gridworld_compare_reward_envs.py <mode>
python3 experiments/GTNC_evaluate_cartpole_compare_reward_envs.py <mode>
python3 experiments/GTNC_evaluate_cmc_compare_reward_envs.py <mode>
python3 experiments/GTNC_evaluate_halfcheetah_compare_reward_envs.py <mode>
```
with appropriate "LOG_DICT"/"SAVE_DIR" variables and "mode" as additional script input as 
described in the "Training Reward Networks after HPO" section. Finally run
```
python3 experiments/GTNC_visualize_gridworld_compare_reward_envs.py
python3 experiments/GTNC_visualize_cartpole_compare_reward_envs.py
python3 experiments/GTNC_visualize_cmc_compare_reward_envs.py
python3 experiments/GTNC_visualize_halfcheetah_compare_reward_envs.py
```
with appropriate "LOG_FILES" variable and matching legend to plot.

Fig. 15: First run
"GTNC_evaluate_gridworld_reward_env.py",
"GTNC_evaluate_cartpole_reward_env.py",
"GTNC_evaluate_cmc.py",
"GTNC_evaluate_halfcheetah.py",
as described in the "Example" section with the different modes as input. Then run
```
python3 experiments/GTNC_evaluate_gridworld_transfer_vary_hp.py <mode>
python3 experiments/GTNC_evaluate_cartpole_transfer_vary_hp.py <mode>
python3 experiments/GTNC_evaluate_cmc_transfer_vary_hp.py <mode>
python3 experiments/GTNC_evaluate_halfcheetah_transfer_vary_hp.py <mode>
```
with appropriate "LOG_DICT"/"SAVE_DIR" variables and "mode" as additional script input as
described in the "Training Reward Networks after HPO" section. Finally run
```
python3 experiments/GTNC_visualize_gridworld_transfer_vary_hp.py
python3 experiments/GTNC_visualize_cartpole_transfer_vary_hp.py
python3 experiments/GTNC_visualize_cmc_transfer_vary_hp.py
python3 experiments/GTNC_visualize_halfcheetah_transfer_vary_hp.py
```
with appropriate "LOG_FILES" variable and matching legend to plot.

Fig. 15: First run
"GTNC_evaluate_gridworld_reward_env.py",
"GTNC_evaluate_cartpole_reward_env.py",
"GTNC_evaluate_cmc.py",
"GTNC_evaluate_halfcheetah.py",
as described in the "Example" section with the different modes as input. Then run
```
python3 experiments/GTNC_evaluate_gridworld_transfer_algo.py <mode>
python3 experiments/GTNC_evaluate_cartpole_transfer_algo.py <mode>
python3 experiments/GTNC_evaluate_cmc_transfer_algo.py <mode>
python3 experiments/GTNC_evaluate_halfcheetah_transfer_algo.py <mode>
```
with appropriate "LOG_DICT"/"SAVE_DIR" variables and "mode" as additional script input as
described in the "Training Reward Networks after HPO" section. Finally run
```
python3 experiments/GTNC_visualize_gridworld_transfer_algo.py
python3 experiments/GTNC_visualize_cartpole_transfer_algo.py
python3 experiments/GTNC_visualize_cmc_transfer_algo.py
python3 experiments/GTNC_visualize_halfcheetah_transfer_algo.py
```
with appropriate "LOG_FILES" variable and matching legend to plot.

Fig. 17: First run "GTNC_evaluate_gridworld_reward_env.py" 
as described in the "Example" section with the different modes as input. Then run
```
python3 experiments/GTNC_evaluate_gridworld_heatmap.py
```
with appropriate "LOG_DICT"/"SAVE_DIR"/"MODE"/"BREAK" variables. Finally run
```
python3 experiments/GTNC_visuaize_gridworld_heatmap.py
```
with a suitable "RESULT_FILE" variable.

