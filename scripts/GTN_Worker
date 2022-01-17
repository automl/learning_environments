#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 3-23:59 # time (D-HH:MM)
#SBATCH -c 1 # number of cores, was 8 before
#SBATCH -a 0-511 # array size

#SBATCH -D /home/ferreira/Projects/learning_environments # Change working_dir
#SBATCH -o /home/ferreira/Projects/learning_environments/scripts/log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /home/ferreira/Projects/learning_environments/scripts/log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --job-name pendulum_se_worker # sets the job name. If not specified, the file name will be used as job name

export HTTP_PROXY=http://tfproxy.informatik.uni-freiburg.de:8080
export HTTPS_PROXY=https://tfproxy.informatik.uni-freiburg.de:8080
export http_proxy=http://tfproxy.informatik.uni-freiburg.de:8080
export https_proxy=https://tfproxy.informatik.uni-freiburg.de:8080

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

echo "source activate"
source ~/.miniconda/bin/activate gtn
echo "run script"
export PYTHONPATH=$PYTHONPATH:.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ferreira/.mujoco/mjpro150/bin
bohb_id=$(($SLURM_ARRAY_TASK_ID/64+20000))
id=$(($SLURM_ARRAY_TASK_ID%64))
cd experiments
python3 -u GTN_Worker.py $bohb_id $id
echo "done"

