#!/bin/bash
#MSUB -N masterSE01
#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=00:04:00:00
#MSUB -l pmem=6gb
#MSUB -d /work/ws/nemo/fr_as1464-se_port_ws-0/learning_environments/                            #TODO: PATH
#MSUB -o /work/ws/nemo/fr_as1464-se_port_ws-0/learning_environments/out_std_${MOAB_JOBID}.out   #TODO: PATH
#MSUB -e /work/ws/nemo/fr_as1464-se_port_ws-0/learning_environments/error_std_${MOAB_JOBID}.err #TODO: PATH

# =============== SOURCING / LINUX ENV ================== #
echo "source bashrc"
source /home/fr/fr_fr/fr_as1464/.bashrc       #TODO: PATH

echo "conda activate"
conda activate nemo_SE_CPU

# ================ DEBUG INFOS ===================== #
echo 'env:' $CONDA_DEFAULT_ENV
echo 'env:' $CONDA_PREFIX
echo 'pythonpath:' $PYTHONPATH
echo "path: $PATH"
echo 'which python:' $(which python)
echo "Workingdir: $PWD";
echo "Started at $(date)";


# =============== Begin actual Code ================== #
echo "Starting/Calling at $(date)";

# EXPERIMENTS HYPERPARAMETERS
BOHB_ID=$((60000))
ID=$((0))
PORT=$((10000))
MIN_WORKER=$((2))
NUMBER_OF_WORKERS=$((16))
RUN_MODE="master"

PYTHON_SCRIPT="syn_env_learn_halfcheetah.py"

echo 'calling ' $PWD/$PYTHON_SCRIPT --bohb_id $BOHB_ID --id $ID --moab_id $MOAB_JOBID --port $PORT --min_workers $MIN_WORKER --number_workers $NUMBER_OF_WORKERS --mode $RUN_MODE

# CALL
/home/fr/fr_fr/fr_as1464/anaconda3/envs/nemo_SE_CPU/bin/python -u $PWD/$PYTHON_SCRIPT --bohb_id $BOHB_ID --id $ID --moab_id $MOAB_JOBID --port $PORT --min_workers $MIN_WORKER --number_workers $NUMBER_OF_WORKERS --mode $RUN_MODE #TODO: PATH

echo "Finished at $(date)";
