#!/bin/bash
#MSUB -N workerSE
#MOAB -t 1-16 # specifies array job indices
#MSUB -l nodes=1:ppn=1
#MSUB -l walltime=01:00:00:00
#MSUB -l pmem=6gb
#MSUB -d /work/ws/nemo/fr_as1464-se_port_ws-0/learning_environments/
#MSUB -o /work/ws/nemo/fr_as1464-se_port_ws-0/learning_environments/out_std_${MOAB_JOBID}.out
#MSUB -e /work/ws/nemo/fr_as1464-se_port_ws-0/learning_environments/error_std_${MOAB_JOBID}.err

# =============== SOURCING / LINUX ENV ================== #
echo "source bashrc"
source /home/fr/fr_fr/fr_as1464/.bashrc

echo "conda activate"
conda activate nemo_SE_CPU
# ================================================== #

# ================ DEBUG INFOS ===================== #
echo 'env:' $CONDA_DEFAULT_ENV
echo 'env:' $CONDA_PREFIX
echo 'pythonpath:' $PYTHONPATH
echo "path: $PATH"
echo 'which python:' $(which python)
echo "Workingdir: $PWD";
echo "Started at $(date)";
# ================================================== #


# =============== Begin actual Code ==================
echo "Starting/Calling at $(date)";

# EXPERIMENTS HYPERPARAMETERS
BOHB_ID=$((50000))
ID=$((MOAB_JOBARRAYINDEX-1))
PORT=$((10000))
MIN_WORKER=$((200))
NUMBER_OF_WORKERS=$((256))
RUN_MODE="worker"

PYTHON_SCRIPT="GTN_Worker.py"

echo 'calling ' $PWD/$PYTHON_SCRIPT --bohb_id $BOHB_ID --id $ID --moab_id $MOAB_JOBID --port $PORT --min_workers $MIN_WORKER --number_workers $NUMBER_OF_WORKERS --mode $RUN_MODE

# CALL
/home/fr/fr_fr/fr_as1464/anaconda3/envs/nemo_SE_CPU/bin/python -u $PWD/$PYTHON_SCRIPT --bohb_id $BOHB_ID --id $ID --moab_id $MOAB_JOBID --port $PORT --min_workers $MIN_WORKER --number_workers $NUMBER_OF_WORKERS --mode $RUN_MODE
### OLD: /home/fr/fr_fr/fr_as1464/anaconda3/envs/nemo_SE_CPU/bin/python -u /work/ws/nemo/fr_as1464-se_port_ws-0/learning_environments-master/GTN_Worker.py $bohb_id $id

echo "Finished at $(date)";
