#!/bin/bash
#MSUB -N workerSE
#MOAB -t 1-2000 # specifies array job indices
#MSUB -l nodes=1:ppn=1
#MSUB -l walltime=03:00:00:00
#MSUB -l pmem=6gb
#MSUB -d /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/
#MSUB -o /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/out_std_${MOAB_JOBID}.out
#MSUB -e /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/error_std_${MOAB_JOBID}.err

# =============== SOURCING / LINUX ENV ================== #
echo "source bashrc"
source /home/fr/fr_fr/fr_ab1184/.bashrc

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


# =============== Begin actual Code ================== #
bohb_id=$((20000))
id=$((MOAB_JOBARRAYINDEX))
echo "bohb_id: $bohb_id";
echo "id: $id";

echo "Starting/Calling at $(date)";
/home/fr/fr_fr/fr_ab1184/anaconda3/envs/nemo_SE_CPU/bin/python -u /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/GTN_Worker.py $bohb_id $id
echo "Finished at $(date)";