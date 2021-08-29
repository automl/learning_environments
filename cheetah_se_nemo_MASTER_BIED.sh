#!/bin/bash
#MSUB -N masterSE01
#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=03:00:30:00
#MSUB -l pmem=6gb
#MSUB -d /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/
#MSUB -o /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/out_std_${MOAB_JOBID}.out
#MSUB -e /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/error_std_${MOAB_JOBID}.err

# =============== SOURCING / LINUX ENV ================== #
echo "source bashrc"
source /home/fr/fr_fr/fr_ab1184/.bashrc

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
bohb_id=$((20000))
/home/fr/fr_fr/fr_ab1184/anaconda3/envs/nemo_SE_CPU/bin/python /work/ws/nemo/fr_ab1184-fabios_experiments-0/learning_environments/syn_env_learn_halfcheetah.py $bohb_id 1
echo "Finished at $(date)";