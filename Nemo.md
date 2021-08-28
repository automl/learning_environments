# Nemo Tutorial

## Steps

1. make a workspace:
2. cd to workspace 
3. clone repo 
4. move `install_1_mujoco_py_centos.sh` and `install_2_mujoco_download.sh` to you home directory `~/` 
5. execute `install_1_mujoco_py_centos.sh` 
6. and `install_2_mujoco_download.sh`
7. copy your mujoco keyfile to `your_home/.mujoco`
8. cd back to workspace repo 
9. execute `conda create env -f environment_cpu.yml`
10. Change paths in `cheetah_se_nemo_MASTER_FABIO.sh`
11. Change paths in `cheetah_se_nemo_WORKER_FABIO.sh`


<br><br>

Now you can start experiments by this order:

1. `msub cheetah_se_nemo_MASTER_FABIO.sh`
2. `msub cheetah_se_nemo_WORKER_FABIO.sh`

<br>

__ATTENTION__: For multiple runs change the `bhob_id` values in: 
- `cheetah_se_nemo_WORKER_FABIO.sh` and 
- `cheetah_se_nemo_MASTER_FABIO.sh`