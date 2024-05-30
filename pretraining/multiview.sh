#!/bin/bash

#SBATCH -c 4                       # Number of CPU Cores (16)
#SBATCH -p gpus                 # Partition (queue)
#SBATCH --gres gpu:2                # gpu:n, where n = number of GPUs
#SBATCH --mem 128G                  # memory pool for all cores
#SBATCH --nodelist monal03      	# SLURM node
#SBATCH --output=slurm.%N.%j.log    # Output and error log (N = node, j = job ID)

#Job commands
export TORCH_HOME='/vol/biomedic3/bglocker/ugproj2324/ds1021/.cache/'
source /vol/biomedic3/bglocker/ugproj2324/ds1021/env1/bin/activate
srun python /vol/biomedic3/bglocker/ugproj2324/ds1021/multiviewpretraining/pretraining/pretrain.py
