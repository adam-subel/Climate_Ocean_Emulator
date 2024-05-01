#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --time=24:00:00
#SBATCH --mem=150GB
#SBATCH --job-name=3d
#SBATCH --output=slurm/job_%j.out
#SBATCH --error=slurm/job_%j.err
#SBATCH --mail-user=sg7761@nyu.edu

module purge


singularity exec --nv \
             --overlay /scratch/sg7761/m2lines/overlay-25GB-500K.ext3:ro \
             /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate emulator; python3 train_lat_boundary_parallel3d.py"