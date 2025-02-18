#!/bin/bash
#SBATCH -o ./slurm.out/gsdepth.%j.out
#SBATCH -J gsdepth_train
#SBATCH -p defq
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4

module load anaconda3
module load cuda11.8/toolkit/11.8.0
module load cudnn8.5-cuda11.8/8.5.0.96

source activate 3dgs

nvidia-smi