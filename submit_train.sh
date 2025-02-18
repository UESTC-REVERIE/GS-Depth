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
python train.py \
    --data_path ~/dataset/KITTI_dataset/raw_data \
    --log_dir ~/code/GS-Depth/models \
    --model_name GS-Depth_baseline_scale0_initdepth0.25_v2 \
    --split eigen_zhou \
    --dataset kitti \
    --num_epochs 20 \
    --scheduler_step_size 15 \
    --learning_rate 1e-4 \
    --num_workers 12 \
    --batch_size 12 \
    --scales 0 \
    --height 192 \
    --width 640 \
    --eval_frequency_best 500 \
    --use_gs \
    --gs_scale 0 \
    --loss_gs_weight 0.25 \
    --png