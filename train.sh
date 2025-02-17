# ps -ef |grep train.py |awk '{print $2}'|xargs kill -9

# ps -ef |grep GS-Depth_scale0_initdepthpeloss0.5 |awk '{print $2}'|xargs kill -9


# ---------------------------------------------------- baseline+gs start ----------------------------------------------------


CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --data_path /data/penghaoming/dataset/KITTI_dataset/raw_data \
    --log_dir /data/penghaoming/code/GS-Depth/models \
    --model_name GS-Depth_baseline_scale32_initdepth0.25_v2 \
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
    --gs_scale 32 \
    --loss_gs_weight 0.25 \
    --png



CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --data_path /data/penghaoming/dataset/KITTI_dataset/raw_data \
    --log_dir /data/penghaoming/code/GS-Depth/models \
    --model_name GS-Depth_baseline_scale16_initdepth0.25_v2 \
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
    --gs_scale 16 \
    --loss_gs_weight 0.25 \
    --png



CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --data_path /data/penghaoming/dataset/KITTI_dataset/raw_data \
    --log_dir /data/penghaoming/code/GS-Depth/models \
    --model_name GS-Depth_baseline_scale8_initdepth0.25_v2 \
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
    --gs_scale 8 \
    --loss_gs_weight 0.25 \
    --png


CUDA_VISIBLE_DEVICES=3 \
python train.py \
    --data_path /data/penghaoming/dataset/KITTI_dataset/raw_data \
    --log_dir /data/penghaoming/code/GS-Depth/models \
    --model_name GS-Depth_baseline_scale4_initdepth0.25_v2 \
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
    --gs_scale 4 \
    --loss_gs_weight 0.25 \
    --png



CUDA_VISIBLE_DEVICES=4 \
python train.py \
    --data_path /data/penghaoming/dataset/KITTI_dataset/raw_data \
    --log_dir /data/penghaoming/code/GS-Depth/models \
    --model_name GS-Depth_baseline_scale2_initdepth0.25_v2 \
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
    --gs_scale 2 \
    --loss_gs_weight 0.25 \
    --png





CUDA_VISIBLE_DEVICES=5 \
python train.py \
    --data_path /data/penghaoming/dataset/KITTI_dataset/raw_data \
    --log_dir /data/penghaoming/code/GS-Depth/models \
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



# ---------------------------------------------------- baseline+gs end ----------------------------------------------------


# ---------------------------------------------------- baseline+gs start ----------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_initpeloss0.25 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png



# CUDA_VISIBLE_DEVICES=1 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_featbeforetrans_initpeloss0.25 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png



# CUDA_VISIBLE_DEVICES=2 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_featbehindtrans_initpeloss0.25 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png



# CUDA_VISIBLE_DEVICES=3 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_initdepthpeloss0.5 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.5 \
#     --png


# CUDA_VISIBLE_DEVICES=7 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_featbeforetrans_initdepthpeloss0.5 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.5 \
#     --png



# CUDA_VISIBLE_DEVICES=8 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_featbehindtrans_initpeloss0.5 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.5 \
#     --png

# CUDA_VISIBLE_DEVICES=9 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_initpeloss0.5_retrain \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_initpeloss0.25_noadd \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png


# CUDA_VISIBLE_DEVICES=1 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_initpeloss0.25_add \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png


# CUDA_VISIBLE_DEVICES=3 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_initpeloss0.25_Fit3Dv1 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png


# CUDA_VISIBLE_DEVICES=2 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS-Depth_scale0_initpeloss0.25_Fit3Dv1_ds2 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --png



# ---------------------------------------------------- baseline+gs end ----------------------------------------------------


# ---------------------------------------------------- baseline+gs inter start ----------------------------------------------------
# CUDA_VISIBLE_DEVICES=4 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS_Depth_scale0_initpe0.25_inter0.1 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --png \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --use_interframe_consistency \
#     --loss_interframe_weight 0.1

# CUDA_VISIBLE_DEVICES=5 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS_Depth_scale0_initpe0.25_inter0.25 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --png \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --use_interframe_consistency \
#     --loss_interframe_weight 0.25


# CUDA_VISIBLE_DEVICES=6 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS_Depth_scale0_initpe0.25_inter0.5 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --png \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --use_interframe_consistency \
#     --loss_interframe_weight 0.5
    

# CUDA_VISIBLE_DEVICES=2 \
# python train.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --log_dir /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models \
#     --model_name GS_Depth_scale0_initpe0.25_inter0.25 \
#     --split eigen_zhou \
#     --dataset kitti \
#     --num_epochs 20 \
#     --scheduler_step_size 15 \
#     --learning_rate 1e-4 \
#     --num_workers 12 \
#     --batch_size 12 \
#     --scales 0 \
#     --height 192 \
#     --width 640 \
#     --png \
#     --eval_frequency_best 500 \
#     --use_gs \
#     --gs_scale 0 \
#     --loss_gs_weight 0.25 \
#     --use_interframe_consistency \
#     --loss_interframe_weight 0.25


# ---------------------------------------------------- baseline+gs inter end ----------------------------------------------------