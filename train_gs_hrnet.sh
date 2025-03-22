# ps -ef |grep train.py |awk '{print $2}'|xargs kill -9

# ps -ef |grep GS-Depth_scale0_initdepthpeloss0.5 |awk '{print $2}'|xargs kill -9

CUDA_VISIBLE_DEVICES=7 \
python train_gs_hrnet.py \
    --data_path /data/penghaoming/dataset/KITTI_dataset/raw_data \
    --log_dir /data/penghaoming/code/GS-Depth/models \
    --model_name v3_2se_offset_scale2_multigs_sm \
    --split eigen_zhou \
    --dataset kitti \
    --num_epochs 20 \
    --scheduler_step_size 15 \
    --learning_rate 1e-4 \
    --num_workers 12 \
    --batch_size 12 \
    --height 192 \
    --width 640 \
    --eval_frequency_best 500 \
    --loss_gs_weight 1 \
    --loss_init_weight 0.25 \
    --png \
    --use_gs \
    --use_init_smoothLoss \
    # --pretrained_model_path /data/penghaoming/code/GS-Depth/models/baseline/hrnet_gs_singlegs_fused_epoch17.pth \
    # --pretrained_models_to_load encoder init_decoder pose_encoder pose \
    # --pretrained_frozen \