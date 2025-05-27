# ps -ef |grep train.py |awk '{print $2}'|xargs kill -9

# ps -ef |grep GS-Depth_scale0_initdepthpeloss0.5 |awk '{print $2}'|xargs kill -9

CUDA_VISIBLE_DEVICES=1 \
python train_gs_multi_frames.py \
    --data_path ~/dataset/KITTI_dataset/raw_data \
    --log_dir ~/code/GS-Depth/models \
    --model_name v9_pn1_fused_skip_se_adaptive\
    --split eigen_zhou \
    --dataset kitti \
    --num_epochs 20 \
    --scheduler_step_size 15 \
    --learning_rate 1e-4 \
    --num_workers 12 \
    --batch_size 12 \
    --height 192 \
    --width 640 \
    --gs_scale 2 \
    --gs_num_per_pixel 1 \
    --eval_frequency_best 500 \
    --loss_gs_weight 1 \
    --loss_perception_weight 1 \
    --loss_init_weight 0.25 \
    --png \
    --use_gs \
    --use_init_smoothLoss \
    --pretrained_model_path ~/code/GS-Depth/models/v3_2se_offset_scale2_pre/models/model_9_147_best_abs_rel_0.09232.pth \
    --pretrained_models_to_load encoder init_decoder pose_encoder pose \
    --pretrained_frozen \