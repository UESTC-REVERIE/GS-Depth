# ps -ef |grep train.py |awk '{print $2}'|xargs kill -9

# ps -ef |grep GS-Depth_scale0_initdepthpeloss0.5 |awk '{print $2}'|xargs kill -9

CUDA_VISIBLE_DEVICES=1 \
python train_baseline.py \
    --data_path ~/dataset/KITTI_dataset/raw_data \
    --log_dir ~/code/GS-Depth/models \
    --model_name monodepth2_baseline_test \
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
    --loss_gs_weight 0.25 \
    --png \
    --use_init_smoothLoss \