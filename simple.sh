

CUDA_VISIBLE_DEVICES=9 \
python simpleTest_KITTI_bestmodel.py \
    --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
    --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initpeloss0.25_Fit3Dv1/models/model_19_2477_best_abs_rel_0.09325.pth \
    --eval_split eigen \
    --use_gs \
    --gs_scale 0