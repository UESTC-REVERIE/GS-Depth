CUDA_VISIBLE_DEVICES=5 \
python evaluate_depth_gs_bestmodel.py \
    --data_path ~/dataset/KITTI_dataset/raw_data \
    --load_weights_path /data/penghaoming/code/GS-Depth/models/v5_gs_hrinit_mse_scale2_pre/models/model_19_1477_best_d1_0.91918.pth \
    --eval_split eigen \
    --eval_mono \
    --use_gs \
    --gs_scale 2 \
    --gs_num_per_pixel 2 \
    # --eval_output_dir /data/penghaoming/code/GS-Depth/models/v5_gs_hrinit_mse_scale2_pre/eval \
    # --save_pred_disps



# CUDA_VISIBLE_DEVICES=7 \
# python evaluate_depth_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline/models/model_19_2477_best_abs_rel_0.09230.pth \
#     --eval_split eigen \
#     --eval_mono
# Total parameters: 9.562260M (9562260)
# Total_trainable_params: 9.562260M (9562260)
# Total parameters: 0.416589M (416589)
# Total_trainable_params: 0.416589M (416589)
# Network total parameters: 9.978849M (9978849)
# Network total_trainable_params: 9.978849M (9978849)
# model_19_2477_best_abs_rel_0.09230
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.741  &   4.466  &   0.179  &   0.897  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale32/models/model_19_2977_best_abs_rel_0.09299.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 32 \
#     --eval_mono
# model_13_879_best_abs_rel_0.09754:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.712  &   4.470  &   0.179  &   0.893  &   0.965  &   0.984  \\
# model_16_2428_best_abs_rel_0.09384:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.100  &   0.726  &   4.412  &   0.177  &   0.899  &   0.966  &   0.983  \\
# model_19_2977_best_abs_rel_0.09299:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.736  &   4.426  &   0.177  &   0.898  &   0.966  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale32_initdepth0.25/models/model_19_477_best_abs_rel_0.09666.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 32 \
#     --eval_mono
# Total parameters: 9.562260M (9562260)
# Total_trainable_params: 9.562260M (9562260)
# Total parameters: 1.439462M (1439462)
# Total_trainable_params: 1.438742M (1438742)
# Network total parameters: 11.001722M (11001722)
# Network total_trainable_params: 11.001002M (11001002)
# model_7_2781_best_abs_rel_0.10240:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.105  &   0.709  &   4.699  &   0.184  &   0.879  &   0.962  &   0.983  \\
# model_16_2428_best_abs_rel_0.09789:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.734  &   4.434  &   0.179  &   0.896  &   0.965  &   0.983  \\
# model_19_477_best_abs_rel_0.09666:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.725  &   4.427  &   0.179  &   0.897  &   0.966  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale16_initdepth0.25/models/model_19_2977_best_abs_rel_0.09268.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 16 \
#     --eval_mono
# model_19_2977_best_abs_rel_0.09268:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.741  &   4.434  &   0.179  &   0.896  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale8_initdepth0.25/models/model_19_1977_best_abs_rel_0.09312.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 8 \
#     --eval_mono
# model_19_1977_best_abs_rel_0.09312:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.103  &   0.774  &   4.500  &   0.180  &   0.896  &   0.964  &   0.983  \\



# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale4_initdepth0.25/models/model_19_2977_best_abs_rel_0.09423.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 4 \
#     --eval_mono
# model_19_2977_best_abs_rel_0.09423:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.777  &   4.521  &   0.180  &   0.896  &   0.965  &   0.983  \\



# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale2_initdepth0.25/models/model_19_2477_best_abs_rel_0.09229.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 2 \
#     --eval_mono
# model_19_2477_best_abs_rel_0.09229:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.103  &   0.755  &   4.491  &   0.180  &   0.895  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale0_initdepth0.25/models/model_19_1477_best_abs_rel_0.17053.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_1477_best_abs_rel_0.17053:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.165  &   1.470  &   5.854  &   0.239  &   0.786  &   0.924  &   0.969  \\






# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale32_initdepth0.25/models/model_19_1977_best_d2_0.96404.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 32 \
#     --eval_mono
# model_18_2794_best_abs_rel_0.09624:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.770  &   4.483  &   0.179  &   0.897  &   0.966  &   0.983  \\
# model_18_3294_best_d1_0.91763:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.759  &   4.467  &   0.179  &   0.897  &   0.966  &   0.983  \\
# model_19_1977_best_d2_0.96404:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.779  &   4.503  &   0.180  &   0.897  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale16_initdepth0.25/models/model_19_1477_best_log_rms_0.17524.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 16 \
#     --eval_mono
# model_16_928_best_abs_rel_0.09370:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.731  &   4.470  &   0.179  &   0.896  &   0.965  &   0.983  \\
# model_19_2977_best_d1_0.91980:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.743  &   4.492  &   0.179  &   0.896  &   0.965  &   0.983  \\
# model_19_1477_best_log_rms_0.17524:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.738  &   4.491  &   0.179  &   0.896  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale32_initdepth0.25_v1/models/model_18_3294_best_d1_0.91866.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 32 \
#     --eval_mono
# model_17_611_best_abs_rel_0.09428:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.756  &   4.461  &   0.179  &   0.896  &   0.965  &   0.983  \\
# model_18_3294_best_d1_0.91866:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.764  &   4.467  &   0.179  &   0.897  &   0.965  &   0.983  \\

# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale16_initdepth0.25_v1/models/model_16_2428_best_abs_rel_0.09490.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 16 \
#     --eval_mono
# model_16_2428_best_abs_rel_0.09490:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.739  &   4.468  &   0.179  &   0.897  &   0.965  &   0.983  \\


# -------------------------------------- v2 start -------------------------------------------


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale32_initdepth0.25_v2/models/model_19_1977_best_abs_rel_0.09636.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 32 \
#     --eval_mono
# model_19_1977_best_abs_rel_0.09636:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.729  &   4.460  &   0.178  &   0.897  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale16_initdepth0.25_v2/models/model_18_2294_best_abs_rel_0.09349.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 16 \
#     --eval_mono
# model_18_2294_best_abs_rel_0.09349:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.750  &   4.467  &   0.180  &   0.896  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale8_initdepth0.25_v2/models/model_18_2794_best_abs_rel_0.09399.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 8 \
#     --eval_mono
# model_18_2794_best_abs_rel_0.09399:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.103  &   0.757  &   4.504  &   0.180  &   0.895  &   0.965  &   0.983  \\

# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale4_initdepth0.25_v2/models/model_19_977_best_abs_rel_0.09248.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 4 \
#     --eval_mono
# model_19_977_best_abs_rel_0.09248:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.761  &   4.484  &   0.180  &   0.896  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale2_initdepth0.25_v2/models/model_18_2294_best_abs_rel_0.09203.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 2 \
#     --eval_mono
# model_18_2294_best_abs_rel_0.09203:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.100  &   0.724  &   4.461  &   0.178  &   0.898  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=5 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_baseline_scale0_initdepth0.25_v2/models/model_19_1477_best_abs_rel_0.09282.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_1477_best_abs_rel_0.09282
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.100  &   0.725  &   4.420  &   0.177  &   0.900  &   0.966  &   0.983  \\


# -------------------------------------- v2 end -------------------------------------------

# ---------------------------------------------------- baseline+gs start ----------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initdepthloss0.5/models/model_19_2477_best_abs_rel_0.09695.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_2477_best_abs_rel_0.09695:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.747  &   4.459  &   0.178  &   0.899  &   0.966  &   0.983  \\


# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initpeloss0.5_retrain/models/model_18_3294_best_abs_rel_0.09337.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_18_3294_best_abs_rel_0.09337:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.747  &   4.484  &   0.180  &   0.895  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initdepthpeloss0.5/models/model_19_2477_best_abs_rel_0.09381.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_2477_best_abs_rel_0.09381:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.750  &   4.473  &   0.180  &   0.897  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_featbeforetrans_initdepthpeloss0.5/models/model_19_2977_best_abs_rel_0.09525.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_2977_best_abs_rel_0.09525:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.103  &   0.736  &   4.479  &   0.181  &   0.894  &   0.964  &   0.983  \\

# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_featbehindtrans_initpeloss0.5/models/model_17_3111_best_abs_rel_0.09734.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_17_3111_best_abs_rel_0.09734:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.103  &   0.755  &   4.466  &   0.180  &   0.892  &   0.964  &   0.983  \\


# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initpeloss0.25/models/model_19_977_best_abs_rel_0.09413.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_977_best_abs_rel_0.09413:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.100  &   0.715  &   4.411  &   0.178  &   0.899  &   0.966  &   0.983  \\


# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_featbeforetrans_initpeloss0.25/models/model_19_977_best_abs_rel_0.09276.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_977_best_abs_rel_0.09276:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.729  &   4.444  &   0.179  &   0.896  &   0.965  &   0.983  \\

# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_featbehindtrans_initpeloss0.25/models/model_19_477_best_abs_rel_0.09347.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_477_best_abs_rel_0.09347:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.739  &   4.428  &   0.178  &   0.898  &   0.966  &   0.983  \\


# CUDA_VISIBLE_DEVICES=7 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initpeloss0.25_add/models/model_18_794_best_abs_rel_0.09326.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_18_794_best_abs_rel_0.09326:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.752  &   4.489  &   0.180  &   0.896  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=7 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initpeloss0.25_noadd/models/model_17_3111_best_abs_rel_0.19563.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_17_3111_best_abs_rel_0.19563:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.184  &   1.597  &   6.334  &   0.260  &   0.743  &   0.903  &   0.963  \\


# CUDA_VISIBLE_DEVICES=7 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initpeloss0.25_Fit3Dv1/models/model_19_2477_best_abs_rel_0.09325.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_2477_best_abs_rel_0.09325:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.103  &   0.776  &   4.496  &   0.179  &   0.895  &   0.965  &   0.983  \\



# CUDA_VISIBLE_DEVICES=7 \
# python evaluate_depth_gs_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS-Depth_scale0_initpeloss0.25_Fit3Dv1_ds2/models/model_19_977_best_abs_rel_0.10454.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_977_best_abs_rel_0.10454:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.108  &   0.882  &   4.772  &   0.188  &   0.886  &   0.961  &   0.981  \\

# ---------------------------------------------------- baseline+gs end ----------------------------------------------------


# ---------------------------------------------------- baseline+gs+inter start ----------------------------------------------------
# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_inter_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS_Depth_scale0_initpe0.25_inter0.1/models/model_18_2294_best_abs_rel_0.09257.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_18_2294_best_abs_rel_0.09257:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.755  &   4.466  &   0.179  &   0.895  &   0.965  &   0.983  \\

# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_inter_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS_Depth_scale0_initpe0.25_inter0.25/models/model_19_2477_best_abs_rel_0.09404.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_19_2477_best_abs_rel_0.09404:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.725  &   4.440  &   0.177  &   0.897  &   0.966  &   0.984  \\


# CUDA_VISIBLE_DEVICES=9 \
# python evaluate_depth_gs_inter_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS_Depth_scale0_initpe0.25_inter0.5/models/model_19_477_best_abs_rel_0.09228.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# # model_19_477_best_abs_rel_0.09228:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.102  &   0.723  &   4.441  &   0.179  &   0.895  &   0.965  &   0.983  \\


# CUDA_VISIBLE_DEVICES=8 \
# python evaluate_depth_gs_inter_bestmodel.py \
#     --data_path /data0/wuhaifeng/dataset/KITTI_dataset/raw_data \
#     --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/GS-Depth/models/GS_Depth_scale0_initpe0.25_inter0.25/models/model_18_2794_best_abs_rel_0.09235.pth \
#     --eval_split eigen \
#     --use_gs \
#     --gs_scale 0 \
#     --eval_mono
# model_18_2794_best_abs_rel_0.09235:
#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.101  &   0.741  &   4.436  &   0.178  &   0.897  &   0.965  &   0.984  \\
# ---------------------------------------------------- baseline+gs+inter end ----------------------------------------------------