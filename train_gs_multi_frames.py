# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer_gs_multi_frames import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()




# retrain

# CUDA_VISIBLE_DEVICES=2 python train.py --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --model_name RA-Depth-retrain --scales 0 --png --log_dir models --split eigen_zhou


# CUDA_VISIBLE_DEVICES=0 python train.py --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --model_name RA-Depth-retrain-remove1 --scales 0 --png --log_dir models --split eigen_wu
# CUDA_VISIBLE_DEVICES=1 python train.py --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --model_name RA-Depth-retrain-remove2 --scales 0 --batch_size 1 --png --log_dir models --split eigen_wu
# CUDA_VISIBLE_DEVICES=5 python train.py --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --model_name RA-Depth-retrain-remove3 --scales 0 --png --log_dir models --split eigen_zhou



# CUDA_VISIBLE_DEVICES=0 python train.py --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/RA-Depth-main/models/RA-Depth-retrain_eigen_wu_5frame/models/weights_4 --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --model_name RA-Depth-retrain_eigen_wu_5frame --scales 0 --png --log_dir models --split eigen_zhou


# 
# CUDA_VISIBLE_DEVICES=3 python train.py --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --model_name RA-Depth-retrain_eigen_wu_5frame --scales 0 --png --log_dir models --split eigen_zhou