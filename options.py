# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
          self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

          # PATHS
          self.parser.add_argument("--data_path",
                                   type=str,
                                   help="path to the training data",
                                   # default=os.path.join(file_dir, "kitti_data")
                                   default='I:/datasets/depthDatasets/KITTI/rawdata/'
                                   )
          self.parser.add_argument("--log_dir",
                                   type=str,
                                   help="log directory",
                                   default=os.path.join(os.path.expanduser("~"), "tmp"))
          self.parser.add_argument("--eval_output_dir",
                                   type=str,
                                   help="eval output directory",
                                   default=os.path.join(os.path.expanduser("~"), "tmp"))

          # TRAINING options
          self.parser.add_argument("--model_name",
                                   type=str,
                                   help="the name of the folder to save the model in",
                                   default="mdp")
          self.parser.add_argument("--split",
                                   type=str,
                                   help="which training split to use",
                                   choices=["eigen_wu", "eigen_small", "eigen_zhou", "eigen_full", "odom", "benchmark", "tmp", "eigen_and_odom", "cityscapes_preprocessed", "nyu"],
                                   default="eigen_zhou")
          self.parser.add_argument("--num_layers",
                                   type=int,
                                   help="number of resnet layers",
                                   default=18,
                                   choices=[18, 34, 50, 101, 152])
          self.parser.add_argument("--dataset",
                                   type=str,
                                   help="dataset to train on",
                                   default="kitti",
                                   choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "kitti_stereo", "cityscapes_preprocessed", "nyu"])
          self.parser.add_argument("--png",
                                   help="if set, trains from raw KITTI png files (instead of jpgs)",
                                   action="store_true")
          self.parser.add_argument("--height",
                                   type=int,
                                   help="input image height",
                                   default=192)
          self.parser.add_argument("--width",
                                   type=int,
                                   help="input image width",
                                   default=640)
          self.parser.add_argument("--disparity_smoothness",
                                   type=float,
                                   help="disparity smoothness weight",
                                   default=1e-3)
          self.parser.add_argument("--scales",
                                   nargs="+",
                                   type=int,
                                   help="scales used in the loss",
                                   default=[0, 1, 2, 3])
          self.parser.add_argument("--min_depth",
                                   type=float,
                                   help="minimum depth",
                                   default=0.1)
          self.parser.add_argument("--max_depth",
                                   type=float,
                                   help="maximum depth",
                                   default=100.0)
          self.parser.add_argument("--use_stereo",
                                   help="if set, uses stereo pair for training",
                                   action="store_true")
          self.parser.add_argument("--frame_ids",
                                   nargs="+",
                                   type=int,
                                   help="frames to load",
                                   default=[0, -1, 1])
          self.parser.add_argument("--eval_frequency_best",
                                   type=int,
                                   help="eval frequency of best model",
                                   default=500)
          self.parser.add_argument("--use_gs",
                                   help="if set, uses 3d gs",
                                   action="store_true")
          self.parser.add_argument("--use_init_smoothLoss",
                                   help="if set, uses 3d gs",
                                   action="store_true")
          self.parser.add_argument("--gs_scale",
                                   type=int,
                                   help="use gs at one scale",
                                   default=32)
          self.parser.add_argument("--gs_num_per_pixel",
                                   type=int,
                                   help="the number of gs per pixel",
                                   default=1)
          self.parser.add_argument("--loss_gs_weight",
                                   type=float,
                                   help="weight of dx depth",
                                   default=0)
          self.parser.add_argument("--loss_init_weight",
                                   type=float,
                                   help="weight of dx depth",
                                   default=0)
          self.parser.add_argument("--loss_perception_weight",
                                   type=float,
                                   help="weight of dx depth",
                                   default=1e-3)
          self.parser.add_argument("--use_interframe_consistency",
                                   help="if set, uses 3d gs",
                                   action="store_true")
          self.parser.add_argument("--loss_interframe_weight",
                                   type=float,
                                   help="weight of dx depth",
                                   default=0)       
          
          
          # OPTIMIZATION options
          self.parser.add_argument("--batch_size",
                                   type=int,
                                   help="batch size",
                                   default=12)
          self.parser.add_argument("--learning_rate",
                                   type=float,
                                   help="learning rate",
                                   default=1e-4)
          self.parser.add_argument("--start_epoch",
                                   type=int,
                                   help="number of epochs",
                                   default=0)
          self.parser.add_argument("--num_epochs",
                                   type=int,
                                   help="number of epochs",
                                   default=20)
          self.parser.add_argument("--scheduler_step_size",
                                   type=int,
                                   help="step size of the scheduler",
                                   default=15)
          self.parser.add_argument("--scheduler_weight",
                                   type=float,
                                   help="scheduler weight",
                                   default=0.1)

          # ABLATION options
          self.parser.add_argument("--v1_multiscale",
                                   help="if set, uses monodepth v1 multiscale",
                                   action="store_true")
          self.parser.add_argument("--avg_reprojection",
                                   help="if set, uses average reprojection loss",
                                   action="store_true")
          self.parser.add_argument("--disable_automasking",
                                   help="if set, doesn't do auto-masking",
                                   action="store_true")
          self.parser.add_argument("--predictive_mask",
                                   help="if set, uses a predictive masking scheme as in Zhou et al",
                                   action="store_true")
          self.parser.add_argument("--no_ssim",
                                   help="if set, disables ssim in the loss",
                                   action="store_true")
          self.parser.add_argument("--weights_init",
                                   type=str,
                                   help="pretrained or scratch",
                                   default="pretrained",
                                   choices=["pretrained", "scratch"])
          self.parser.add_argument("--pose_model_input",
                                   type=str,
                                   help="how many images the pose network gets",
                                   default="pairs",
                                   choices=["pairs", "all"])
          self.parser.add_argument("--pose_model_type",
                                   type=str,
                                   help="whether or not using shared encoder for both depth and pose",
                                   default="separate_resnet",
                                   choices=["posecnn", "separate_resnet", "shared"])
          self.parser.add_argument("--use_interframe_normal_loss",
                                   help="if set, uses inter frame normal loss",
                                   action="store_true")
          self.parser.add_argument("--use_interframe_distance_loss",
                                   help="if set, uses inter frame distance loss",
                                   action="store_true")
          self.parser.add_argument("--use_self_normal_loss",
                                   help="if set, uses inter frame distance loss",
                                   action="store_true")
          self.parser.add_argument("--use_gnpd_loss",
                                   help="if set, uses inter frame distance loss",
                                   action="store_true")
          self.parser.add_argument("--use_self_normal_loss_epoch",
                                   type=int,
                                   default=0)
          self.parser.add_argument("--use_plannar_loss",
                                   help="if set, uses inter frame distance loss",
                                   action="store_true")
          self.parser.add_argument("--use_edge_smooth_loss",
                                   help="if set, uses inter frame distance loss",
                                   action="store_true")
          self.parser.add_argument("--use_res_disp",
                                   help="if set, uses inter frame distance loss",
                                   action="store_true")
          self.parser.add_argument("--disable_smooth_loss",
                                   help="if set, not use smooth loss",
                                   action="store_true")
          
          # SYSTEM options
          self.parser.add_argument("--no_cuda",
                                   help="if set disables CUDA",
                                   action="store_true")
          self.parser.add_argument("--num_workers",
                                   type=int,
                                   help="number of dataloader workers",
                                   default=12)

          # LOADING to retrain options
          self.parser.add_argument("--load_weights_path",
                                   type=str,
                                   help="path of model to load for retrain or eval")
          self.parser.add_argument("--load_weights_folder",
                                   type=str,
                                   help="name of model to load")
          self.parser.add_argument("--models_to_load",
                                   nargs="+",
                                   type=str,
                                   help="models to load",
                                   default=["encoder", "depth", "pose_encoder", "pose"])
          self.parser.add_argument("--resume_checkpoint_path", 
                                   type=str, 
                                   default=None,
                                   help="the path of checkpoint for resume interrupted training")
          # LOGGING pretrained models options
          self.parser.add_argument("--pretrained_model_path",
                                   type=str,
                                   help="the path of pretrained model to load",)
          self.parser.add_argument("--pretrained_models_to_load",
                                   nargs="+",
                                   type=str,
                                   help="pretrained models",
                                   default=None)
          self.parser.add_argument("--pretrained_frozen",
                                   help="if set frozen the pretrained models loaded",
                                   action="store_true")
          # LOGGING options
          self.parser.add_argument("--log_frequency",
                                   type=int,
                                   help="number of batches between each tensorboard log",
                                   default=250)
          self.parser.add_argument("--save_frequency",
                                   type=int,
                                   help="number of epochs between each save",
                                   default=1)

          # EVALUATION options
          self.parser.add_argument("--eval_stereo",
                                   help="if set evaluates in stereo mode",
                                   action="store_true")
          self.parser.add_argument("--eval_mono",
                                   help="if set evaluates in mono mode",
                                   action="store_true")
          self.parser.add_argument("--disable_median_scaling",
                                   help="if set disables median scaling in evaluation",
                                   action="store_true")
          self.parser.add_argument("--pred_depth_scale_factor",
                                   help="if set multiplies predictions by this number",
                                   type=float,
                                   default=1)
          self.parser.add_argument("--ext_disp_to_eval",
                                   type=str,
                                   help="optional path to a .npy disparities file to evaluate")
          self.parser.add_argument("--eval_split",
                                   type=str,
                                   default="eigen",
                                   choices=[
                                        "gecaoji", "eigen_wu", "eigen_small", "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "make3d", "improved_eigen", "cityscapes"],
                                   help="which split to run eval on")
          self.parser.add_argument("--save_pred_disps",
                                   help="if set saves predicted disparities",
                                   action="store_true")
          self.parser.add_argument("--no_eval",
                                   help="if set disables evaluation",
                                   action="store_true")
          self.parser.add_argument("--eval_eigen_to_benchmark",
                                   help="if set assume we are loading eigen results from npy but "
                                        "we want to evaluate using the new benchmark.",
                                   action="store_true")
          self.parser.add_argument("--eval_out_dir",
                                   help="if set will output the disparities to this folder",
                                   type=str)
          self.parser.add_argument("--post_process",
                                   help="if set will perform the flipping post processing "
                                        "from the original monodepth paper",
                                   action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


#CUDA_VISIBLE_DEVICES=0 python train.py --model_name RA-Depth --scales 0 --png --log_dir models --data_path /test/datasets/Kitti/Kitti_raw_data



