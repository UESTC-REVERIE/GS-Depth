# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

from tqdm import tqdm

# 改动中代码：
# 编码器：[ResNet18, HRNet18]，
# 使用init_decoder的输出[独立，通过光栅化同一高斯表达]预测多尺度高斯，
# 由高斯光栅化的特征使用[HRNet18p特征融合解码器输出最终尺度深度图，Monodepth2的Decoder输出多尺度深度图]，
# init depth[参与，未参与]loss计算

# gs_baseline代码：
# 使用HRNet18作为编码器，
# 使用编码器输出独立预测多尺度高斯，
# 由高斯光栅化的特征使用HRNet18p特征融合解码器输出最终尺度深度图，
# init depth参与loss计算
class Trainer:
    def __init__(self, options):
        # torch.autograd.set_detect_anomaly(True)
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.opt.no_cuda else "cpu")

        # self.num_scales = len(self.opt.scales)
        self.num_scales = 4
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # 存在frame_ids不为[0]的情况或非stereo(monocular)的情况，需要使用pose_net，default: True
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        # region Depth模型初始化
        # encoder 
        # self.models["encoder"] = networks.hrnet18(self.opt.weights_init == "pretrained")
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        # init decoder(monodepth2) 初始解码器，特征融合并解码得到初始深度图
        self.models["init_decoder"] = networks.InitDepthDecoder(
            num_ch_enc=self.models["encoder"].num_ch_enc,
            scales=self.opt.scales
        )
        self.models["init_decoder"].to(self.device)
        self.parameters_to_train += list(self.models["init_decoder"].parameters())

        if self.opt.use_gs:
            # gs feature leverage
            self.models["gs_leverage"] = networks.GaussianFeatureLeverage(
                # TODO concat初始的深度图提供结构信息
                num_ch_in=self.models["init_decoder"].num_ch_dec, 
                scales=self.opt.scales,
                height=self.opt.height, width=self.opt.width,
                # TODO 修改光栅器默认输出维度不为64
                leveraged_feat_ch=64, 
                min_depth=self.opt.min_depth, max_depth=self.opt.max_depth
            )
            self.models["gs_leverage"].to(self.device)
            self.parameters_to_train += list(self.models["gs_leverage"].parameters())
            
            # gs feature decoder
            self.models["gs_decoder"] = networks.GSDepthDecoder(
                num_ch_enc=self.models["gs_leverage"].num_ch_out,
                scales=self.opt.scales
            )
            self.models["gs_decoder"].to(self.device)
            self.parameters_to_train += list(self.models["gs_decoder"].parameters())

        # endregion
        
        # region Pose模型初始化
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet": # default
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
        # endregion
        
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        #创建优化器
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        # region 加载预训练模型
        if self.opt.pretrained_model_path is not None:
            self.load_pretrained_model(self.opt.pretrained_model_path, self.opt.pretrained_models_to_load, self.opt.pretrained_frozen)
        # endregion
        
        # region retrain
        if self.opt.load_weights_folder is not None:
            # self.load_model()
            self.load_model_checkpoints()
            self.model_optimizer.param_groups[0]['lr'] = self.opt.learning_rate
            self.model_optimizer.param_groups[0]['initial_lr'] = self.opt.learning_rate
        # endregion

        # default scheduler_step_size=15, scheduler_weight=0.1, 每scheduler_step_size个epoch，学习率乘以scheduler_weight
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, self.opt.scheduler_weight)        

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # region 数据处理
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "nyu": datasets.NYURAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        # default self.opt.split : eigen_zhou
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_val=True, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        # endregion
        
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        # self.project_depth = {}
        # self.detail_guide = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth_removedxy(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D_removedxy(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)


        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
        self.best_eval_measures_lower_better = torch.zeros(4).cpu() + 1e3
        self.best_eval_measures_higher_better = torch.zeros(3).cpu()
        self.best_eval_epochs = np.zeros(7, dtype=np.int32)
        self.best_eval_batch_idxs = np.zeros(7, dtype=np.int32)


        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.epoch >= 15:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            #前向传播
            outputs, losses = self.process_batch('train', inputs)
            #清零梯度
            self.model_optimizer.zero_grad()
            #反向传播
            losses["loss"].backward()

            #更新参数
            self.model_optimizer.step()
 
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            # late_phase = self.step % 2000 == 0

            # if early_phase or late_phase:
            if batch_idx % 10 == 0:
                self.log_time(batch_idx, duration, losses)

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)

                # self.log("train", inputs, outputs, losses)
                # self.val()

            # self.step += 1

            # if self.step % 100 == 0:
            #     self.save_model()
            
            if self.epoch < 14 and self.step % 1000 == 0:
                self.val(batch_idx)
            elif self.epoch >= 14 and self.step % self.opt.eval_frequency_best == 0:
                self.val(batch_idx)
            self.step += 1
            
        self.model_lr_scheduler.step()

    def process_batch(self, mode, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # features_HiS = self.models["encoder"](inputs["color_HiS", 0, 0])
            # outputs["out_HiS"] = self.models["depth"](features_HiS)
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            inv_K = []
            K = []
            if self.opt.use_gs: # 得到inv_k,k列表用于高斯
                for scale in self.opt.scales:
                    inv_K.append(inputs[("inv_K", scale)].to(self.device))
                    K.append(inputs[("K", scale)].to(self.device))

            features = self.models["encoder"](inputs["color", 0, 0]) if mode == 'val' else self.models["encoder"](inputs["color_aug", 0, 0])

            outputs, gs_input_features = self.models["init_decoder"](features)
            if self.opt.use_gs:
                leveraged_features = self.models["gs_leverage"](
                    init_features = gs_input_features,
                    init_disps = list(outputs["init_disp", i] for i in self.opt.scales),
                    inv_K = inv_K, K = K
                )
                
                outputs.update(self.models["gs_decoder"](leveraged_features))
            else :
                for i in self.opt.scales:
                    outputs[("disp", i)] = outputs[("init_disp", i)]

        if mode == 'train':
            if self.opt.predictive_mask:
                outputs["predictive_mask"] = self.models["predictive_mask"](features)

            if self.use_pose_net:
                outputs.update(self.predict_poses(inputs, features))

            self.generate_images_pred(inputs, outputs)

            losses = self.compute_losses(inputs, outputs)

            return outputs, losses
        else:
            return outputs

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_pose_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self, batch_idx):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        eval_measures = torch.zeros(8).cuda(device=self.device)

        with torch.no_grad():
            for _, inputs in enumerate(tqdm(self.val_loader)):
                outputs = self.process_batch('val', inputs)

                eval_measures += self.compute_depth_losses(inputs, outputs)

                # del inputs, outputs

                # self.log("val", inputs, outputs, losses)
        # compute metrics
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[7].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples '.format(int(cnt)), ', post_process: ', self.opt.post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('abs_rel', 'sq_rel', 'rms',
                                                                            'log_rms', 'd1', 'd2', 'd3'))
        for i in range(7):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print()
        for i in range(7):
            measure = eval_measures_cpu[i]
            is_best = False
            if i < 4 and measure < self.best_eval_measures_lower_better[i]:
                old_best = self.best_eval_measures_lower_better[i].item()
                self.best_eval_measures_lower_better[i] = measure.item()
                is_best = True
            elif i >= 4 and measure > self.best_eval_measures_higher_better[i-4]:
                old_best = self.best_eval_measures_higher_better[i-4].item()
                self.best_eval_measures_higher_better[i-4] = measure.item()
                is_best = True
            if is_best:
                old_best_epoch = self.best_eval_epochs[i]
                old_best_batch_idx = self.best_eval_batch_idxs[i]
                old_best_name = 'model_{}_{}_best_{}_{:.5f}.pth'.format(old_best_epoch, old_best_batch_idx, eval_metrics[i], old_best)
                model_path = os.path.join(self.log_path, "models", old_best_name)
                if os.path.exists(model_path):
                    command = 'rm {}'.format(model_path)
                    os.system(command)
                self.best_eval_epochs[i] = self.epoch
                self.best_eval_batch_idxs[i] = batch_idx
                model_save_name = 'model_{}_{}_best_{}_{:.5f}.pth'.format(self.epoch, batch_idx, eval_metrics[i], measure)
                
                print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                checkpoint = {'best_epoch': self.epoch,
                            'best_batch': batch_idx,
                                'encoder': self.models["encoder"].state_dict(),
                                'init_decoder': self.models["init_decoder"].state_dict(),
                                'gs_leverage': self.models["gs_leverage"].state_dict() if self.opt.use_gs else None,
                                'gs_decoder': self.models["gs_decoder"].state_dict() if self.opt.use_gs else None,
                                'pose_encoder': self.models["pose_encoder"].state_dict(),
                                'pose': self.models["pose"].state_dict(),
                                'optimizer': self.model_optimizer.state_dict(),
                                'best_eval_measures_higher_better': self.best_eval_measures_higher_better,
                                'best_eval_measures_lower_better': self.best_eval_measures_lower_better,
                                'optimizer': self.model_optimizer.state_dict(),
                                }
                torch.save(checkpoint, os.path.join(self.log_path, "models", model_save_name))

        del inputs, outputs, eval_measures

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            # disp_HiS = outputs["out_HiS"][("disp", scale)]
            disp = outputs[("disp", scale)]

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # outputs[("disp", scale)] = disp
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth
            outputs[("disp_full_res", scale)] = disp
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        train_save_path = os.path.join(os.path.dirname(__file__), "models/{}/results/".format(self.opt.model_name))
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)] # 高斯特征预测的视差图
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            if self.step % 500 == 0:
                # TODO 保存变化尺度后的depthmap
                disps = {}
                disps["gs_disp"] = outputs[("disp_full_res", scale)]
                disps["init_disp"] = outputs[("init_disp", scale)]
                _depth = outputs[("depth", 0, scale)]
                self.save_img(_depth,disps,target,scale,train_save_path)

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:

                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # identity_reprojection_loss += torch.randn(
                #     identity_reprojection_loss.shape).cuda() * 0.00001
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            
            else:
                combined = reprojection_loss
                

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            
            if self.opt.use_init_smoothLoss:
                # print("Using init smooth loss")
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)
                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        
        return losses
    
    
    def compute_depth_losses(self, inputs, outputs):
        eval_measures = torch.zeros(8).to(self.device)
        if self.opt.dataset == "nyu":
            MIN_DEPTH = 0.01
            MAX_DEPTH = 10
        else:
            MIN_DEPTH = 1e-3
            MAX_DEPTH = 80
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        # depth_pred = outputs[("depth", 0, 0)]
        depth_gt = inputs["depth_gt"].to(self.device)
        gt_height, gt_width = depth_gt.shape[2:]
        disp = outputs[("disp", 0)]
        disp_pred, _ = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        # disp_pred = torch.clamp(F.interpolate(
        #     disp_pred, [gt_height, gt_width], mode="bilinear", align_corners=False), 1e-3, 80)
        disp_pred = F.interpolate(disp_pred, [gt_height, gt_width], mode="bilinear", align_corners=False)
        depth_pred = 1 / disp_pred
        # depth_pred = depth_pred.detach()

        for i in range(depth_pred.shape[0]):
            depth_gt_each = depth_gt[i].squeeze()
            depth_pred_each = depth_pred[i].squeeze()

            if self.opt.dataset == "eigen":
                mask = torch.logical_and(depth_gt_each > MIN_DEPTH, depth_gt_each < MAX_DEPTH)
                # print(mask.shape)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop = torch.from_numpy(crop)
                crop_mask = torch.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = torch.logical_and(mask, crop_mask)

            elif self.opt.dataset == 'nyu':
                mask = torch.logical_and(depth_gt_each > MIN_DEPTH, depth_gt_each < MAX_DEPTH)
                crop_mask = torch.zeros(mask.shape).to(self.device)
                # crop = np.array([45, 471, 41, 601]).astype(np.int32)
                # crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                crop_mask[self.dataset.default_crop[2]:self.dataset.default_crop[3], \
                self.dataset.default_crop[0]:self.dataset.default_crop[1]] = 1
                mask = torch.logical_and(mask, crop_mask)

            else:
                mask = depth_gt_each > 0

        
            depth_gt_each = depth_gt_each[mask]
            depth_pred_each = depth_pred_each[mask]
            # print(len(depth_gt_each), len(depth_pred_each))
            if not self.opt.disable_median_scaling:
                depth_pred_each *= torch.median(depth_gt_each) / torch.median(depth_pred_each)
            # depth_pred_each = torch.clamp(depth_pred_each, min=1e-3, max=80)
            depth_pred_each[depth_pred_each < MIN_DEPTH] = MIN_DEPTH
            depth_pred_each[depth_pred_each > MAX_DEPTH] = MAX_DEPTH
            
            measures = compute_depth_errors(depth_gt_each, depth_pred_each)
            eval_measures[:7] += torch.tensor(measures).cuda(device=self.device)
            # eval_measures[:7] += measures
            eval_measures[7] += 1

        return eval_measures




    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
    
        print_string = "epo {:>3} | bat {:>6} | ex/s: {:5.1f}" + \
            " | loss: {:.5f} | te: {} | tl: {} | lr: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"].cpu().data,
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.state_dict()['param_groups'][0]['lr']))

    def save_img(self, depth, disps, target, scale, train_save_path):
        model_name = self.opt.model_name
        for disp_type, disp in disps.items():
            # 保存视差图（伪彩色图，原逻辑）
            disp_np = disp[0].squeeze().cpu().detach().numpy()
            vmax = np.percentile(disp_np, 95)
            # 打印数据统计信息（调试用）
            print(f"{disp_type}: Scale {scale}: min={disp_np.min()}, max={disp_np.max()}, mean={disp_np.mean()}")
            
            normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
            colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
            im = Image.fromarray(colormapped_im)
            im.save(os.path.join(train_save_path, f"{model_name}_{disp_type}_scale{scale}.png"))

        # 保存深度图（16位灰度图）
        depth_np = depth[0].squeeze().cpu().detach().numpy()
        depth_normalized = cv2.normalize(depth_np, None, 0, 65535, cv2.NORM_MINMAX)
        depth_16u = depth_normalized.astype(np.uint16)
        cv2.imwrite(os.path.join(train_save_path, f"{model_name}_depth_scale{scale}.png"), depth_16u)

        # 保存原图
        target_ = target[0].cpu().detach().numpy() * 255
        target_ = np.transpose(target_, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(target_).save(os.path.join(train_save_path, f"{model_name}_image.png"))
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        # save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch, self.step))
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model_checkpoints(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        # assert os.path.isdir(self.opt.load_weights_folder), \
        #     "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # if self.opt.load_weights_folder != '':
        if os.path.isfile(self.opt.load_weights_folder):
            print("== Loading checkpoint '{}'".format(self.opt.load_weights_folder))
            checkpoint = torch.load(self.opt.load_weights_folder)

            for n in self.opt.models_to_load:
                print("Loading {} weights...".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = checkpoint["{}".format(n)]
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

            # loading adam state
            print("Loading Adam weights")
            self.model_optimizer.load_state_dict(checkpoint['optimizer'])
            # loading best values
            self.best_epoch = checkpoint['best_epoch']
            self.best_batch = checkpoint['best_batch']
            self.best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
            self.best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
        else:
            print("== No checkpoint found at '{}'".format(self.opt.load_weights_folder))
        del checkpoint
        
    def load_pretrained_model(self, path, models_to_load, frozen):
        """Load model(s) from disk
        """
        path = os.path.expanduser(path)

        # false时断言触发
        assert not os.path.isdir(path), \
            "--pretrained_model_path should be a model file, but {} is a directory".format(path)

        if os.path.isfile(path):
            print("loading model from {}".format(path))
            checkpoint = torch.load(path)

            for n in models_to_load:
                print("Loading {} weights...".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = checkpoint["{}".format(n)]
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
                
                if frozen:
                    # 冻结模型参数（不参与梯度更新）
                    for param in self.models[n].parameters():
                        param.requires_grad = False
                    # 从优化器参数列表中移除冻结参数
                    self.parameters_to_train = [
                        p for p in self.parameters_to_train 
                        if p not in set(self.models[n].parameters())
                    ]  # 确保优化器不更新冻结层
                    print(f"Frozen the pretrained model: {n}")
            del checkpoint  
        else:
            print("== No checkpoint found at '{}'".format(self.opt.load_weights_folder))

