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
        self.models["encoder"] = networks.hrnet18(self.opt.weights_init == "pretrained")
        # self.models["encoder"] = networks.ResnetEncoder(
        #     self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        # init decoder(monodepth2) 初始解码器，特征融合并解码得到初始深度图
        self.models["init_decoder"] = networks.InitDepthDecoder(
            num_ch_enc=self.models["encoder"].num_ch_enc,
            scales=self.opt.scales
        )
        # self.models["init_decoder"] = networks.HRDepthDecoder(
        #     num_ch_enc=self.models["encoder"].num_ch_enc,
        #     scales=self.opt.scales
        # )
        self.models["init_decoder"].to(self.device)
        self.parameters_to_train += list(self.models["init_decoder"].parameters())

        if self.opt.use_gs:
            # gs feature leverage
            self.models["gs_leverage"] = networks.GaussianFeatureLeverage(
                # TODO concat初始的深度图提供结构信息
                num_ch_in=self.models["encoder"].num_ch_enc, 
                scales=self.opt.scales,
                height=self.opt.height, width=self.opt.width,
                # TODO 修改光栅器默认输出维度不为64
                leveraged_feat_ch=64, 
                min_depth=self.opt.min_depth, max_depth=self.opt.max_depth,
                num_ch_concat = 3 + 1 * 4, 
                gs_scale=self.opt.gs_scale,
                gs_num_pixel=self.opt.gs_num_per_pixel
            )
            self.models["gs_leverage"].to(self.device)
            self.parameters_to_train += list(self.models["gs_leverage"].parameters())
            
            # gs feature decoder
            self.models["gs_decoder"] = networks.HRDepthDecoder(
                num_ch_enc=self.models["gs_leverage"].num_ch_out,
                scales=self.opt.scales
            )
            # self.models["gs_decoder"] = networks.HRNetDepthDecoder(
            #     num_ch_enc=self.models["gs_leverage"].num_ch_out,
            #     scales=self.opt.scales
            # )
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
        if self.opt.load_weights_path is not None:
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
        start_epoch = self.opt.start_epoch
        self.epoch = 0
        self.step = 0
        
        if self.opt.resume_checkpoint_path is not None:
            self.load_checkpoint_resume(self.opt.resume_checkpoint_path)
            start_epoch = self.epoch + 1
        
        self.start_time = time.time()
        for self.epoch in range(start_epoch, self.opt.num_epochs):
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
        T = {}
        inv_K = []
        K = []
        if mode == "val":
            input_colors = list(inputs["color", 0, i] for i in range(6))
            lookup_input_colors = list(inputs["color", -1, i] for i in range(6))
        else:
            input_colors = list(inputs["color_aug", 0, i] for i in range(6))
            lookup_input_colors = list(inputs["color_aug", -1, i] for i in range(6))
            
        features = self.models["encoder"](input_colors[0])
        lookup_features = self.models["encoder"](lookup_input_colors[0])
        if mode == 'train' and self.use_pose_net:
            outputs = self.predict_poses(inputs, self.num_pose_frames)
            for frame_id in self.opt.frame_ids[1:]:
                T[frame_id] = outputs[("cam_T_cam", 0, frame_id)] 
                
        with torch.no_grad(): # following manydepth, 构建cost volume的相对位姿不计算梯度
            lookup_T = self.predict_poses(inputs, 1)
            
        if self.opt.use_gs: # 得到inv_k,k列表用于高斯
            for scale in range(6):
                inv_K.append(inputs[("inv_K", scale)].to(self.device))
                K.append(inputs[("K", scale)].to(self.device))
        
        init_outputs, _ , _ = self.models["init_decoder"](features)
        for k, v in init_outputs.items():
            outputs[("init_disp", k[1])] = v
        
        if self.opt.use_gs:
            leveraged_features, gs_features = self.models["gs_leverage"](
                init_features = features,
                init_disps = list(outputs["init_disp", i] for i in self.opt.scales),
                colors = input_colors,
                inv_K = inv_K, K = K,
                # T = T if mode=="train" else None,
                lookup_features = lookup_features,
                lookup_T = lookup_T
            )

            outputs.update(self.models["gs_decoder"](leveraged_features))
        else :
            for i in self.opt.scales:
                outputs[("disp", i)] = outputs[("init_disp", i)]

        if mode == 'train':
            losses={}
            if self.opt.predictive_mask:
                outputs["predictive_mask"] = self.models["predictive_mask"](features)
            if self.opt.use_gs:
                # init_disp_list = list(outputs[("init_disp", i)] for i in range(4))
                # self.generate_images_pred(inputs, outputs, init_disp_list, 4, "init_")
                # losses["init_loss"] = self.compute_losses(inputs, outputs, init_disp_list, 4, True, "init_") * self.opt.loss_init_weight
                
                disp_list = list(outputs[("disp", i)] for i in range(4))
                self.generate_images_pred(inputs, outputs, disp_list, 4)
                losses["gs_loss"] = self.compute_losses(inputs, outputs, disp_list, 4, self.opt.use_init_smoothLoss) * self.opt.loss_gs_weight
                
                # # 相邻帧重新预测高斯分布光栅化得到的特征
                # nearby_gs_features = self.generate_features_pred(inputs, outputs, inv_K, K)
                # # 计算高斯一致性损失
                # losses["gs_ft_loss"] = self.compute_perceptional_loss(nearby_gs_features, gs_features) * self.opt.loss_perception_weight
                
                # losses["loss"] = losses["init_loss"] + losses["gs_loss"]
                # losses["loss"] = losses["gs_loss"] + losses["gs_ft_loss"]
                losses["loss"] = losses["gs_loss"]
            else:
                disp_list = list(outputs[("disp", i)] for i in range(4))
                self.generate_images_pred(inputs, outputs, disp_list, 4)
                losses["loss"] = self.compute_losses(inputs, outputs, disp_list, 4, True)

            

            return outputs, losses
        else:
            return outputs

    def predict_poses(self, inputs, num_pose_frames):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.
            pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        elif num_pose_frames == 1:
            pose_inputs = [inputs["color", -1, 0], inputs["color", 0, 0]]
            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
            axisangle, translation = self.models["pose"](pose_inputs)
            return transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=True)

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
                self.save_checkpoint(os.path.join(self.log_path, "models", model_save_name))

        del inputs, outputs, eval_measures

        self.set_train()

    def generate_images_pred(self, inputs, outputs, disps, num_scales, tag=""):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in range(num_scales):
            # disp_HiS = outputs["out_HiS"][("disp", scale)]
            disp = disps[scale]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # outputs[("disp", scale)] = disp
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[(f"{tag}depth", 0, scale)] = depth
            outputs[(f"{tag}disp_full_res", scale)] = disp
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                # 将源图像的像素坐标投影到目标方位并采样，相当于目标方位的图像投影到源图像方位
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[(f"{tag}color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border",
                    align_corners = True)
                
    def generate_features_pred(self, inputs, outputs, inv_K, K):
        """
        相邻帧重新预测高斯分布光栅化得到的特征
        ONLY 1/4 scale
        """
        nearby_gs_features = {}
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            # TODO 讨论统一使用增强的color数据是否合理？（考虑到FeatDepth中使用未数据增强数据）
            input_colors = list(inputs["color_aug", frame_id, i] for i in range(6))
            features = self.models["encoder"](input_colors[0])
            _ , gs_features = self.models["gs_leverage"](
                init_features = features,
                init_disps = list(outputs["init_disp", i] for i in self.opt.scales),
                colors = input_colors,
                inv_K = inv_K, K = K,
                T = None
            )
            nearby_gs_features[(frame_id, 0)] = gs_features[(0, 0)]
        return nearby_gs_features
    def robust_l1(self, pred, target):
        """L1 Loss
        """
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)
    
    def compute_perceptional_loss(self, s_features, t_features):
        """Compute GS-Features perceptional loss
        """
        total_loss = 0
        # 只计算1/4分辨率上的gs特征重投影损失
        for index in range(1):
            loss = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = s_features[(frame_id, index)]
                target = t_features[(0, index)]
                loss.append(self.robust_l1(pred, target).mean(1, True))
                if self.step % 500 == 0:
                    save_dir = os.path.join(os.path.dirname(__file__), f"models/{self.opt.model_name}/results/gs_features/")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    target_path = os.path.join(save_dir, f"scale{index}_frameid{frame_id}_target0.png")
                    pred_path = os.path.join(save_dir, f"scale{index}_frameid{frame_id}_pred{frame_id}.png")
                    self.save_feat_img(target[0], target_path)
                    self.save_feat_img(pred[0], pred_path)
            # TODO 考虑是否加入auto-masking?
            loss = torch.cat(loss, 1)
            min_loss, idxs = torch.min(loss, 1) # 考虑遮挡问题，这里使用最小损失
            total_loss += min_loss.mean() # 转换为标量
        return total_loss
    
    def save_feat_img(self, feature, path):
        # 计算通道均值
        mean_feature = torch.mean(feature, dim=0)  # H×W
    
        # 保存特征图多通道的均值
        mean_feature_np = mean_feature.cpu().detach().numpy()
        normalized = (mean_feature_np - np.min(mean_feature_np)) / (np.max(mean_feature_np) - np.min(mean_feature_np))
        heatmap = np.uint8(255 * normalized)
        
        # 应用颜色映射并保存
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(path, heatmap)
        # # 转换为网格图像（自动拼接多通道）
        # grid = vutils.make_grid(feat_norm.unsqueeze(1), nrow=8, padding=2, normalize=True)
        
        # # 转换为OpenCV格式并保存
        # grid_np = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        # grid_bgr = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)  # 确保颜色通道正确
        # cv2.imwrite(path, grid_bgr)
        
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

    def compute_losses(self, inputs, outputs, disps, num_scales, use_smooth_loss, tag=""):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        total_loss = 0

        train_save_path = os.path.join(os.path.dirname(__file__), "models/{}/results/".format(self.opt.model_name))
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)

        for scale in range(num_scales):
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = disps[scale]
            # 只有计算smooth_loss时用到
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            if self.step % 500 == 0:
                # TODO 保存变化尺度后的depthmap
                _disps = {}
                _disps[f"{tag}disp"] = outputs[(f"{tag}disp_full_res", scale)]
                # disps["init_disp"] = outputs[("init_disp", scale)]
                _depth = outputs[(f"{tag}depth", 0, scale)]
                self.save_img(_depth,_disps,target,scale,train_save_path)

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[(f"{tag}color", frame_id, scale)]
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

            loss += to_optimise.mean()
            
            if use_smooth_loss:
                # print("Using init smooth loss")
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)
                theta = 2 ** scale
                loss += self.opt.disparity_smoothness * smooth_loss / theta
                
            total_loss += loss
            # losses[f"{tag}loss/{scale}"] = loss

        total_loss /= num_scales
        # losses["loss"] = total_loss
        
        return total_loss
    
    
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
    
        # print_string = "epo {:>3} | bat {:>6} | ex/s: {:5.1f}" + \
        #     " | loss: {:.5f} gs_ft {:.5f} | te: {} | tl: {} | lr: {}"
        # print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"].cpu().data,losses["gs_ft_loss"],
        #     sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.state_dict()['param_groups'][0]['lr']))
        print_string = "epo {:>3} | bat {:>6} | ex/s: {:5.1f}" + \
            " | loss: {:.5f} | te: {} | tl: {} | lr: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"].cpu().data,
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.state_dict()['param_groups'][0]['lr']))
    def save_img(self, depth, disps, target, scale, train_save_path):
        model_name = self.opt.model_name
        for disp_type, disp in disps.items():
            # 保存视差图
            disp_np = disp[0].squeeze().cpu().detach().numpy()
            vmax = np.percentile(disp_np, 95)
            # 打印数据统计信息（调试用）
            # print(f"{disp_type}: Scale {scale}: min={disp_np.min()}, max={disp_np.max()}, mean={disp_np.mean()}")
            
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
        
    def save_checkpoint(self, save_path):
        """保存完整训练状态"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
 
            'encoder': self.models["encoder"].state_dict(),
            'init_decoder': self.models["init_decoder"].state_dict(),
            'gs_leverage': self.models["gs_leverage"].state_dict() if self.opt.use_gs else None,
            'gs_decoder': self.models["gs_decoder"].state_dict() if self.opt.use_gs else None,
            'pose_encoder': self.models["pose_encoder"].state_dict(),
            'pose': self.models["pose"].state_dict(),

            'optimizer_state': self.model_optimizer.state_dict(),
            'scheduler_state': self.model_lr_scheduler.state_dict(),
            'best_metrics': {
                'lower': self.best_eval_measures_lower_better,
                'higher': self.best_eval_measures_higher_better,
                'epochs': self.best_eval_epochs,
                'batch_idxs': self.best_eval_batch_idxs
            },
            'rng_states': {  # 随机数状态
                'cpu': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }
        torch.save(checkpoint, save_path)
        
    def load_checkpoint_resume(self, path):
        """Load model(s) from disk for retrain
        """

        assert not os.path.isdir(path), \
            "--resume_checkpoint_path should be a model file, but {} is a directory".format(path)
        print("loading model to resume from path {}".format(path))

        if os.path.isfile(path):
            checkpoint = torch.load(path)
            # 恢复模型参数
            for name, model in self.models.items():
                print("Loading {} weights...".format(name))
                assert name in checkpoint, \
                    f"in loaded resume checkpoint, we not found the model named {name}! "
                model.load_state_dict(checkpoint[name])
                    
            # 恢复优化器和调度器
            print("Loading Adam and Scheduler weights")
            self.model_optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.model_lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            # 关键设置：确保调度器从正确epoch开始
            print("Resume Scheduler from epoch {}".format(checkpoint['epoch']))
            self.model_lr_scheduler.last_epoch = checkpoint['epoch']  
            
            # 恢复训练状态
            print(f"Continue Training from epoch {checkpoint['epoch']} step {checkpoint['step']}")
            self.epoch = checkpoint['epoch']
            self.step = checkpoint['step']
            
            # 恢复随机数状态
            print("Resume pytorch random status")
            torch.set_rng_state(checkpoint['rng_states']['cpu'])
            if torch.cuda.is_available() and checkpoint['rng_states']['cuda']:
                torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])
            # loading best values
            self.best_eval_epochs = checkpoint['best_metrics']['epochs']
            self.best_eval_batch_idxs = checkpoint['best_metrics']['batch_idxs']
            self.best_eval_measures_higher_better = checkpoint['best_metrics']['higher'].cpu()
            self.best_eval_measures_lower_better = checkpoint['best_metrics']['lower'].cpu()
        else:
            print("== No checkpoint found at '{}'".format(path))
        del checkpoint
        
    def load_pretrained_model(self, path, models_to_load, frozen):
        """Load pretrained model(s) from disk
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
                # 只加载现在版本代码中模型存在的组件
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
            print("== No checkpoint found at '{}'".format(path))

