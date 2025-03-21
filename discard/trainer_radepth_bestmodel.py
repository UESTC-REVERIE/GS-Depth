# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
from tqdm import tqdm


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.hrnet18(self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder_MSF_GS_FiT_v2(
            self.models["encoder"].num_ch_enc, 
            self.opt.scales,
            use_gs=self.opt.use_gs, 
            gs_scale=self.opt.gs_scale, 
            min_depth=self.opt.min_depth, 
            max_depth=self.opt.max_depth, 
            height=self.opt.height, 
            width=self.opt.width)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
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

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        
        # self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.model_optimizer, [15,20], 0.3)

        # if self.opt.load_weights_folder is not None:
        #     self.load_model()
        if self.opt.load_weights_folder is not None:
            # self.load_model()
            self.load_model_checkpoints()
            self.model_optimizer.param_groups[0]['lr'] = self.opt.learning_rate
            self.model_optimizer.param_groups[0]['initial_lr'] = self.opt.learning_rate

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset_RADepth,
                         "kitti_odom": datasets.KITTIOdomDataset_RADepth}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=True, img_ext=img_ext, gs_scale=self.opt.gs_scale)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_val=True, img_ext=img_ext, gs_scale=self.opt.gs_scale)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.detail_guide = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.detail_guide[scale] = DetailGuide(self.opt.batch_size, h, w)
            self.detail_guide[scale].to(self.device)

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
        for self.epoch in range(self.opt.num_epochs):
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

            outputs, losses = self.process_batch('train', inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

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
            if mode == 'val':
                inv_K_MiS = []
                K_MiS = []
                if self.opt.use_gs:
                    if self.opt.gs_scale != 0:
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", self.opt.gs_scale)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", self.opt.gs_scale)].to(self.device))
                    elif self.opt.gs_scale == 0:
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 0)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 1)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 2)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 3)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 4)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 0)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 1)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 2)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 3)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 4)].to(self.device))

                features_MiS = self.models["encoder"](inputs["color_MiS_aug", 0, 0])
                outputs["out_MiS"] = self.models["depth"](features_MiS, inv_K_MiS, K_MiS)
            else:    
                inv_K_HiS = []
                K_HiS = []
                inv_K_MiS = []
                K_MiS = []
                inv_K_LoS = []
                K_LoS = []
                if self.opt.use_gs:
                    if self.opt.gs_scale != 0:
                        inv_K_HiS.append(inputs[("inv_K_gs_HiS", self.opt.gs_scale)].to(self.device))
                        K_HiS.append(inputs[("K_gs_HiS", self.opt.gs_scale)].to(self.device))

                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", self.opt.gs_scale)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", self.opt.gs_scale)].to(self.device))

                        inv_K_LoS.append(inputs[("inv_K_gs_LoS", self.opt.gs_scale)].to(self.device))
                        K_LoS.append(inputs[("K_gs_LoS", self.opt.gs_scale)].to(self.device))
                    elif self.opt.gs_scale == 0:
                        inv_K_HiS.append(inputs[("inv_K_gs_HiS", 1)].to(self.device))
                        inv_K_HiS.append(inputs[("inv_K_gs_HiS", 2)].to(self.device))
                        inv_K_HiS.append(inputs[("inv_K_gs_HiS", 3)].to(self.device))
                        inv_K_HiS.append(inputs[("inv_K_gs_HiS", 4)].to(self.device))
                        inv_K_HiS.append(inputs[("inv_K_gs_HiS", 5)].to(self.device))

                        K_HiS.append(inputs[("K_gs_HiS", 1)].to(self.device))
                        K_HiS.append(inputs[("K_gs_HiS", 2)].to(self.device))
                        K_HiS.append(inputs[("K_gs_HiS", 3)].to(self.device))
                        K_HiS.append(inputs[("K_gs_HiS", 4)].to(self.device))
                        K_HiS.append(inputs[("K_gs_HiS", 5)].to(self.device))

                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 1)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 2)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 3)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 4)].to(self.device))
                        inv_K_MiS.append(inputs[("inv_K_gs_MiS", 5)].to(self.device))

                        K_MiS.append(inputs[("K_gs_MiS", 1)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 2)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 3)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 4)].to(self.device))
                        K_MiS.append(inputs[("K_gs_MiS", 5)].to(self.device))

                        inv_K_LoS.append(inputs[("inv_K_gs_LoS", 1)].to(self.device))
                        inv_K_LoS.append(inputs[("inv_K_gs_LoS", 2)].to(self.device))
                        inv_K_LoS.append(inputs[("inv_K_gs_LoS", 3)].to(self.device))
                        inv_K_LoS.append(inputs[("inv_K_gs_LoS", 4)].to(self.device))
                        inv_K_LoS.append(inputs[("inv_K_gs_LoS", 5)].to(self.device))

                        K_LoS.append(inputs[("K_gs_LoS", 1)].to(self.device))
                        K_LoS.append(inputs[("K_gs_LoS", 2)].to(self.device))
                        K_LoS.append(inputs[("K_gs_LoS", 3)].to(self.device))
                        K_LoS.append(inputs[("K_gs_LoS", 4)].to(self.device))
                        K_LoS.append(inputs[("K_gs_LoS", 5)].to(self.device))

                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                features_HiS = self.models["encoder"](inputs["color_HiS", 0, 0])
                outputs["out_HiS"] = self.models["depth"](features_HiS, inv_K_HiS, K_HiS)
                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                features_MiS = self.models["encoder"](inputs["color_MiS_aug", 0, 0])
                outputs["out_MiS"] = self.models["depth"](features_MiS, inv_K_MiS, K_MiS)
                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                features_LoS = self.models["encoder"](inputs["color_LoS", 0, 0])
                outputs["out_LoS"] = self.models["depth"](features_LoS, inv_K_LoS, K_LoS)

        if mode == 'train':
            if self.opt.predictive_mask:
                outputs["predictive_mask"] = self.models["predictive_mask"](features)

            if self.use_pose_net:
                outputs.update(self.predict_poses(inputs, features_MiS))

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
                pose_feats = {f_i: inputs["color_MiS", f_i, 0] for f_i in self.opt.frame_ids}

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
                                'depth': self.models["depth"].state_dict(),
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
            disp_HiS = outputs["out_HiS"][("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # disp = F.interpolate(
                #     disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            _, depth_HiS = disp_to_depth(disp_HiS, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth_HiS", 0, scale)] = depth_HiS


            disp_MiS = outputs["out_MiS"][("disp", scale)]
            _, depth_MiS = disp_to_depth(disp_MiS, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth_MiS", 0, scale)] = depth_MiS


            disp_LoS = outputs["out_LoS"][("disp", scale)]
            _, depth_LoS = disp_to_depth(disp_LoS, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth_LoS", 0, scale)] = depth_LoS

            if self.opt.use_gs and self.opt.gs_scale != 0:
                disp_gs_HiS = outputs["out_HiS"][("disp_init", self.opt.gs_scale)]
                disp_gs_HiS = F.interpolate(
                        disp_gs_HiS, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth_gs_HiS = disp_to_depth(disp_gs_HiS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_gs_HiS", 0, scale)] = depth_gs_HiS

                disp_gs_MiS = outputs["out_MiS"][("disp_init", self.opt.gs_scale)]
                disp_gs_MiS = F.interpolate(
                        disp_gs_MiS, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth_gs_MiS = disp_to_depth(disp_gs_MiS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_gs_MiS", 0, scale)] = depth_gs_MiS

                disp_gs_LoS = outputs["out_LoS"][("disp_init", self.opt.gs_scale)]
                disp_gs_LoS = F.interpolate(
                        disp_gs_LoS, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth_gs_LoS = disp_to_depth(disp_gs_LoS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_gs_LoS", 0, scale)] = depth_gs_LoS

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

                cam_points_HiS = self.backproject_depth[source_scale](
                    depth_HiS, inputs[("inv_K_HiS", source_scale)], inputs[("dxy_HiS")])
                pix_coords_HiS = self.project_3d[source_scale](
                    cam_points_HiS, inputs[("K_HiS", source_scale)], T, inputs[("dxy_HiS")])
                outputs[("sample_HiS", frame_id, scale)] = pix_coords_HiS
                outputs[("color_HiS", frame_id, scale)] = F.grid_sample(
                    inputs[("color_HiS", frame_id, source_scale)],
                    outputs[("sample_HiS", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)

                cam_points_MiS = self.backproject_depth[source_scale](
                    depth_MiS, inputs[("inv_K_MiS", source_scale)], inputs[("dxy_MiS")])
                pix_coords_MiS = self.project_3d[source_scale](
                    cam_points_MiS, inputs[("K_MiS", source_scale)], T, inputs[("dxy_MiS")])
                outputs[("sample_MiS", frame_id, scale)] = pix_coords_MiS
                outputs[("color_MiS", frame_id, scale)] = F.grid_sample(
                    inputs[("color_MiS", frame_id, source_scale)],
                    outputs[("sample_MiS", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)

                cam_points_LoS = self.backproject_depth[source_scale](
                    depth_LoS, inputs[("inv_K_LoS", source_scale)], inputs[("dxy_LoS")])
                pix_coords_LoS = self.project_3d[source_scale](
                    cam_points_LoS, inputs[("K_LoS", source_scale)], T, inputs[("dxy_LoS")])
                outputs[("sample_LoS", frame_id, scale)] = pix_coords_LoS
                outputs[("color_LoS", frame_id, scale)] = F.grid_sample(
                    inputs[("color_LoS", frame_id, source_scale)],
                    outputs[("sample_LoS", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)
                
                if self.opt.use_gs and self.opt.gs_scale != 0:
                    cam_points_gs_HiS = self.backproject_depth[source_scale](
                        depth_gs_HiS, inputs[("inv_K_HiS", source_scale)], inputs[("dxy_HiS")])
                    pix_coords_gs_HiS = self.project_3d[source_scale](
                        cam_points_gs_HiS, inputs[("K_HiS", source_scale)], T, inputs[("dxy_HiS")])
                    outputs[("sample_gs_HiS", frame_id, scale)] = pix_coords_gs_HiS
                    outputs[("color_gs_HiS", frame_id, scale)] = F.grid_sample(
                        inputs[("color_HiS", frame_id, source_scale)],
                        outputs[("sample_gs_HiS", frame_id, scale)],
                        padding_mode="border",
                        align_corners = True)
                    
                    cam_points_gs_MiS = self.backproject_depth[source_scale](
                        depth_gs_MiS, inputs[("inv_K_MiS", source_scale)], inputs[("dxy_MiS")])
                    pix_coords_gs_MiS = self.project_3d[source_scale](
                        cam_points_gs_MiS, inputs[("K_MiS", source_scale)], T, inputs[("dxy_MiS")])
                    outputs[("sample_gs_MiS", frame_id, scale)] = pix_coords_gs_MiS
                    outputs[("color_gs_MiS", frame_id, scale)] = F.grid_sample(
                        inputs[("color_MiS", frame_id, source_scale)],
                        outputs[("sample_gs_MiS", frame_id, scale)],
                        padding_mode="border",
                        align_corners = True)
                    
                    cam_points_gs_LoS = self.backproject_depth[source_scale](
                        depth_gs_LoS, inputs[("inv_K_LoS", source_scale)], inputs[("dxy_LoS")])
                    pix_coords_gs_LoS = self.project_3d[source_scale](
                        cam_points_gs_LoS, inputs[("K_LoS", source_scale)], T, inputs[("dxy_LoS")])
                    outputs[("sample_gs_LoS", frame_id, scale)] = pix_coords_gs_LoS
                    outputs[("color_gs_LoS", frame_id, scale)] = F.grid_sample(
                        inputs[("color_LoS", frame_id, source_scale)],
                        outputs[("sample_gs_LoS", frame_id, scale)],
                        padding_mode="border",
                        align_corners = True)
                    
            
                # if not self.opt.disable_automasking:
                #     outputs[("color_identity", frame_id, scale)] = \
                #         inputs[("color_MiS", frame_id, source_scale)]

        if self.opt.use_gs and self.opt.gs_scale == 0:
                for scale in range(1, 6):
                    
                    if self.opt.v1_multiscale:
                        source_scale = scale
                    else:
                        source_scale = 0

                    disp_gs_HiS = outputs["out_HiS"][("disp_init", 2**scale)]
                    disp_gs_HiS = F.interpolate(
                            disp_gs_HiS, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    outputs[("disp_gs_HiS", 0, scale)] = disp_gs_HiS
                    _, depth_gs_HiS = disp_to_depth(disp_gs_HiS, self.opt.min_depth, self.opt.max_depth)
                    outputs[("depth_gs_HiS", 0, scale)] = depth_gs_HiS

                    disp_gs_MiS = outputs["out_MiS"][("disp_init", 2**scale)]
                    disp_gs_MiS = F.interpolate(
                            disp_gs_MiS, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    outputs[("disp_gs_MiS", 0, scale)] = disp_gs_MiS
                    _, depth_gs_MiS = disp_to_depth(disp_gs_MiS, self.opt.min_depth, self.opt.max_depth)
                    outputs[("depth_gs_MiS", 0, scale)] = depth_gs_MiS

                    disp_gs_LoS = outputs["out_LoS"][("disp_init", 2**scale)]
                    disp_gs_LoS = F.interpolate(
                            disp_gs_LoS, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    outputs[("disp_gs_LoS", 0, scale)] = disp_gs_LoS
                    _, depth_gs_LoS = disp_to_depth(disp_gs_LoS, self.opt.min_depth, self.opt.max_depth)
                    outputs[("depth_gs_LoS", 0, scale)] = depth_gs_LoS

                    # disp_LoS = outputs["out_LoS"][("disp", scale)]
                    # _, depth_LoS = disp_to_depth(disp_LoS, self.opt.min_depth, self.opt.max_depth)
                    # outputs[("depth_LoS", 0, scale)] = depth_LoS

                    for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                        if frame_id == "s":
                            T = inputs["stereo_T"]
                        else:
                            T = outputs[("cam_T_cam", 0, frame_id)]

                        # from the authors of https://arxiv.org/abs/1712.00175
                        if self.opt.pose_model_type == "posecnn":

                            axisangle = outputs[("axisangle", 0, frame_id)]
                            translation = outputs[("translation", 0, frame_id)]

                            inv_depth = 1 / depth_gs_MiS
                            mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                            T = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                        
                        cam_points_gs_HiS = self.backproject_depth[source_scale](
                        depth_gs_HiS, inputs[("inv_K_HiS", source_scale)], inputs[("dxy_HiS")])
                        pix_coords_gs_HiS = self.project_3d[source_scale](
                            cam_points_gs_HiS, inputs[("K_HiS", source_scale)], T, inputs[("dxy_HiS")])
                        outputs[("sample_gs_HiS", frame_id, scale)] = pix_coords_gs_HiS
                        outputs[("color_gs_HiS", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            outputs[("sample_gs_HiS", frame_id, scale)],
                            padding_mode="border",
                            align_corners = True)
                        
                        cam_points_gs_MiS = self.backproject_depth[source_scale](
                            depth_gs_MiS, inputs[("inv_K_MiS", source_scale)], inputs[("dxy_MiS")])
                        pix_coords_gs_MiS = self.project_3d[source_scale](
                            cam_points_gs_MiS, inputs[("K_MiS", source_scale)], T, inputs[("dxy_MiS")])
                        outputs[("sample_gs_MiS", frame_id, scale)] = pix_coords_gs_MiS
                        outputs[("color_gs_MiS", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            outputs[("sample_gs_MiS", frame_id, scale)],
                            padding_mode="border",
                            align_corners = True)
                        
                        cam_points_gs_LoS = self.backproject_depth[source_scale](
                            depth_gs_LoS, inputs[("inv_K_LoS", source_scale)], inputs[("dxy_LoS")])
                        pix_coords_gs_LoS = self.project_3d[source_scale](
                            cam_points_gs_LoS, inputs[("K_LoS", source_scale)], T, inputs[("dxy_LoS")])
                        outputs[("sample_gs_LoS", frame_id, scale)] = pix_coords_gs_LoS
                        outputs[("color_gs_LoS", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            outputs[("sample_gs_LoS", frame_id, scale)],
                            padding_mode="border",
                            align_corners = True)

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

    def compute_reprojection_loss_disp(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        # l1_loss = torch.log(1.0 + torch.abs(target - pred))
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.5 * ssim_loss + 0.5 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        loss_HiS = 0
        loss_MiS = 0
        loss_LoS = 0
        loss_consis_HiS = 0
        loss_consis_LoS = 0
        loss_gs_HiS = 0
        loss_gs_MiS = 0
        loss_gs_LoS = 0

        train_save_path = os.path.join(os.path.dirname(__file__), "models/{}/results/".format(self.opt.model_name))
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)

        for scale in self.opt.scales:
            reprojection_losses_HiS = []
            reprojection_losses_MiS = []
            reprojection_losses_LoS = []

            reprojection_losses_gs_HiS = []
            reprojection_losses_gs_MiS = []
            reprojection_losses_gs_LoS = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp_HiS = outputs["out_HiS"][("disp", scale)]
            color_HiS = inputs[("color_HiS", 0, source_scale)]
            target_HiS = inputs[("color_HiS", 0, source_scale)]

            disp_MiS = outputs["out_MiS"][("disp", scale)]
            color_MiS = inputs[("color_MiS", 0, source_scale)]
            target_MiS = inputs[("color_MiS", 0, source_scale)]

            disp_LoS = outputs["out_LoS"][("disp", scale)]
            color_LoS = inputs[("color_LoS", 0, source_scale)]
            target_LoS = inputs[("color_LoS", 0, source_scale)]

            if self.opt.use_gs and self.opt.gs_scale != 0:
                disp_gs_HiS = outputs["out_HiS"][("disp_init", self.opt.gs_scale)]
                disp_gs_MiS = outputs["out_MiS"][("disp_init", self.opt.gs_scale)]
                disp_gs_LoS = outputs["out_LoS"][("disp_init", self.opt.gs_scale)]

            if self.step % 500 == 0:
                dn_to_disp_resized_np = disp_MiS[0].squeeze().cpu().detach().numpy()
                vmax = np.percentile(dn_to_disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=dn_to_disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma') #magma
                colormapped_im = (mapper.to_rgba(dn_to_disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)
                im.save(os.path.join(train_save_path, "{}_disp.png".format(self.opt.model_name)))
                
                target_ = target_MiS[0].cpu().detach().numpy() * 255
                target_ = np.transpose(target_, (1, 2, 0)).astype(np.uint8)
                target_ = Image.fromarray(target_)
                target_.save(os.path.join(train_save_path, "{}_image.png".format(self.opt.model_name)))

                
                if self.opt.use_gs and self.opt.gs_scale != 0:
                    dn_to_disp_resized_np = disp_gs_MiS[0].squeeze().cpu().detach().numpy()
                    vmax = np.percentile(dn_to_disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=dn_to_disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma') #magma
                    colormapped_im = (mapper.to_rgba(dn_to_disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    im = pil.fromarray(colormapped_im)
                    im.save(os.path.join(train_save_path, "{}_{}_disp_init.png".format(self.opt.model_name, self.opt.gs_scale)))

            for frame_id in self.opt.frame_ids[1:]:
                pred_HiS = outputs[("color_HiS", frame_id, scale)]
                reprojection_losses_HiS.append(self.compute_reprojection_loss(pred_HiS, target_HiS))
                pred_MiS = outputs[("color_MiS", frame_id, scale)]
                reprojection_losses_MiS.append(self.compute_reprojection_loss(pred_MiS, target_MiS))
                pred_LoS = outputs[("color_LoS", frame_id, scale)]
                reprojection_losses_LoS.append(self.compute_reprojection_loss(pred_LoS, target_LoS))

                if self.opt.use_gs and self.opt.gs_scale != 0:
                    pred_gs_HiS = outputs[("color_gs_HiS", frame_id, scale)]
                    reprojection_losses_gs_HiS.append(self.compute_reprojection_loss(pred_gs_HiS, target_HiS))
                    pred_gs_MiS = outputs[("color_gs_MiS", frame_id, scale)]
                    reprojection_losses_gs_MiS.append(self.compute_reprojection_loss(pred_gs_MiS, target_MiS))
                    pred_gs_LoS = outputs[("color_gs_LoS", frame_id, scale)]
                    reprojection_losses_gs_LoS.append(self.compute_reprojection_loss(pred_gs_LoS, target_LoS))

            reprojection_losses_HiS = torch.cat(reprojection_losses_HiS, 1)
            reprojection_losses_MiS = torch.cat(reprojection_losses_MiS, 1)
            reprojection_losses_LoS = torch.cat(reprojection_losses_LoS, 1)
            if self.opt.use_gs and self.opt.gs_scale != 0:
                reprojection_losses_gs_HiS = torch.cat(reprojection_losses_gs_HiS, 1)
                reprojection_losses_gs_MiS = torch.cat(reprojection_losses_gs_MiS, 1)
                reprojection_losses_gs_LoS = torch.cat(reprojection_losses_gs_LoS, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses_HiS = []
                identity_reprojection_losses_MiS = []
                identity_reprojection_losses_LoS = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred_HiS = inputs[("color_HiS", frame_id, source_scale)]
                    identity_reprojection_losses_HiS.append(
                        self.compute_reprojection_loss(pred_HiS, target_HiS))

                    pred_MiS = inputs[("color_MiS", frame_id, source_scale)]
                    identity_reprojection_losses_MiS.append(
                        self.compute_reprojection_loss(pred_MiS, target_MiS))

                    pred_LoS = inputs[("color_LoS", frame_id, source_scale)]
                    identity_reprojection_losses_LoS.append(
                        self.compute_reprojection_loss(pred_LoS, target_LoS))

                identity_reprojection_losses_HiS = torch.cat(identity_reprojection_losses_HiS, 1)
                identity_reprojection_losses_MiS = torch.cat(identity_reprojection_losses_MiS, 1)
                identity_reprojection_losses_LoS = torch.cat(identity_reprojection_losses_LoS, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss_HiS = identity_reprojection_losses_HiS.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss_HiS = identity_reprojection_losses_HiS
                    identity_reprojection_loss_MiS = identity_reprojection_losses_MiS
                    identity_reprojection_loss_LoS = identity_reprojection_losses_LoS

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses_HiS *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss_HiS = reprojection_losses_HiS.mean(1, keepdim=True)
                if self.opt.use_gs and self.opt.gs_scale != 0:
                    reprojection_loss_gs_HiS = reprojection_losses_gs_HiS.mean(1, keepdim=True)
                    reprojection_loss_gs_MiS = reprojection_losses_gs_MiS.mean(1, keepdim=True)
                    reprojection_loss_gs_LoS = reprojection_losses_gs_LoS.mean(1, keepdim=True)
            else:
                reprojection_loss_HiS = reprojection_losses_HiS
                reprojection_loss_MiS = reprojection_losses_MiS
                reprojection_loss_LoS = reprojection_losses_LoS
                if self.opt.use_gs and self.opt.gs_scale != 0:
                    reprojection_loss_gs_HiS = reprojection_losses_gs_HiS
                    reprojection_loss_gs_MiS = reprojection_losses_gs_MiS
                    reprojection_loss_gs_LoS = reprojection_losses_gs_LoS

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss_HiS += torch.randn(
                    identity_reprojection_loss_HiS.shape).cuda() * 0.00001
                identity_reprojection_loss_MiS += torch.randn(
                    identity_reprojection_loss_MiS.shape).cuda() * 0.00001
                identity_reprojection_loss_LoS += torch.randn(
                    identity_reprojection_loss_LoS.shape).cuda() * 0.00001

                combined_HiS = torch.cat((identity_reprojection_loss_HiS, reprojection_loss_HiS), dim=1)
                combined_MiS = torch.cat((identity_reprojection_loss_MiS, reprojection_loss_MiS), dim=1)
                combined_LoS = torch.cat((identity_reprojection_loss_LoS, reprojection_loss_LoS), dim=1)

                if self.opt.use_gs and self.opt.gs_scale != 0:
                    combined_gs_HiS = torch.cat((identity_reprojection_loss_HiS, reprojection_loss_gs_HiS), dim=1)
                    combined_gs_MiS = torch.cat((identity_reprojection_loss_MiS, reprojection_loss_gs_MiS), dim=1)
                    combined_gs_LoS = torch.cat((identity_reprojection_loss_LoS, reprojection_loss_gs_LoS), dim=1)

            else:
                combined_HiS = reprojection_loss_HiS

            if combined_HiS.shape[1] == 1:
                to_optimise_HiS = combined_HiS
                if self.opt.use_gs and self.opt.gs_scale != 0:
                    to_optimise_gs_HiS  = combined_gs_HiS 
                    to_optimise_gs_MiS = combined_gs_MiS
                    to_optimise_gs_LoS = combined_gs_LoS
            else:
                to_optimise_HiS, idxs_HiS = torch.min(combined_HiS, dim=1)
                to_optimise_MiS, idxs_MiS = torch.min(combined_MiS, dim=1)
                to_optimise_LoS, idxs_LoS = torch.min(combined_LoS, dim=1)
                if self.opt.use_gs and self.opt.gs_scale != 0:
                    to_optimise_gs_HiS, _ = torch.min(combined_gs_HiS, dim=1)
                    to_optimise_gs_MiS, _ = torch.min(combined_gs_MiS, dim=1)
                    to_optimise_gs_LoS, _ = torch.min(combined_gs_LoS, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs_HiS > identity_reprojection_loss_HiS.shape[1] - 1).float()

            mask4consis = torch.ones(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
            mask4consis[idxs_MiS == 0] = 0 #[12,192,640]
            mask4consis[idxs_MiS == 1] = 0
            mask4consis = mask4consis.unsqueeze(1).detach() #[12,1,192,640]

            loss_HiS += to_optimise_HiS.mean()
            mean_disp_HiS = disp_HiS.mean(2, True).mean(3, True)
            norm_disp_HiS = disp_HiS / (mean_disp_HiS + 1e-7)
            smooth_loss_HiS = get_smooth_loss(norm_disp_HiS, color_HiS)
            loss_HiS += self.opt.disparity_smoothness * smooth_loss_HiS

            loss_MiS += to_optimise_MiS.mean()
            mean_disp_MiS = disp_MiS.mean(2, True).mean(3, True)
            norm_disp_MiS = disp_MiS / (mean_disp_MiS + 1e-7)
            smooth_loss_MiS = get_smooth_loss(norm_disp_MiS, color_MiS)
            loss_MiS += self.opt.disparity_smoothness * smooth_loss_MiS

            loss_LoS += to_optimise_LoS.mean()
            mean_disp_LoS = disp_LoS.mean(2, True).mean(3, True)
            norm_disp_LoS = disp_LoS / (mean_disp_LoS + 1e-7)
            smooth_loss_LoS = get_smooth_loss(norm_disp_LoS, color_LoS)
            loss_LoS += self.opt.disparity_smoothness * smooth_loss_LoS

            if self.opt.use_gs and self.opt.gs_scale != 0:
                loss_gs_HiS += to_optimise_gs_HiS.mean()
                loss_gs_MiS += to_optimise_gs_MiS.mean()
                loss_gs_MiS += to_optimise_gs_MiS.mean()

            disp_diff_batch = 0
            for i in range(self.opt.batch_size):
                x_start = round( self.opt.width * int(inputs[("dxy_HiS")][i,0]) * 1.0 / int(inputs[("resize_HiS")][i,0]) )
                y_start = round( self.opt.height * int(inputs[("dxy_HiS")][i,1]) * 1.0 / int(inputs[("resize_HiS")][i,1]) )
                width_union = round( self.opt.width * self.opt.width * 1.0 / int(inputs[("resize_HiS")][i,0]) )
                height_union = round( self.opt.height * self.opt.height * 1.0 / int(inputs[("resize_HiS")][i,1]) )
                disp_union_HiS = F.interpolate(disp_HiS[i].unsqueeze(0), [height_union, width_union], mode="bilinear", align_corners=False)
                disp_union_MiS = disp_MiS[i,:,y_start:y_start+height_union,x_start:x_start+width_union].unsqueeze(0)
                mask_disp = mask4consis[i,:,y_start:y_start+height_union,x_start:x_start+width_union].unsqueeze(0)   #[12,1,h,w]
                disp_diff = self.compute_reprojection_loss(disp_union_MiS, disp_union_HiS)             #[12,1,h,w]
                disp_diff_mask = disp_diff * mask_disp
                disp_diff_batch += (disp_diff_mask.sum() / (mask_disp.sum() + 1e-7))
            loss_consis_HiS += disp_diff_batch / self.opt.batch_size

            disp_diff_batch_LoS = 0
            for i in range(self.opt.batch_size):
                width_un_LoS = int(inputs[("resize_LoS")][i,0])
                height_un_LoS = int(inputs[("resize_LoS")][i,1])
                disp_un_LoS = F.interpolate(disp_LoS[i,:,0:height_un_LoS,0:width_un_LoS].unsqueeze(0), [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                disp_un_MiS = disp_MiS[i].unsqueeze(0)
                disp_diff_LoS = self.compute_reprojection_loss(disp_un_MiS, disp_un_LoS)
                mask_disp_LoS = mask4consis.unsqueeze(0)
                disp_diff_LoS_mask = disp_diff_LoS * mask_disp_LoS
                disp_diff_batch_LoS += (disp_diff_LoS_mask.sum() / (mask_disp_LoS.sum() + 1e-7))
            loss_consis_LoS += disp_diff_batch_LoS / self.opt.batch_size

            if self.opt.use_gs and self.opt.gs_scale != 0:
                loss_gs_HiS += to_optimise_gs_HiS.mean()
                loss_gs_MiS += to_optimise_gs_MiS.mean()
                loss_gs_LoS += to_optimise_gs_LoS.mean()

            losses["loss/{}".format(scale)] = loss_HiS
        
        if self.opt.use_gs and self.opt.gs_scale == 0:
            for scale in range(1, 6):
                reprojection_losses_gs_HiS = []
                reprojection_losses_gs_MiS = []
                reprojection_losses_gs_LoS = []

                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    source_scale = 0
                disp_gs_HiS = outputs[("disp_gs_HiS", 0, scale)]
                disp_gs_MiS = outputs[("disp_gs_MiS", 0, scale)]
                disp_gs_LoS = outputs[("disp_gs_LoS", 0, scale)]
                color_HiS = inputs[("color_HiS", 0, source_scale)]
                target_HiS = inputs[("color_HiS", 0, source_scale)]

                color_MiS = inputs[("color_MiS", 0, source_scale)]
                target_MiS = inputs[("color_MiS", 0, source_scale)]

                color_LoS = inputs[("color_LoS", 0, source_scale)]
                target_LoS = inputs[("color_LoS", 0, source_scale)]

                if self.step % 500 == 0:
                    dn_to_disp_resized_np = disp_gs_MiS[0].squeeze().cpu().detach().numpy()
                    vmax = np.percentile(dn_to_disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=dn_to_disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma') #magma
                    colormapped_im = (mapper.to_rgba(dn_to_disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    im = pil.fromarray(colormapped_im)
                    im.save(os.path.join(train_save_path, "{}_{}_init_disp.png".format(self.opt.model_name, scale)))

                for frame_id in self.opt.frame_ids[1:]:
                    pred_gs_HiS = outputs[("color_gs_HiS", frame_id, scale)]
                    reprojection_losses_gs_HiS.append(self.compute_reprojection_loss(pred_gs_HiS, target_HiS))
                    pred_gs_MiS = outputs[("color_gs_MiS", frame_id, scale)]
                    reprojection_losses_gs_MiS.append(self.compute_reprojection_loss(pred_gs_MiS, target_HiS))
                    pred_gs_LoS = outputs[("color_gs_LoS", frame_id, scale)]
                    reprojection_losses_gs_LoS.append(self.compute_reprojection_loss(pred_gs_LoS, target_HiS))

                reprojection_losses_gs_HiS = torch.cat(reprojection_losses_gs_HiS, 1)
                reprojection_losses_gs_MiS = torch.cat(reprojection_losses_gs_MiS, 1)
                reprojection_losses_gs_LoS = torch.cat(reprojection_losses_gs_LoS, 1)

                if self.opt.avg_reprojection:
                    reprojection_loss_gs_HiS = reprojection_losses_gs_HiS.mean(1, keepdim=True)
                    reprojection_loss_gs_MiS = reprojection_losses_gs_MiS.mean(1, keepdim=True)
                    reprojection_loss_gs_LoS = reprojection_losses_gs_LoS.mean(1, keepdim=True)
                else:
                    reprojection_loss_gs_HiS = reprojection_losses_gs_HiS
                    reprojection_loss_gs_MiS = reprojection_losses_gs_MiS
                    reprojection_loss_gs_LoS = reprojection_losses_gs_LoS

                if not self.opt.disable_automasking:
                    combined_gs_HiS = torch.cat((identity_reprojection_loss_HiS, reprojection_loss_gs_HiS), dim=1)
                    combined_gs_MiS = torch.cat((identity_reprojection_loss_MiS, reprojection_loss_gs_MiS), dim=1)
                    combined_gs_LoS = torch.cat((identity_reprojection_loss_LoS, reprojection_loss_gs_LoS), dim=1)
                else:
                    combined_gs_HiS = reprojection_loss_gs_HiS
                    combined_gs_MiS = reprojection_loss_gs_MiS
                    combined_gs_LoS = reprojection_loss_gs_LoS

                if combined_gs_MiS.shape[1] == 1:
                    to_optimise_gs_HiS = combined_gs_HiS
                    to_optimise_gs_MiS = combined_gs_MiS
                    to_optimise_gs_LoS = combined_gs_LoS
                else:
                    to_optimise_gs_HiS, _ = torch.min(combined_gs_HiS, dim=1)
                    to_optimise_gs_MiS, _ = torch.min(combined_gs_MiS, dim=1)
                    to_optimise_gs_LoS, _ = torch.min(combined_gs_LoS, dim=1)

                loss_gs_HiS += to_optimise_gs_HiS.mean()
                loss_gs_MiS += to_optimise_gs_MiS.mean()
                loss_gs_LoS += to_optimise_gs_LoS.mean()
                # if self.opt.use_init_smoothLoss:
                #     mean_disp_gs = disp_gs.mean(2, True).mean(3, True)
                #     norm_disp_gs = disp_gs / (mean_disp_gs + 1e-7)
                #     smooth_loss_gs = get_smooth_loss(norm_disp_gs, color)
                #     loss_gs += self.opt.disparity_smoothness * smooth_loss_gs


        loss_HiS /= self.num_scales
        loss_MiS /= self.num_scales
        loss_LoS /= self.num_scales
        loss_consis_HiS /= self.num_scales
        loss_consis_LoS /= self.num_scales
        total_loss = (loss_HiS + loss_MiS + loss_LoS) + 1.0*(loss_consis_HiS + loss_consis_LoS)

        if self.opt.use_gs and self.opt.gs_scale == 0:
            loss_gs_HiS /= 5
            loss_gs_MiS /= 5
            loss_gs_LoS /= 5
            loss_gs = self.opt.loss_gs_weight * (loss_gs_HiS + loss_gs_MiS + loss_gs_LoS)
            total_loss += loss_gs
            losses["loss_gs"] = loss_gs

        elif self.opt.use_gs:
            loss_gs = self.opt.loss_gs_weight * (loss_gs_HiS + loss_gs_MiS + loss_gs_LoS)
            total_loss += loss_gs
            losses["loss_gs"] = loss_gs

        losses["loss"] = total_loss
        losses["loss_HiS"] = loss_HiS
        losses["loss_MiS"] = loss_MiS
        losses["loss_LoS"] = loss_LoS
        losses["loss_consis_HiS"] = loss_consis_HiS
        losses["loss_consis_LoS"] = loss_consis_LoS

        # print("loss_HiS:",loss_HiS.cpu().data, "loss_MiS:",loss_MiS.cpu().data, "loss_LoS:",loss_LoS.cpu().data, "loss_consis_HiS:",loss_consis_HiS.cpu().data, "loss_consis_LoS:",loss_consis_LoS.cpu().data, "loss:",total_loss.cpu().data)
        return losses

    def compute_consistency_loss(self, inputs, disp, disp_2):
        loss_c = 0
        cnt = 1.0
        for i in range(self.opt.batch_size):
            x1, y1 = int(inputs[("dxy")][i,0]), int(inputs[("dxy")][i,1])
            x2, y2 = int(inputs[("dxy_2")][i,0]), int(inputs[("dxy_2")][i,1])
            x_min, x_max = min(x1,x2), max(x1,x2)
            y_min, y_max = min(y1,y2), max(y1,y2)
            if x1==x2 and y1==y2:
                continue
            if abs(x1-x2) >= self.opt.width or abs(y1-y2) >= self.opt.height:
                continue
            x_s, y_s = x_max, y_max
            x_e, y_e = x_min+self.opt.width, y_min+self.opt.height
            if (x_e-x_s)*(y_e-y_s) <= (self.opt.width * self.opt.height)*1.0/4 or (x_e-x_s)*(y_e-y_s) >= (self.opt.width * self.opt.height)*7.0/8:
                continue
            disp_unite_1 = disp[i, 0, abs(y1-y_s):abs(y1-y_e), abs(x1-x_s):abs(x1-x_e)]
            disp_unite_2 = disp_2[i, 0, abs(y2-y_s):abs(y2-y_e), abs(x2-x_s):abs(x2-x_e)]
            diff_disp = ((disp_unite_1 - disp_unite_2).abs() /
                  (disp_unite_1 + disp_unite_2).abs()).clamp(0, 1)
            loss_c += diff_disp.mean()
            cnt += 1

        loss_c /= cnt
        return loss_c


    def compute_depth_losses(self, inputs, outputs):
        eval_measures = torch.zeros(8).cuda(device=self.device)
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
        disp = outputs["out_MiS"][("disp", 0)]
        disp_pred, _ = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        # disp_pred = torch.clamp(F.interpolate(
        #     disp_pred, [gt_height, gt_width], mode="bilinear", align_corners=False), 1e-3, 80)
        disp_pred = F.interpolate(disp_pred, [gt_height, gt_width], mode="bilinear", align_corners=False)
        depth_pred = 1 / disp_pred
        # depth_pred = depth_pred.detach()

        for i in range(depth_pred.shape[0]):
            depth_gt_each = depth_gt[i].squeeze()
            depth_pred_each = depth_pred[i].squeeze()
            # print("111: ", depth_gt_each.min(), depth_gt_each.max(), depth_pred_each.min(), depth_pred_each.max())
            # print("***: ", depth_gt_each.shape, depth_pred_each.shape)

            # mask = depth_gt_each > 0
            # mask = torch.logical_and(depth_gt_each > MIN_DEPTH, depth_gt_each < MAX_DEPTH)

            # # garg/eigen crop
            # crop_mask = torch.zeros_like(mask)
            # crop_mask[:, 153:371, 44:1197] = 1
            # # mask = mask * crop_mask
            # mask = torch.logical_and(mask, crop_mask)

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

            # for i in range(depth_pred.shape[0]):
            #     depth_pred_each = depth_pred[i]
            #     depth_gt_each = depth_gt[i]
            #     depth_pred_each *= torch.median(depth_gt_each) / torch.median(depth_pred_each)
            #     depth_pred_each = torch.clamp(depth_pred_each, min=1e-3, max=80)
            
            measures = compute_depth_errors(depth_gt_each, depth_pred_each)
            eval_measures[:7] += torch.tensor(measures).cuda(device=self.device)
            # eval_measures[:7] += measures
            eval_measures[7] += 1

        return eval_measures

        # for i, metric in enumerate(self.depth_metric_names):
        #     losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        
        if self.opt.use_gs:
            print_string = "epo {:>3} | bat {:>6} | ex/s: {:5.1f}" + \
                " | lH: {:.4f} | lM: {:.4f} | lL: {:.4f} | cH: {:.10f} | cL: {:.10f} | lM_gs: {:.4f} | loss: {:.5f} | te: {} | tl: {} | lr: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss_HiS"].cpu().data, losses["loss_MiS"].cpu().data, losses["loss_LoS"].cpu().data, 
                                    losses["loss_consis_HiS"].cpu().data, losses["loss_consis_LoS"].cpu().data, losses["loss_gs"].cpu().data, losses["loss"].cpu().data,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            print_string = "epo {:>3} | bat {:>6} | ex/s: {:5.1f}" + \
                " | lH: {:.4f} | lM: {:.4f} | lL: {:.4f} | cH: {:.10f} | cL: {:.10f} | loss: {:.5f} | te: {} | tl: {} | lr: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss_HiS"].cpu().data, losses["loss_MiS"].cpu().data, losses["loss_LoS"].cpu().data, 
                                    losses["loss_consis_HiS"].cpu().data, losses["loss_consis_LoS"].cpu().data, losses["loss"].cpu().data,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.state_dict()['param_groups'][0]['lr']))
            

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

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

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

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
