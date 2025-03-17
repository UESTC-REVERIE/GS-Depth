# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
# os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
# os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
# os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import random
import numpy as np
# import cv2
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

# cv2.setNumThreads(0)

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
        gs_scale
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 is_val=False,
                 is_test=False,
                 img_ext='.png'
                #  gs_scale=0
                 ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.ANTIALIAS   #a high-quality downsampling filter
        self.interp = Image.LANCZOS   #a high-quality downsampling filter  

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test
        self.img_ext = img_ext
        # self.gs_scale = gs_scale

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
            transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(6):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        ##print(self.resize) #{0: Resize(size=(192, 640), interpolation=lanczos, max_size=None, antialias=warn)}
        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            # frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(6):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        # print("test2: ", inputs)
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                if i == -1:
                    continue
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
        # print("test3: ", inputs)
    def load_intrinsics(self, folder, frame_index):
        return self.K.copy()
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        # do_crop_aug = self.is_train

        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
        if type(self).__name__ in ["CityscapesPreprocessedDataset", "CityscapesEvalDataset"]:
            inputs.update(self.get_colors(folder, frame_index, side, do_flip))
        else:
            for i in self.frame_idxs: # default: [0, -1, 1]
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                    except FileNotFoundError as e:
                        if i != 0:
                            # fill with dummy values
                            inputs[("color", i, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                            # poses[i] = None
                        else:
                            raise FileNotFoundError(f'Cannot find frame - make sure your '
                                                    f'--data_path is set correctly, or try adding'
                                                    f' the --png flag. {e}')
        
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
            # color_aug = transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        # print("test1: ", inputs)
        self.preprocess(inputs, color_aug)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(6):
            K = self.load_intrinsics(folder, frame_index)
            # 根据当前尺度的图像分辨率重新计算实际的焦距和主点坐标。
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # if self.gs_scale == 0:
        #     for scale in range(0, 6):
        #         # K = self.K.copy()
        #         K = self.load_intrinsics(folder, frame_index)
        #         K[0, :] *= self.width // (2 ** scale)
        #         K[1, :] *= self.height // (2 ** scale)
        #         inv_K = np.linalg.pinv(K)
        #         inputs[("K_gs", scale)] = torch.from_numpy(K)
        #         inputs[("inv_K_gs", scale)] = torch.from_numpy(inv_K)
        # else:
        #     scale = self.gs_scale
        #     # K = self.K.copy()
        #     K = self.load_intrinsics(folder, frame_index)
        #     K[0, :] *= self.width // scale
        #     K[1, :] *= self.height // scale
        #     inv_K = np.linalg.pinv(K)
        #     inputs[("K_gs", scale)] = torch.from_numpy(K)
        #     inputs[("inv_K_gs", scale)] = torch.from_numpy(inv_K)

        #删除原尺寸图像，-1表示原始图像(1242, 375)
        for i in self.frame_idxs:
            if self.is_test:
                inputs[("color", i, -1)] = self.to_tensor(inputs[("color", i, -1)])
            else:
                del inputs[("color", i, -1)]

        if self.is_val or self.is_test:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
            

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            stereo_T_inv = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            stereo_T_inv[0, 3] = side_sign * baseline_sign * (-0.1)

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
            inputs["stereo_T_inv"] = torch.from_numpy(stereo_T_inv)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
