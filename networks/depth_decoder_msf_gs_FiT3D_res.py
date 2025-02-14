# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


#depth_decoder
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
import torch.nn.functional as F


class DepthDecoder_MSF_GS_FiT_Res(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, 
                 use_gs=False, gs_scale=0, min_depth=0.1, max_depth=100.0, height=192, width=640):
        super(DepthDecoder_MSF_GS_FiT_Res, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc        #features in encoder, [64, 18, 36, 72, 144]
        self.use_gs = use_gs
        self.gs_scale = gs_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.height = height
        self.width = width
        self.num_ch_dec = num_ch_enc
        self.use_skips = use_skips

        self.convs = OrderedDict()

        if self.use_gs:
            if self.gs_scale != 0:
                self.min_level = int(torch.log2(torch.tensor(self.gs_scale)))
                for i in range(4, self.min_level - 1, -1) if self.min_level != 5 else range(4, 3, -1):
                    # upconv_0
                    num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

                    # upconv_1
                    num_ch_in = self.num_ch_dec[i]
                    if self.use_skips and i > 0:
                        num_ch_in = num_ch_in + self.num_ch_enc[i - 1]
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
                    
                if self.min_level == 5:
                    self.convs[("dispconv_init", self.gs_scale)] = Conv3x3(self.num_ch_dec[4], self.num_output_channels)
                else:
                    self.convs[("dispconv_init", self.gs_scale)] = Conv3x3(self.num_ch_dec[self.min_level], self.num_output_channels)
                    
            else:
                for i in range(4, 0, -1):
                    # upconv_0
                    num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

                    # upconv_1
                    num_ch_in = self.num_ch_dec[i]
                    if self.use_skips and i > 0:
                        num_ch_in = num_ch_in + self.num_ch_enc[i - 1]
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
                for s in range(1, 5):
                    self.convs[("dispconv_init", 2**s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
                self.convs[("dispconv_init", 32)] = Conv3x3(self.num_ch_dec[4], self.num_output_channels)
            # 3D Gaussian generator
            if self.gs_scale == 0:
                # 1. position head
                # self.convs[("gs_position_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                # self.convs[("gs_position_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 1)
                self.backproject_32 = BackprojectDepth_PointCloud(height=self.height//32, width=self.width//32)
                # self.convs[("gs_position_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                # self.convs[("gs_position_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 1)
                self.backproject_16 = BackprojectDepth_PointCloud(height=self.height//16, width=self.width//16)
                # self.convs[("gs_position_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                # self.convs[("gs_position_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 1)
                self.backproject_8 = BackprojectDepth_PointCloud(height=self.height//8, width=self.width//8)
                # self.convs[("gs_position_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                # self.convs[("gs_position_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 1)
                self.backproject_4 = BackprojectDepth_PointCloud(height=self.height//4, width=self.width//4)
                # self.convs[("gs_position_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                # self.convs[("gs_position_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 1)
                self.backproject_2 = BackprojectDepth_PointCloud(height=self.height//2, width=self.width//2)
                # 2. rotation head
                self.convs[("gs_rotation_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_rotation_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 4)
                self.convs[("gs_rotation_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_rotation_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 4)
                self.convs[("gs_rotation_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_rotation_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 4)
                self.convs[("gs_rotation_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_rotation_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 4)
                self.convs[("gs_rotation_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_rotation_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 4)
                # 3. scale head
                self.convs[("gs_scale_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_scale_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 3)
                self.convs[("gs_scale_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_scale_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 3)
                self.convs[("gs_scale_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_scale_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 3)
                self.convs[("gs_scale_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_scale_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 3)
                self.convs[("gs_scale_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_scale_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 3)
                # 4. opacity head
                self.convs[("gs_opacity_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_opacity_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 1)
                self.convs[("gs_opacity_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_opacity_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 1)
                self.convs[("gs_opacity_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_opacity_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 1)
                self.convs[("gs_opacity_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_opacity_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 1)
                self.convs[("gs_opacity_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_opacity_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 1)
                # 5. feature head
                self.convs[("gs_feature_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_feature_conv"), 1, 32] = ConvBlock(self.num_ch_enc[4], 64)
                self.convs[("gs_feature_conv"), 2, 32] = ConvBlock(64, self.num_ch_enc[4])
                self.convs[("gs_feature_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_feature_conv"), 1, 16] = ConvBlock(self.num_ch_enc[3], 64)
                self.convs[("gs_feature_conv"), 2, 16] = ConvBlock(64, self.num_ch_enc[3])
                self.convs[("gs_feature_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_feature_conv"), 1, 8] = ConvBlock(self.num_ch_enc[2], 64)
                self.convs[("gs_feature_conv"), 2, 8] = ConvBlock(64, self.num_ch_enc[2])
                self.convs[("gs_feature_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_feature_conv"), 1, 4] = ConvBlock(self.num_ch_enc[1], 64)
                self.convs[("gs_feature_conv"), 2, 4] = ConvBlock(64, self.num_ch_enc[1])
                self.convs[("gs_feature_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_feature_conv"), 1, 2] = ConvBlock(self.num_ch_enc[0], 64)
                self.convs[("gs_feature_conv"), 2, 2] = ConvBlock(64, self.num_ch_enc[0])

                # 6. 3D GS Rasterizer
                self.feature_rasterizer_32 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//32, image_width=self.width//32)
                self.feature_rasterizer_16 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//16, image_width=self.width//16)
                self.feature_rasterizer_8 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//8, image_width=self.width//8)
                self.feature_rasterizer_4 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//4, image_width=self.width//4)
                self.feature_rasterizer_2 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//2, image_width=self.width//2)
            elif self.gs_scale == 32:
                # 1. position head
                # self.convs[("gs_position_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                # self.convs[("gs_position_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 1)
                self.backproject_32 = BackprojectDepth_PointCloud(height=self.height//self.gs_scale, width=self.width//self.gs_scale)
                # 2. rotation head
                self.convs[("gs_rotation_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_rotation_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 4)
                # 3. scale head
                self.convs[("gs_scale_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_scale_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 3)
                # 4. opacity head
                self.convs[("gs_opacity_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_opacity_conv"), 1, 32] = Conv3x3(self.num_ch_enc[4], 1)
                # 5. feature head
                self.convs[("gs_feature_conv"), 0, 32] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
                self.convs[("gs_feature_conv"), 1, 32] = ConvBlock(self.num_ch_enc[4], 64)
                self.convs[("gs_feature_conv"), 2, 32] = ConvBlock(64, self.num_ch_enc[4])
                # 6. Rasterizer
                self.feature_rasterizer_32 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//self.gs_scale, 
                                                                    image_width=self.width//self.gs_scale)
            elif self.gs_scale == 16:
                # 1. position head
                # self.convs[("gs_position_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                # self.convs[("gs_position_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 1)
                self.backproject_16 = BackprojectDepth_PointCloud(height=self.height//self.gs_scale, width=self.width//self.gs_scale)
                # 2. rotation head
                self.convs[("gs_rotation_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_rotation_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 4)
                # 3. scale head
                self.convs[("gs_scale_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_scale_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 3)
                # 4. opacity head
                self.convs[("gs_opacity_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_opacity_conv"), 1, 16] = Conv3x3(self.num_ch_enc[3], 1)
                # 5. feature head
                self.convs[("gs_feature_conv"), 0, 16] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
                self.convs[("gs_feature_conv"), 1, 16] = ConvBlock(self.num_ch_enc[3], 64)
                self.convs[("gs_feature_conv"), 2, 16] = ConvBlock(64, self.num_ch_enc[3])
                # 6. Rasterizer
                self.feature_rasterizer_16 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//self.gs_scale, 
                                                                    image_width=self.width//self.gs_scale)
            elif self.gs_scale == 8:
                # 1. position head
                # self.convs[("gs_position_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                # self.convs[("gs_position_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 1)
                self.backproject_8 = BackprojectDepth_PointCloud(height=self.height//self.gs_scale, width=self.width//self.gs_scale)
                # 2. rotation head
                self.convs[("gs_rotation_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_rotation_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 4)
                # 3. scale head
                self.convs[("gs_scale_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_scale_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 3)
                # 4. opacity head
                self.convs[("gs_opacity_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_opacity_conv"), 1, 8] = Conv3x3(self.num_ch_enc[2], 1)
                # 5. feature head
                self.convs[("gs_feature_conv"), 0, 8] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
                self.convs[("gs_feature_conv"), 1, 8] = ConvBlock(self.num_ch_enc[2], 64)
                self.convs[("gs_feature_conv"), 2, 8] = ConvBlock(64, self.num_ch_enc[2])
                # 6. Rasterizer
                self.feature_rasterizer_8 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//self.gs_scale, 
                                                                    image_width=self.width//self.gs_scale)
            elif self.gs_scale == 4:
                # 1. position head
                # self.convs[("gs_position_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                # self.convs[("gs_position_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 1)
                self.backproject_4 = BackprojectDepth_PointCloud(height=self.height//self.gs_scale, width=self.width//self.gs_scale)
                # 2. rotation head
                self.convs[("gs_rotation_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_rotation_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 4)
                # 3. scale head
                self.convs[("gs_scale_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_scale_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 3)
                # 4. opacity head
                self.convs[("gs_opacity_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_opacity_conv"), 1, 4] = Conv3x3(self.num_ch_enc[1], 1)
                # 5. feature head
                self.convs[("gs_feature_conv"), 0, 4] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
                self.convs[("gs_feature_conv"), 1, 4] = ConvBlock(self.num_ch_enc[1], 64)
                self.convs[("gs_feature_conv"), 2, 4] = ConvBlock(64, self.num_ch_enc[1])
                # 6. Rasterizer
                self.feature_rasterizer_4 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//self.gs_scale, 
                                                                    image_width=self.width//self.gs_scale)
            elif self.gs_scale == 2:
                # 1. position head
                # self.convs[("gs_position_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                # self.convs[("gs_position_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 1)
                self.backproject_2 = BackprojectDepth_PointCloud(height=self.height//self.gs_scale, width=self.width//self.gs_scale)
                # 2. rotation head
                self.convs[("gs_rotation_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_rotation_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 4)
                # 3. scale head
                self.convs[("gs_scale_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_scale_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 3)
                # 2. opacity head
                self.convs[("gs_opacity_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_opacity_conv"), 1, 2] = Conv3x3(self.num_ch_enc[0], 1)
                # 5. feature head
                self.convs[("gs_feature_conv"), 0, 2] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
                self.convs[("gs_feature_conv"), 1, 2] = ConvBlock(self.num_ch_enc[0], 64)
                self.convs[("gs_feature_conv"), 2, 2] = ConvBlock(64, self.num_ch_enc[0])
                # 6. Rasterizer
                self.feature_rasterizer_2 = Rasterize_Gaussian_Feature_FiT3D(image_height=self.height//self.gs_scale, 
                                                                    image_width=self.width//self.gs_scale)
            
        # decoder
        self.convs[("parallel_conv"), 0, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 0, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 0, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 0, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("parallel_conv"), 0, 4] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
        self.convs[("conv1x1", 0, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 4_3)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[3])
        self.convs[("conv1x1", 0, 4_2)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 4_1)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 1, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 1, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 1, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("conv1x1", 1, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 1, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 1, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 3, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("conv1x1", 3, 1_0)] = ConvBlock1x1(self.num_ch_enc[1], self.num_ch_enc[0])

        self.convs[("parallel_conv"), 4, 0] = ConvBlock(self.num_ch_enc[0], 32)
        self.convs[("parallel_conv"), 5, 0] = ConvBlock(32, 16)
        self.convs[("dispconv", 0)] = Conv3x3(16, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features, inv_K, K):
        self.outputs = {}

        # features in encoder
        e4 = input_features[4]
        e3 = input_features[3]
        e2 = input_features[2]
        e1 = input_features[1]
        e0 = input_features[0]

        d0_0 = self.convs[("parallel_conv"), 0, 0](e0)
        d0_1 = self.convs[("parallel_conv"), 0, 1](e1)
        d0_2 = self.convs[("parallel_conv"), 0, 2](e2)
        d0_3 = self.convs[("parallel_conv"), 0, 3](e3)
        d0_4 = self.convs[("parallel_conv"), 0, 4](e4)

        # init decoder
        x = input_features[-1]
        if self.use_gs:
            if self.gs_scale != 0:
                for i in range(4, self.min_level - 1, -1) if self.min_level != 5 else range(4, 3, -1):
                    x = self.convs[("upconv", i, 0)](x)
                    if i == 4 and i == self.min_level - 1:
                        self.outputs[("disp_init", 2**(i + 1))] = self.sigmoid(self.convs[("dispconv_init", 2**(i + 1))](x))
                    else:
                        x = [upsample(x)]
                        if self.use_skips and i > 0:
                            x = x + [input_features[i - 1]]
                        x = torch.cat(x, 1)
                        x = self.convs[("upconv", i, 1)](x)
                        if i == self.min_level:
                            self.outputs[("disp_init", self.gs_scale)] = self.sigmoid(self.convs[("dispconv_init", self.gs_scale)](x))
            else:
                for i in range(4, 0, -1):
                    x = self.convs[("upconv", i, 0)](x)
                    if i == 4:
                        self.outputs[("disp_init", 2**(i + 1))] = self.sigmoid(self.convs[("dispconv_init", 2**(i + 1))](x))
                    x = [upsample(x)]
                    if self.use_skips and i > 0:
                        x = x + [input_features[i - 1]]
                    x = torch.cat(x, 1)
                    x = self.convs[("upconv", i, 1)](x)

                    self.outputs[("disp_init", 2**i)] = self.sigmoid(self.convs[("dispconv_init", 2**i)](x))

        if self.use_gs:
            if self.gs_scale == 0:
                bs = e4.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_32 = self.outputs[("disp_init", 32)]
                _, depth_init_32 = disp_to_depth(disp_init_32, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 32)] = depth_init_32
                depth_init_32 = depth_init_32.clamp(min=self.min_depth, max=self.max_depth)
                e4_position_32 = self.backproject_32(depth_init_32, inv_K[4])
                e4_position = e4_position_32.permute(0, 2, 1).contiguous()
                # 2. rotation head, [B, H*W, 4]
                e4_rotation = self.convs[("gs_rotation_conv"), 0, 32](e4)
                e4_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 32](e4_rotation), dim=1, p=2)
                e4_rotation = e4_rotation.permute(0, 2, 3, 1).contiguous()
                e4_rotation = e4_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e4_scale = self.convs[("gs_scale_conv"), 0, 32](e4)
                e4_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 32](e4_scale))
                e4_scale = e4_scale.permute(0, 2, 3, 1).contiguous()
                e4_scale = e4_scale.view(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e4_opacity = self.convs[("gs_opacity_conv"), 0, 32](e4)
                e4_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 32](e4_opacity))
                e4_opacity = e4_opacity.permute(0, 2, 3, 1).contiguous()
                e4_opacity = e4_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e4_feature = self.convs[("gs_feature_conv"), 0, 32](e4)
                e4_feature = self.convs[("gs_feature_conv"), 1, 32](e4_feature)
                e4_feature = e4_feature.permute(0, 2, 3, 1).contiguous()
                e4_feature = e4_feature.view(bs, -1, e4_feature.shape[-1])
                gs_feature = self.feature_rasterizer_32(e4_position, e4_rotation, e4_scale, e4_opacity, e4_feature, inv_K[4], K[4])
                e4 = self.convs[("gs_feature_conv"), 2, 32](gs_feature)

                bs = e3.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_16 = self.outputs[("disp_init", 16)]
                _, depth_init_16 = disp_to_depth(disp_init_16, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 16)] = depth_init_16
                depth_init_16 = depth_init_16.clamp(min=self.min_depth, max=self.max_depth)
                e3_position_16 = self.backproject_16(depth_init_16, inv_K[3])
                e3_position = e3_position_16.reshape(bs, -1, 3)
                # 2. rotation head, [B, H*W, 4]
                e3_rotation = self.convs[("gs_rotation_conv"), 0, 16](e3)
                e3_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 16](e3_rotation), dim=1, p=2)
                e3_rotation = e3_rotation.reshape(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e3_scale = self.convs[("gs_scale_conv"), 0, 16](e3)
                e3_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 16](e3_scale))
                e3_scale = e3_scale.reshape(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e3_opacity = self.convs[("gs_opacity_conv"), 0, 16](e3)
                e3_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 16](e3_opacity))
                e3_opacity = e3_opacity.reshape(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e3_feature = self.convs[("gs_feature_conv"), 0, 16](e3)
                e3_feature = self.convs[("gs_feature_conv"), 1, 16](e3_feature)
                e3_feature = e3_feature.reshape(bs, -1, e3_feature.shape[1])
                gs_feature = self.feature_rasterizer_16(e3_position, e3_rotation, e3_scale, e3_opacity, e3_feature, inv_K[3], K[3])
                e3 = self.convs[("gs_feature_conv"), 2, 16](gs_feature)

                bs = e2.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_8 = self.outputs[("disp_init", 8)]
                _, depth_init_8 = disp_to_depth(disp_init_8, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 8)] = depth_init_8
                depth_init_8 = depth_init_8.clamp(min=self.min_depth, max=self.max_depth)
                e2_position_8 = self.backproject_8(depth_init_8, inv_K[2])
                e2_position = e2_position_8.permute(0, 2, 1).contiguous()
                e2_position = e2_position.view(bs, -1, 3)
                # 2. rotation head, [B, H*W, 4]
                e2_rotation = self.convs[("gs_rotation_conv"), 0, 8](e2)
                e2_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 8](e2_rotation), dim=1, p=2)
                e2_rotation = e2_rotation.permute(0, 2, 3, 1).contiguous()
                e2_rotation = e2_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e2_scale = self.convs[("gs_scale_conv"), 0, 8](e2)
                e2_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 8](e2_scale))
                e2_scale = e2_scale.permute(0, 2, 3, 1).contiguous()
                e2_scale = e2_scale.view(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e2_opacity = self.convs[("gs_opacity_conv"), 0, 8](e2)
                e2_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 8](e2_opacity))
                e2_opacity = e2_opacity.permute(0, 2, 3, 1).contiguous()
                e2_opacity = e2_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e2_feature = self.convs[("gs_feature_conv"), 0, 8](e2)
                e2_feature = self.convs[("gs_feature_conv"), 1, 8](e2_feature)
                e2_feature = e2_feature.permute(0, 2, 3, 1).contiguous()
                e2_feature = e2_feature.view(bs, -1, e2_feature.shape[-1])
                gs_feature = self.feature_rasterizer_8(e2_position, e2_rotation, e2_scale, e2_opacity, e2_feature, inv_K[2], K[2])
                e2 = self.convs[("gs_feature_conv"), 2, 8](gs_feature)

                bs = e1.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_4 = self.outputs[("disp_init", 4)]
                _, depth_init_4 = disp_to_depth(disp_init_4, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 4)] = depth_init_4
                depth_init_4 = depth_init_4.clamp(min=self.min_depth, max=self.max_depth)
                e1_position_4 = self.backproject_4(depth_init_4, inv_K[1])
                e1_position = e1_position_4.permute(0, 2, 1).contiguous()
                e1_position = e1_position.view(bs, -1, 3)
                # 2. rotation head, [B, H*W, 4]
                e1_rotation = self.convs[("gs_rotation_conv"), 0, 4](e1)
                e1_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 4](e1_rotation), dim=1, p=2)
                e1_rotation = e1_rotation.permute(0, 2, 3, 1).contiguous()
                e1_rotation = e1_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e1_scale = self.convs[("gs_scale_conv"), 0, 4](e1)
                e1_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 4](e1_scale))
                e1_scale = e1_scale.permute(0, 2, 3, 1).contiguous()
                e1_scale = e1_scale.view(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e1_opacity = self.convs[("gs_opacity_conv"), 0, 4](e1)
                e1_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 4](e1_opacity))
                e1_opacity = e1_opacity.permute(0, 2, 3, 1).contiguous()
                e1_opacity = e1_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e1_feature = self.convs[("gs_feature_conv"), 0, 4](e1)
                e1_feature = self.convs[("gs_feature_conv"), 1, 4](e1_feature)
                e1_feature = e1_feature.permute(0, 2, 3, 1).contiguous()
                e1_feature = e1_feature.view(bs, -1, e1_feature.shape[-1])
                gs_feature = self.feature_rasterizer_4(e1_position, e1_rotation, e1_scale, e1_opacity, e1_feature, inv_K[1], K[1])
                e1 = self.convs[("gs_feature_conv"), 2, 4](gs_feature)

                bs = e0.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_2 = self.outputs[("disp_init", 2)]
                _, depth_init_2 = disp_to_depth(disp_init_2, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 2)] = depth_init_2
                depth_init_2 = depth_init_2.clamp(min=self.min_depth, max=self.max_depth)
                e0_position_2 = self.backproject_2(depth_init_2, inv_K[0])
                e0_position = e0_position_2.permute(0, 2, 1).contiguous()
                e0_position = e0_position.view(bs, -1, 3)
                # 2. rotation head, [B, H*W, 2]
                e0_rotation = self.convs[("gs_rotation_conv"), 0, 2](e0)
                e0_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 2](e0_rotation), dim=1, p=2)
                e0_rotation = e0_rotation.permute(0, 2, 3, 1).contiguous()
                e0_rotation = e0_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e0_scale = self.convs[("gs_scale_conv"), 0, 2](e0)
                e0_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 2](e0_scale))
                e0_scale = e0_scale.permute(0, 2, 3, 1).contiguous()
                e0_scale = e0_scale.view(bs, -1, 3)
                # 2. opacity head, [B, H*W, 1]
                e0_opacity = self.convs[("gs_opacity_conv"), 0, 2](e0)
                e0_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 2](e0_opacity))
                e0_opacity = e0_opacity.permute(0, 2, 3, 1).contiguous()
                e0_opacity = e0_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e0_feature = self.convs[("gs_feature_conv"), 0, 2](e0)
                e0_feature = self.convs[("gs_feature_conv"), 1, 2](e0_feature)
                e0_feature = e0_feature.permute(0, 2, 3, 1).contiguous()
                e0_feature = e0_feature.view(bs, -1, e0_feature.shape[-1])
                gs_feature = self.feature_rasterizer_2(e0_position, e0_rotation, e0_scale, e0_opacity, e0_feature, inv_K[0], K[0])
                e0 = self.convs[("gs_feature_conv"), 2, 2](gs_feature)

            elif self.gs_scale == 32:
                bs = e4.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_32 = self.outputs[("disp_init", 32)]
                _, depth_init_32 = disp_to_depth(disp_init_32, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 32)] = depth_init_32
                depth_init_32 = depth_init_32.clamp(min=self.min_depth, max=self.max_depth)
                e4_position_32 = self.backproject_32(depth_init_32, inv_K[0])
                e4_position = e4_position_32.permute(0, 2, 1).contiguous()
                # 2. rotation head, [B, H*W, 4]
                e4_rotation = self.convs[("gs_rotation_conv"), 0, 32](e4)
                e4_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 32](e4_rotation), dim=1, p=2)
                e4_rotation = e4_rotation.permute(0, 2, 3, 1).contiguous()
                e4_rotation = e4_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e4_scale = self.convs[("gs_scale_conv"), 0, 32](e4)
                e4_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 32](e4_scale))
                e4_scale = e4_scale.permute(0, 2, 3, 1).contiguous()
                e4_scale = e4_scale.view(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e4_opacity = self.convs[("gs_opacity_conv"), 0, 32](e4)
                e4_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 32](e4_opacity))
                e4_opacity = e4_opacity.permute(0, 2, 3, 1).contiguous()
                e4_opacity = e4_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e4_feature = self.convs[("gs_feature_conv"), 0, 32](e4)
                e4_feature = self.convs[("gs_feature_conv"), 1, 32](e4_feature)
                e4_feature = e4_feature.permute(0, 2, 3, 1).contiguous()
                e4_feature = e4_feature.view(bs, -1, e4_feature.shape[-1])
                gs_feature = self.feature_rasterizer_32(e4_position, e4_rotation, e4_scale, e4_opacity, e4_feature, inv_K[0], K[0])
                e4 = self.convs[("gs_feature_conv"), 2, 32](gs_feature)

            elif self.gs_scale == 16:
                bs = e3.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_16 = self.outputs[("disp_init", 16)]
                _, depth_init_16 = disp_to_depth(disp_init_16, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 16)] = depth_init_16
                depth_init_16 = depth_init_16.clamp(min=self.min_depth, max=self.max_depth)
                e3_position_16 = self.backproject_16(depth_init_16, inv_K[0])
                e3_position = e3_position_16.reshape(bs, -1, 3)
                # 2. rotation head, [B, H*W, 4]
                e3_rotation = self.convs[("gs_rotation_conv"), 0, 16](e3)
                e3_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 16](e3_rotation), dim=1, p=2)
                e3_rotation = e3_rotation.reshape(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e3_scale = self.convs[("gs_scale_conv"), 0, 16](e3)
                e3_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 16](e3_scale))
                e3_scale = e3_scale.reshape(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e3_opacity = self.convs[("gs_opacity_conv"), 0, 16](e3)
                e3_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 16](e3_opacity))
                e3_opacity = e3_opacity.reshape(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e3_feature = self.convs[("gs_feature_conv"), 0, 16](e3)
                e3_feature = self.convs[("gs_feature_conv"), 1, 16](e3_feature)
                e3_feature = e3_feature.reshape(bs, -1, e3_feature.shape[1])
                gs_feature = self.feature_rasterizer_16(e3_position, e3_rotation, e3_scale, e3_opacity, e3_feature, inv_K[0], K[0])
                e3 = self.convs[("gs_feature_conv"), 2, 16](gs_feature)

            elif self.gs_scale == 8:
                bs = e2.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_8 = self.outputs[("disp_init", 8)]
                _, depth_init_8 = disp_to_depth(disp_init_8, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 8)] = depth_init_8
                depth_init_8 = depth_init_8.clamp(min=self.min_depth, max=self.max_depth)
                e2_position_8 = self.backproject_8(depth_init_8, inv_K[0])
                e2_position = e2_position_8.permute(0, 2, 1).contiguous()
                e2_position = e2_position.view(bs, -1, 3)
                # 2. rotation head, [B, H*W, 4]
                e2_rotation = self.convs[("gs_rotation_conv"), 0, 8](e2)
                e2_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 8](e2_rotation), dim=1, p=2)
                e2_rotation = e2_rotation.permute(0, 2, 3, 1).contiguous()
                e2_rotation = e2_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e2_scale = self.convs[("gs_scale_conv"), 0, 8](e2)
                e2_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 8](e2_scale))
                e2_scale = e2_scale.permute(0, 2, 3, 1).contiguous()
                e2_scale = e2_scale.view(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e2_opacity = self.convs[("gs_opacity_conv"), 0, 8](e2)
                e2_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 8](e2_opacity))
                e2_opacity = e2_opacity.permute(0, 2, 3, 1).contiguous()
                e2_opacity = e2_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e2_feature = self.convs[("gs_feature_conv"), 0, 8](e2)
                e2_feature = self.convs[("gs_feature_conv"), 1, 8](e2_feature)
                e2_feature = e2_feature.permute(0, 2, 3, 1).contiguous()
                e2_feature = e2_feature.view(bs, -1, e2_feature.shape[-1])
                gs_feature = self.feature_rasterizer_8(e2_position, e2_rotation, e2_scale, e2_opacity, e2_feature, inv_K[0], K[0])
                e2 = self.convs[("gs_feature_conv"), 2, 8](gs_feature)

            elif self.gs_scale == 4:
                bs = e1.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_4 = self.outputs[("disp_init", 4)]
                _, depth_init_4 = disp_to_depth(disp_init_4, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 4)] = depth_init_4
                depth_init_4 = depth_init_4.clamp(min=self.min_depth, max=self.max_depth)
                e1_position_4 = self.backproject_4(depth_init_4, inv_K[0])
                e1_position = e1_position_4.permute(0, 2, 1).contiguous()
                e1_position = e1_position.view(bs, -1, 3)
                # 2. rotation head, [B, H*W, 4]
                e1_rotation = self.convs[("gs_rotation_conv"), 0, 4](e1)
                e1_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 4](e1_rotation), dim=1, p=2)
                e1_rotation = e1_rotation.permute(0, 2, 3, 1).contiguous()
                e1_rotation = e1_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e1_scale = self.convs[("gs_scale_conv"), 0, 4](e1)
                e1_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 4](e1_scale))
                e1_scale = e1_scale.permute(0, 2, 3, 1).contiguous()
                e1_scale = e1_scale.view(bs, -1, 3)
                # 4. opacity head, [B, H*W, 1]
                e1_opacity = self.convs[("gs_opacity_conv"), 0, 4](e1)
                e1_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 4](e1_opacity))
                e1_opacity = e1_opacity.permute(0, 2, 3, 1).contiguous()
                e1_opacity = e1_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e1_feature = self.convs[("gs_feature_conv"), 0, 4](e1)
                e1_feature = self.convs[("gs_feature_conv"), 1, 4](e1_feature)
                e1_feature = e1_feature.permute(0, 2, 3, 1).contiguous()
                e1_feature = e1_feature.view(bs, -1, e1_feature.shape[-1])
                gs_feature = self.feature_rasterizer_4(e1_position, e1_rotation, e1_scale, e1_opacity, e1_feature, inv_K[0], K[0])
                e1 = self.convs[("gs_feature_conv"), 2, 4](gs_feature)

            elif self.gs_scale == 2:
                bs = e0.shape[0]
                # 1. position head [B, H*W, 3]
                disp_init_2 = self.outputs[("disp_init", 2)]
                _, depth_init_2 = disp_to_depth(disp_init_2, self.min_depth, self.max_depth)
                self.outputs[("depth_init", 2)] = depth_init_2
                depth_init_2 = depth_init_2.clamp(min=self.min_depth, max=self.max_depth)
                e0_position_2 = self.backproject_2(depth_init_2, inv_K[0])
                e0_position = e0_position_2.permute(0, 2, 1).contiguous()
                e0_position = e0_position.view(bs, -1, 3)
                # 2. rotation head, [B, H*W, 2]
                e0_rotation = self.convs[("gs_rotation_conv"), 0, 2](e0)
                e0_rotation = F.normalize(self.convs[("gs_rotation_conv"), 1, 2](e0_rotation), dim=1, p=2)
                e0_rotation = e0_rotation.permute(0, 2, 3, 1).contiguous()
                e0_rotation = e0_rotation.view(bs, -1, 4)
                # 3. scale head, [B, H*W, 3]
                e0_scale = self.convs[("gs_scale_conv"), 0, 2](e0)
                e0_scale = torch.abs(self.convs[("gs_scale_conv"), 1, 2](e0_scale))
                e0_scale = e0_scale.permute(0, 2, 3, 1).contiguous()
                e0_scale = e0_scale.view(bs, -1, 3)
                # 2. opacity head, [B, H*W, 1]
                e0_opacity = self.convs[("gs_opacity_conv"), 0, 2](e0)
                e0_opacity = self.sigmoid(self.convs[("gs_opacity_conv"), 1, 2](e0_opacity))
                e0_opacity = e0_opacity.permute(0, 2, 3, 1).contiguous()
                e0_opacity = e0_opacity.view(bs, -1, 1)
                # 5. feature head, [B, H*W, C]
                e0_feature = self.convs[("gs_feature_conv"), 0, 2](e0)
                e0_feature = self.convs[("gs_feature_conv"), 1, 2](e0_feature)
                e0_feature = e0_feature.permute(0, 2, 3, 1).contiguous()
                e0_feature = e0_feature.view(bs, -1, e0_feature.shape[-1])
                gs_feature = self.feature_rasterizer_2(e0_position, e0_rotation, e0_scale, e0_opacity, e0_feature, inv_K[0], K[0])
                e0 = self.convs[("gs_feature_conv"), 2, 2](gs_feature)

        if self.use_gs:
            if self.gs_scale == 0:
                # self.outputs[("gs_feature", 2)] = e0
                # self.outputs[("gs_feature", 4)] = e1
                # self.outputs[("gs_feature", 8)] = e2
                # self.outputs[("gs_feature", 16)] = e3
                # self.outputs[("gs_feature", 32)] = e4
                d0_0 = d0_0 + e0
                d0_1 = d0_1 + e1
                d0_2 = d0_2 + e2
                d0_3 = d0_3 + e3
                d0_4 = d0_4 + e4
            elif self.gs_scale == 2:
                # self.outputs[("gs_feature", 2)] = e0
                d0_0 = d0_0 + e0
            elif self.gs_scale == 4:
                # self.outputs[("gs_feature", 4)] = e1
                d0_1 = d0_1 + e1
            elif self.gs_scale == 8:
                # self.outputs[("gs_feature", 8)] = e2
                d0_2 = d0_2 + e2
            elif self.gs_scale == 16:
                # self.outputs[("gs_feature", 16)] = e3
                d0_3 = d0_3 + e3
            elif self.gs_scale == 32:
                # self.outputs[("gs_feature", 32)] = e4
                d0_4 = d0_4 + e4

        d0_2_1 = updown_sample(d0_2, 2)
        d0_3_2 = updown_sample(d0_3, 2)
        d0_3_1 = updown_sample(d0_3, 4)
        d0_4_3 = updown_sample(d0_4, 2)
        d0_4_2 = updown_sample(d0_4, 4)
        d0_4_1 = updown_sample(d0_4, 8)

        d0_2_1 = self.convs[("conv1x1", 0, 2_1)](d0_2_1)
        d0_3_2 = self.convs[("conv1x1", 0, 3_2)](d0_3_2)
        d0_3_1 = self.convs[("conv1x1", 0, 3_1)](d0_3_1)
        d0_4_3 = self.convs[("conv1x1", 0, 4_3)](d0_4_3)
        d0_4_2 = self.convs[("conv1x1", 0, 4_2)](d0_4_2)
        d0_4_1 = self.convs[("conv1x1", 0, 4_1)](d0_4_1)

        d0_1_msf = d0_1 + d0_2_1 + d0_3_1 + d0_4_1
        d0_2_msf = d0_2 + d0_3_2 + d0_4_2
        d0_3_msf = d0_3 + d0_4_3


        d1_1 = self.convs[("parallel_conv"), 1, 1](d0_1_msf)
        d1_2 = self.convs[("parallel_conv"), 1, 2](d0_2_msf)
        d1_3 = self.convs[("parallel_conv"), 1, 3](d0_3_msf)

        d1_2_1 = updown_sample(d1_2, 2)
        d1_3_2 = updown_sample(d1_3, 2)
        d1_3_1 = updown_sample(d1_3, 4)

        d1_2_1 = self.convs[("conv1x1", 1, 2_1)](d1_2_1)
        d1_3_2 = self.convs[("conv1x1", 1, 3_2)](d1_3_2)
        d1_3_1 = self.convs[("conv1x1", 1, 3_1)](d1_3_1)

        d1_1_msf = d1_1 + d1_2_1 + d1_3_1
        d1_2_msf = d1_2 + d1_3_2


        d2_1 = self.convs[("parallel_conv"), 2, 1](d1_1_msf)
        d2_2 = self.convs[("parallel_conv"), 2, 2](d1_2_msf)
        d2_2_1 = updown_sample(d2_2, 2)
        d2_2_1 = self.convs[("conv1x1", 2, 2_1)](d2_2_1)
        d2_1_msf = d2_1 + d2_2_1


        d3_0 = self.convs[("parallel_conv"), 3, 0](d0_0)
        d3_1 = self.convs[("parallel_conv"), 3, 1](d2_1_msf)
        d3_1_0 = updown_sample(d3_1, 2)
        d3_1_0 = self.convs[("conv1x1", 3, 1_0)](d3_1_0)
        d3_0_msf = d3_0 + d3_1_0


        d4_0 = self.convs[("parallel_conv"), 4, 0](d3_0_msf)
        d4_0 = updown_sample(d4_0, 2)
        d5 = self.convs[("parallel_conv"), 5, 0](d4_0)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d5))

        return self.outputs     #single-scale depth
    