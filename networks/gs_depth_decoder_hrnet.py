from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
import torch.nn.functional as F

from networks.gs_feature_leverage import GaussianFeatureLeverage

# GS特征输出的深度解码器，用于从leveraged_feature得到最终的深度图
class GSDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(GSDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc  # np.array([64, 18, 36, 72, 144])
        # decoder stage 0-6 number of channels
        self.num_ch_dec = np.array([32, 64, 18, 36, 72, 144])
        self.use_skips = use_skips

        self.convs = OrderedDict()
        
        # ("parallel_conv", x, y) stage(scale) x, index y
        # stage_4 scale 1/16 
        stage_5_ch_num = self.num_ch_dec[5]
        stage_4_ch_num = self.num_ch_dec[4]
        self.convs[("parallel_conv", 4, 0)] = ConvBlock(stage_4_ch_num, stage_4_ch_num)
        stage_3_ch_num = self.num_ch_dec[3]
        self.convs[("parallel_conv", 3, 0)] = ConvBlock(stage_3_ch_num, stage_3_ch_num)
        self.convs[("parallel_conv", 3, 1)] = ConvBlock(stage_3_ch_num, stage_3_ch_num)
        stage_2_ch_num = self.num_ch_dec[2]
        self.convs[("parallel_conv", 2, 0)] = ConvBlock(stage_2_ch_num, stage_2_ch_num)
        self.convs[("parallel_conv", 2, 1)] = ConvBlock(stage_2_ch_num, stage_2_ch_num)
        self.convs[("parallel_conv", 2, 2)] = ConvBlock(stage_2_ch_num, stage_2_ch_num)
        stage_1_ch_num = self.num_ch_dec[1]
        self.convs[("parallel_conv", 1, 0)] = ConvBlock(stage_1_ch_num, stage_1_ch_num)
        # self.convs[("parallel_conv", 1, 1)] = ConvBlock(stage_1_ch_num, stage_0_ch_num)
        stage_0_ch_num = self.num_ch_dec[0]
        self.convs[("parallel_conv", 0, 0)] = ConvBlock(stage_0_ch_num, stage_0_ch_num)
        
        # ("conv1x1", x, y, z) 
        # upsampled feature conv for declining channels: No.z frome stage x to stage y
        # stage 5
        self.convs[("conv1x1", 5, 4, 0)] = ConvBlock1x1(stage_5_ch_num, stage_4_ch_num)
        self.convs[("conv1x1", 5, 3, 0)] = ConvBlock1x1(stage_5_ch_num, stage_3_ch_num)
        self.convs[("conv1x1", 5, 2, 0)] = ConvBlock1x1(stage_5_ch_num, stage_2_ch_num)
        # stage 4
        self.convs[("conv1x1", 4, 3, 0)] = ConvBlock1x1(stage_4_ch_num, stage_3_ch_num)
        self.convs[("conv1x1", 4, 3, 1)] = ConvBlock1x1(stage_4_ch_num, stage_3_ch_num)
        self.convs[("conv1x1", 4, 2, 0)] = ConvBlock1x1(stage_4_ch_num, stage_2_ch_num)
        self.convs[("conv1x1", 4, 2, 1)] = ConvBlock1x1(stage_4_ch_num, stage_2_ch_num)
        # stage 3
        self.convs[("conv1x1", 3, 2, 0)] = ConvBlock1x1(stage_3_ch_num, stage_2_ch_num)
        self.convs[("conv1x1", 3, 2, 1)] = ConvBlock1x1(stage_3_ch_num, stage_2_ch_num)
        self.convs[("conv1x1", 3, 2, 2)] = ConvBlock1x1(stage_3_ch_num, stage_2_ch_num)
        # stage 2
        self.convs[("conv1x1", 2, 1, 0)] = ConvBlock1x1(stage_2_ch_num, stage_1_ch_num)
        # stage 1
        self.convs[("conv1x1", 1, 0, 0)] = ConvBlock1x1(stage_1_ch_num, stage_0_ch_num)
        
        self.convs[("dispconv", 0)] = nn.Sequential(
            Conv3x3(stage_0_ch_num, stage_0_ch_num // 2),
            Conv3x3(stage_0_ch_num // 2, self.num_output_channels),
        )

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features):
        self.outputs = {}
        
        # d1_0 = input_features[0]
        d2_0 = input_features[0]
        d3_0 = input_features[1]
        d4_0 = input_features[2]
        d5_0 = input_features[3]

        d3to2_0 = updown_sample(d3_0, 2)
        d4to3_0 = updown_sample(d4_0, 2)
        d4to2_0 = updown_sample(d4_0, 4)
        d5to4_0 = updown_sample(d5_0, 2)
        d5to3_0 = updown_sample(d5_0, 4)
        d5to2_0 = updown_sample(d5_0, 8)

        d5to4_0 = self.convs[("conv1x1", 5, 4, 0)](d5to4_0)
        d5to3_0 = self.convs[("conv1x1", 5, 3, 0)](d5to3_0)
        d5to2_0 = self.convs[("conv1x1", 5, 2, 0)](d5to2_0)
        d4to3_0 = self.convs[("conv1x1", 4, 3, 0)](d4to3_0)
        d4to2_0 = self.convs[("conv1x1", 4, 2, 0)](d4to2_0)
        d3to2_0 = self.convs[("conv1x1", 3, 2, 0)](d3to2_0)

        d2_msf_0 = d5to2_0 + d4to2_0 + d3to2_0 + d2_0
        d3_msf_0 = d5to3_0 + d4to3_0 + d3_0
        d4_msf_0 = d5to4_0 + d4_0


        d4_1 = self.convs[("parallel_conv", 4, 0)](d4_msf_0)
        d3_1 = self.convs[("parallel_conv", 3, 1)](d3_msf_0)
        d2_1 = self.convs[("parallel_conv", 2, 1)](d2_msf_0)

        d3to2_1 = updown_sample(d3_1, 2)
        d4to3_1 = updown_sample(d4_1, 2)
        d4to2_1 = updown_sample(d4_1, 4)

        d3to2_1 = self.convs[("conv1x1", 3, 2, 1)](d3to2_1)
        d4to3_1 = self.convs[("conv1x1", 4, 3, 1)](d4to3_1)
        d4to2_1 = self.convs[("conv1x1", 4, 2, 1)](d4to2_1)

        d2_msf_1 = d3to2_1 + d4to2_1 + d2_1
        d3_msf_1 = d4to3_1 + d3_1

        d2_2 = self.convs[("parallel_conv", 2, 1)](d2_msf_1)
        d3_2 = self.convs[("parallel_conv", 3, 1)](d3_msf_1)
        
        d3to2_2 = updown_sample(d3_2, 2)
        d3to2_2 = self.convs[("conv1x1", 3, 2, 2)](d3to2_2)
        d2_msf_2 = d3to2_2 + d2_2

        d2_3 = self.convs[("parallel_conv", 2, 2)](d2_msf_2)
        
        d2to1_0 = updown_sample(d2_3, 2)
        d2to1_0 = self.convs[("conv1x1", 2, 1, 0)](d2to1_0)
        # d1_msf_0 = d2to1_0 + d1_0

        d1_1 = self.convs[("parallel_conv", 1, 0)](d2to1_0)
        # d1_1 = self.convs[("parallel_conv", 1, 0)](d1_msf_0)
        d1to0_0 = updown_sample(d1_1, 2)
        d1to0_0 = self.convs[("conv1x1", 1, 0, 0)](d1to0_0)
        
        d0_0 = self.convs[("parallel_conv", 0, 0)](d1to0_0)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d0_0))
        # self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d1_1))

        return self.outputs
    