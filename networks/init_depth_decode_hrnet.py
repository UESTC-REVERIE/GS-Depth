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

from networks.gs_feature_leverage import GaussianFeatureLeverage

# hrnet深度解码器，用于提供初始深度图（gs center）
class InitDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(InitDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc[-4:]  # features in encoder(HRNet), [64, 18, 36, 72, 144]

        self.num_ch_dec = np.array([18, 36, 72, 144])
        self.use_skips = use_skips

        self.convs = OrderedDict()
        #decoder
        for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 3 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i != 3:
                num_ch_in = num_ch_in + self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
        for s in range(4):
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)


        self.init_decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features):
        self.outputs = {}

        input_features = input_features[-4:]
        # decoder
        x = input_features[-1]
        # 预测高斯参数的特征输入
        decoder_features = {}
        # 预测初始深度图
        for i in range(3, -1, -1): # up-conv 每个尺度上都进行两次卷积，并上采样
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)] if i != 3 else [x]
            if self.use_skips and i !=3:
                x += [input_features[i]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales: # 在{0，1，2，3}尺度上都进行一次卷积生成视差图来计算loss
                decoder_features[i] = x
                self.outputs[("init_disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs, decoder_features