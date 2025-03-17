# # Copyright Niantic 2019. Patent Pending. All rights reserved.
# #
# # This software is licensed under the terms of the Monodepth2 licence
# # which allows for non-commercial use only, the full terms of which are made
# # available in the LICENSE file.


# #depth_decoder
# from __future__ import absolute_import, division, print_function

# import numpy as np
# import torch
# import torch.nn as nn

# from collections import OrderedDict
# from layers import *
# import torch.nn.functional as F

# from networks.gs_feature_leverage import GaussianFeatureLeverage


# class GSDepthDecoder(nn.Module):
#     def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, 
#                  use_gs=False, gs_scale=0, min_depth=0.1, max_depth=100.0, height=192, width=640):
#         super(GSDepthDecoder, self).__init__()

#         self.num_output_channels = num_output_channels
#         self.scales = scales

#         self.num_ch_enc = num_ch_enc
#         self.use_gs = use_gs
#         self.gs_scale = gs_scale
#         self.min_depth = min_depth
#         self.max_depth = max_depth
#         self.height = height
#         self.width = width

#         self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#         self.use_skips = use_skips

#         self.convs = OrderedDict()
#         #decoder
#         for i in range(4, -1, -1):
#             # upconv_0
#             num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

#             # upconv_1
#             num_ch_in = self.num_ch_dec[i]
#             if self.use_skips and i > 0:
#                 num_ch_in = num_ch_in + self.num_ch_enc[i - 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
#         for s in range(4):
#             # TODO 测试随分辨率变化特征维度
#             # self.convs[("gs_leverage", s)] = GaussianFeatureLeverage(
#             #     feat_ch_in=self.num_ch_dec[s], 
#             #     feat_ch_out=self.num_ch_dec[s],
#             #     scale=s,
#             #     height=self.height, width=self.width,
#             #     leveraged_feat_ch = 64 // (2 ** s), # 需要修改光栅器默认输出维度不为64
#             #     min_depth=self.min_depth, max_depth=self.max_depth
#             # )
#             self.convs[("gs_leverage", s)] = GaussianFeatureLeverage(
#                 feat_ch_in=self.num_ch_dec[s], # TODO concat初始的深度图提供结构信息
#                 feat_ch_out=self.num_ch_dec[s],
#                 scale=s,
#                 height=self.height, width=self.width,
#                 leveraged_feat_ch=64,
#                 min_depth=self.min_depth, max_depth=self.max_depth
#             )
#             self.convs[("init_dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
#             # TODO concat初始的深度图提供结构信息
#             self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
#         # TODO 测试统一的一个高斯表达，用不同的rasterization得到多分辨率       
#         # self.convs[("gs_leverage", 0)] = GaussianFeatureLeverage(
#         #     feat_ch_in=self.num_ch_dec[0], 
#         #     feat_ch_out=self.num_ch_dec[0],
#         #     scale=s,
#         #     height=self.height, width=self.width,
#         #     leveraged_feat_ch=64,
#         #     min_depth=self.min_depth, max_depth=self.max_depth
#         # )
#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()



#     def forward(self, init_disps, input_features, inv_K, K):
#         self.outputs = {}
#         # decoder
#         x = input_features[-1]
#         # 预测高斯参数的特征输入
#         gs_input_features = {}
#         # 预测初始深度图
#         for i in range(4, -1, -1): # up-conv 每个尺度上都进行两次卷积，并上采样
#             x = self.convs[("upconv", i, 0)](x)
#             x = [upsample(x)] # 反卷积可能造成空洞这里采用最邻近插值
#             if self.use_skips and i > 0:
#                 x += [input_features[i - 1]]
#             x = torch.cat(x, 1)
#             x = self.convs[("upconv", i, 1)](x)
#             if i in self.scales: # 在{0，1，2，3}尺度上都进行一次卷积生成视差图来计算loss
#                 gs_input_features[i] = x
#                 self.outputs[("init_disp", i)] = self.sigmoid(self.convs[("init_dispconv", i)](x))

#         # 预测leveraged深度图
#         for scale in range(4):
#             if self.use_gs:
#                 leveraged_feature = self.convs[("gs_leverage", scale)](
#                     gs_input_features[scale], self.outputs[("init_disp", scale)], inv_K[scale], K[scale])
#                 self.outputs[("disp", scale)] = self.sigmoid(self.convs[("dispconv", scale)](leveraged_feature))
#             else: # 不使用高斯
#                 self.outputs[("disp", scale)] = self.outputs[("init_disp", scale)]
            
#         return self.outputs

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

# Monodepth2相似逻辑的深度解码器，用于从leveraged_feature得到最终的深度图
class GSDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(GSDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        self.num_ch_enc = num_ch_enc  # np.array([16, 32, 64, 128])

        self.num_ch_dec = np.array([16, 32, 64, 128])
        self.use_skips = use_skips

        self.convs = OrderedDict()
        #decoder
        for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 3 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = num_ch_out
            if self.use_skips and i != 3:
                num_ch_in = num_ch_in + self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
        for s in range(4):
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)


        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        # 预测初始深度图
        for i in range(3, -1, -1): # up-conv 每个尺度上都进行两次卷积，并上采样
            x = self.convs[("upconv", i, 0)](x)
            # 反卷积可能造成空洞这里采用最邻近插值
            x = [upsample(x)] if i!=3 else [x]
            if self.use_skips and i != 3:
                x += [input_features[i]]
            
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales: # 在{0，1，2，3}尺度上都进行一次卷积生成视差图来计算loss
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

            
        return self.outputs