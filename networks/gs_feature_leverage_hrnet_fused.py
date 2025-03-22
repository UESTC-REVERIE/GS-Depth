from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

# 根据统一的分辨率高斯光栅化不同的分辨率特征，
# 使用se注意力机制，
# 使用一个head多通道预测高斯参数，同时预测position-offset
class GaussianFeatureLeverage(nn.Module):

    def __init__(self, num_ch_in, scales, height=192, width=640, leveraged_feat_ch=64, min_depth=0.1, max_depth=100.0, num_ch_concat=0, gs_scale=0, split_dimensions = [3,1,3,4]):
        """gaussian feature leverage

        Args:
            num_ch_in (_type_): 输入特征的通道数
            scales (_type_): 输出特征的多个尺度
            height (int, optional): 未缩放的原始高度. Defaults to 192.
            width (int, optional): 未缩放的原始宽度. Defaults to 640.
            leveraged_feat_ch (int, optional): 存储在高斯基元中的特征通道数. Defaults to 64.
            min_depth (float, optional): min depth. Defaults to 0.1.
            max_depth (float, optional): max depth. Defaults to 100.0.
            multi_scale_gs (bool, optional): 每个尺度独立预测高斯或只预测一个统一的高斯表达，默认每个尺度独自预测高斯. Defaults to True.
            split_dimensions: 高斯参数预测头的通道划分
        """
        super().__init__()
        self.num_ch_in = num_ch_in # np.array([18, 36, 72, 144])
        self.leveraged_feat_ch = leveraged_feat_ch # gs光栅化后统一的特征维度
        self.num_ch_out = np.array([18, 36, 72, 144])
        self.scales = scales
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_ch_concat = num_ch_concat
        self.gs_scale = gs_scale
        self.split_dimensions = split_dimensions
        
        self.convs = OrderedDict()
        self.backprojector = OrderedDict()
        self.rasterizer = OrderedDict()

        total_ch_in = sum(num_ch_in) + num_ch_concat
        # print(f"total channels = {total_ch_in}")
        # self.convs["gs_rotation_conv"] = self.create_gaussian_head(total_ch_in, 4)
        # self.convs["gs_scale_conv"] = self.create_gaussian_head(total_ch_in, 3)
        # self.convs["gs_opacity_conv"] = self.create_gaussian_head(total_ch_in, 1)
        
        # rotation scale opacity depth-offset
        self.convs["gs_conv_head"] = self.create_gaussian_head(total_ch_in, sum(self.split_dimensions))
        
        self.convs["gs_feature_leve_conv"] = nn.Sequential(
            ConvBlock(total_ch_in, total_ch_in),
            ConvBlock(total_ch_in, leveraged_feat_ch)
        )
        # 反投影得到高斯中心
        self.backprojector = BackprojectDepth_PointCloud(
            height = self.height // (2 ** self.gs_scale),
            width = self.width // (2 ** self.gs_scale)
        )
        self.backprojector.to("cuda")
        for i in range(4):
            # Up-Rasterization 4, 8, 16, 32
            scale = 2 ** ( i + 2 )
            self.rasterizer[i] = Rasterize_Gaussian_Feature_FiT3D_v1(
                image_height = self.height // scale,
                image_width = self.width // scale,
                min_depth = self.min_depth,
                max_depth = self.max_depth)
            self.rasterizer[i].to("cuda")
            _num_ch_in = leveraged_feat_ch + num_ch_in[i] + 1 + 3
            self.convs[("gs_feature_resume_conv", i)] = nn.Sequential(
                SEBlock(_num_ch_in), # 添加SE注意力机制      
                ConvBlock(_num_ch_in, self.num_ch_out[i])
            )
        self.gs_leverage = nn.ModuleList(list(self.convs.values()))

    def forward(self, init_features, colors, init_disps, inv_K, K):
        init_features = init_features[-4:]
        leveraged_features = []
        # init_disps: 1/4~1/32
        # colors: 1~1/32
        disp = init_disps[0]
        bs = disp.shape[0]
        # h, w = disp[2:]
        h = self.height // (2 ** self.gs_scale)
        w = self.width // (2 ** self.gs_scale)
        disp = F.interpolate(disp, [h, w], mode="bilinear", align_corners=False)
        features = []
        # 基本特征融合
        for i in range(4):
            features = features + [F.interpolate(init_features[i], [h, w], mode="bilinear", align_corners=False)]
        
        # init_disp融合
        for i in range(4):
            features = features + [F.interpolate(init_disps[i], [h, w], mode="bilinear", align_corners=False)]
        # color融合
        features = features + [colors[self.gs_scale]]

        feature = torch.cat(features, 1)
        
        gs_params = self.convs["gs_conv_head"](feature)
        offset_s, opacity_s, scale_s, rotation_s = gs_params.split(self.split_dimensions, dim=1)
        # print(f"{feature.shape}")
        # 1. Position calculation
        # TODO 预测Multi-Gaussians per pixel for modeling occluded surfaces
        _, depth_init = disp_to_depth(disp, self.min_depth, self.max_depth)
        depth_init = depth_init.clamp(self.min_depth, self.max_depth)
        position = self.backprojector(depth_init, inv_K[self.gs_scale])
        offset_s = offset_s.view(bs, 3, -1) # [B,3,H*W]
        position = position + offset_s
        position = position.permute(0,2,1).contiguous()
        position = position.view(bs, -1, 3)
        
        # 2. Rotation parameters
        rotation = F.normalize(rotation_s, dim=1, p=2)
        rotation = rotation.permute(0,2,3,1).contiguous()
        rotation = rotation.view(bs, -1, 4)
        
        # 3. Scale parameters
        # 改为exp \times lambda
        # s = torch.abs(self.convs["gs_scale_conv"](feature))
        scale = torch.exp(scale_s) * 0.01
        scale = scale.permute(0,2,3,1).contiguous()
        scale = scale.view(bs, -1, 3)
        
        # 4. Opacity parameters
        opacity = torch.sigmoid(opacity_s)
        opacity = opacity.permute(0,2,3,1).contiguous()
        opacity = opacity.view(bs, -1, 1)
        
        # 5. Feature generation
        gs_feature = self.convs["gs_feature_leve_conv"](feature)
        gs_feature = gs_feature.permute(0,2,3,1).contiguous()
        gs_feature = gs_feature.view(bs, -1, gs_feature.shape[-1])
        
        
        for i in range(4):
            # 6. Rasterization
            leveraged_feature = self.rasterizer[i](position, rotation, scale, opacity, gs_feature, inv_K[i+2], K[i+2])
            # TODO cat init_disp and color ?
            fused_feature = [leveraged_feature]
            fused_feature = fused_feature + [init_features[i]] +[colors[i+2]] +[init_disps[i]]
            fused_feature = torch.cat(fused_feature, 1)
            leveraged_features.append(self.convs[("gs_feature_resume_conv", i)](fused_feature))
            
        return leveraged_features
    
    def create_gaussian_head(self, in_ch: int, out_ch: int, expand_ratio=2) -> nn.Sequential:
        """
        高斯参数预测头工厂函数
        :param in_ch: 输入通道数
        :param out_ch: 目标参数维度 
        :param expand_ratio: 中间层通道扩展倍数
        """
        return nn.Sequential(
            SEBlock(in_ch), # 添加SE避免cat的不同特征数值差异过大
            nn.Conv2d(in_ch, out_ch * expand_ratio, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(out_ch * expand_ratio, out_ch, 3, 1, 1, padding_mode='replicate')
            
            # ConvBlock(in_ch, in_ch)
            # Conv3x3(in_ch, out_ch)
        )
        
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
