from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class GaussianFeatureLeverage(nn.Module):

    def __init__(self, num_ch_in, scales, height=192, width=640, leveraged_feat_ch=64, min_depth=0.1, max_depth=100.0, multi_scale_gs=True):
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
        """
        super().__init__()
        self.num_ch_in = num_ch_in # np.array([18, 36, 72, 144])
        self.leveraged_feat_ch = leveraged_feat_ch # gs光栅化后统一的特征维度
        self.num_ch_out = np.array([64, 18, 36, 72, 144])
        self.scales = scales
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self.convs = OrderedDict()
        self.backprojector = OrderedDict()
        self.rasterizer = OrderedDict()
        # if not multi_scale_gs:
        #     # TODO 特征融合
            
        #     self.convs["gs_rotation_conv"] = self.create_gaussian_head(num_ch_in[scale], 4)
        #     ...
        #     self.backprojector[0] = BackprojectDepth_PointCloud(
        #         height = self.height,
        #         width = self.width
        #     )
        #     # Rasterization
        #     for scale in self.scales:
        #         self.rasterizer[scale] = Rasterize_Gaussian_Feature_FiT3D_v1(
        #             image_height = self.height // ( 2**scale ),
        #             image_width = self.width // ( 2**scale ),
        #             min_depth = self.min_depth,
        #             max_depth = self.max_depth) 
        #         self.convs[("gs_feature_resume_conv", scale)] = ConvBlock(leveraged_feat_ch, self.num_ch_out[scale])
            
            
        for i in range(4):
            # 独立的高斯参数预测头
            self.convs[("gs_rotation_conv", i)] = self.create_gaussian_head(num_ch_in[i], 4)
            self.convs[("gs_scale_conv", i)] = self.create_gaussian_head(num_ch_in[i], 3) 
            self.convs[("gs_opacity_conv", i)] = self.create_gaussian_head(num_ch_in[i], 1)
            # self.feature_leve_convs = self.create_gaussian_head(feat_ch_in, leveraged_feat_ch, expand_ratio=1)
            self.convs[("gs_feature_leve_conv", i)] = nn.Sequential(
                ConvBlock(num_ch_in[i], num_ch_in[i]),
                ConvBlock(num_ch_in[i], leveraged_feat_ch),
            )
            scale = 2 ** ( i + 2 ) # 4, 8, 16, 32
            # 反投影得到高斯中心
            self.backprojector[i] = BackprojectDepth_PointCloud(
                height = self.height // scale,
                width = self.width // scale
            )
            self.backprojector[i].to("cuda")
            # Up-Rasterization 2, 4, 8, 16
            self.rasterizer[i] = Rasterize_Gaussian_Feature_FiT3D_v1(
                image_height = self.height // ( scale // 2 ),
                image_width = self.width // ( scale // 2 ),
                min_depth = self.min_depth,
                max_depth = self.max_depth)
            self.rasterizer[i].to("cuda")
            # 将特征恢复到指定维度 64, 18, 36, 72
            self.convs[("gs_feature_resume_conv", i)] = ConvBlock(leveraged_feat_ch, self.num_ch_out[i])
        # 保留1/32分辨率的特征，channels=144
        self.convs[("gs_feature_resume_conv", 4)] = ConvBlock(self.num_ch_in[-1], self.num_ch_out[-1])
        
        self.gs_leverage = nn.ModuleList(list(self.convs.values()))

    def forward(self, init_features, init_disps, inv_K, K):
        leveraged_features = []
        # TODO 测试统一的一个高斯表达，用不同的rasterization得到多分辨率    
        for i in range(4):
            init_feature = init_features[i]
            init_disp = init_disps[i]
            bs = init_feature.shape[0]
            
            # 1. Position calculation
            _, depth_init = disp_to_depth(init_disp, self.min_depth, self.max_depth)
            depth_init = depth_init.clamp(self.min_depth, self.max_depth)
            position = self.backprojector[i](depth_init, inv_K[i])
            position = position.permute(0,2,1).contiguous()
            position = position.view(bs, -1, 3)
            
            # 2. Rotation parameters
            rotation = F.normalize(self.convs[("gs_rotation_conv", i)](init_feature), dim=1, p=2)
            rotation = rotation.permute(0,2,3,1).contiguous()
            rotation = rotation.view(bs, -1, 4)
            
            # 3. Scale parameters
            s = torch.abs(self.convs[("gs_scale_conv", i)](init_feature))
            s = s.permute(0,2,3,1).contiguous()
            s = s.view(bs, -1, 3)
            
            # 4. Opacity parameters
            opacity = torch.sigmoid(self.convs[("gs_opacity_conv", i)](init_feature))
            opacity = opacity.permute(0,2,3,1).contiguous()
            opacity = opacity.view(bs, -1, 1)
            
            # 5. Feature generation
            gs_feature = self.convs[("gs_feature_leve_conv", i)](init_feature)
            gs_feature = gs_feature.permute(0,2,3,1).contiguous()
            gs_feature = gs_feature.view(bs, -1, gs_feature.shape[-1])
            
            # 6. Rasterization
            leveraged_feature = self.rasterizer[i](position, rotation, s, opacity, gs_feature, inv_K[i+1], K[i+1])
            
            leveraged_features.append(self.convs[("gs_feature_resume_conv", i)](leveraged_feature))
        leveraged_features.append(self.convs[("gs_feature_resume_conv", 4)](init_features[3]))
        return leveraged_features
    
    def create_gaussian_head(self, in_ch: int, out_ch: int, expand_ratio=2) -> nn.Sequential:
        """
        高斯参数预测头工厂函数
        :param in_ch: 输入通道数
        :param out_ch: 目标参数维度 
        :param expand_ratio: 中间层通道扩展倍数
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * expand_ratio, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(out_ch * expand_ratio, out_ch, 3, 1, 1, padding_mode='replicate')
            
            # ConvBlock(in_ch, in_ch)
            # Conv3x3(in_ch, out_ch)
        )