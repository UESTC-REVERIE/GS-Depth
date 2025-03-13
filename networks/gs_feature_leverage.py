import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class GaussianFeatureLeverage(nn.Module):
    def __init__(self, feat_ch_in : int, feat_ch_out : int, scale, height=192, width=640, leveraged_feat_ch=64, min_depth=0.1, max_depth=100.0):
        super().__init__()
        """
        高斯参数生成器
        :param feat_ch_in: 输入特征通道数
        :param feat_ch_out: 输出特征通道数
        :param leveraged_feat_ch: 存放在每个高斯基元中的特征通道数
        :param scale: 缩放尺度
        :param height: 未缩放的原始图像高度
        :param width: 未缩放的原始图像宽度
        """
        self.feat_ch_in = feat_ch_in
        self.feat_ch_out = feat_ch_out
        self.leveraged_feat_ch = leveraged_feat_ch
        self.scale = scale
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # 独立的高斯参数预测头
        self.rotation_convs = self.create_gaussian_head(feat_ch_in, 4)
        self.scale_convs = self.create_gaussian_head(feat_ch_in, 3) 
        self.opacity_convs = self.create_gaussian_head(feat_ch_in, 1)
        # self.feature_leve_convs = self.create_gaussian_head(feat_ch_in, leveraged_feat_ch, expand_ratio=1)
        self.feature_leve_convs = nn.Sequential(
            ConvBlock(feat_ch_in, feat_ch_in),
            ConvBlock(feat_ch_in, leveraged_feat_ch),
        )
        # 反投影得到高斯中心
        self.backprojector = BackprojectDepth_PointCloud(height=self.height//(2**scale), width=self.width//(2**scale))
        
        # Feature head
        # self.feature_leve_convs = nn.Sequential(
        #     ConvBlock(feat_ch_in, feat_ch_in),
        #     ConvBlock(feat_ch_in, leveraged_feat_ch),
        # )
        
        # Rasterization
        self.rasterizer = Rasterize_Gaussian_Feature_FiT3D_v1(image_height=self.height//(2**scale), image_width=self.width//(2**scale),min_depth=self.min_depth, max_depth=self.max_depth)
        
        # 将特征恢复到指定维度
        self.feature_resume_conv = ConvBlock(leveraged_feat_ch, feat_ch_out)
        
        

    def forward(self, init_feature, disp_init, inv_K, K):
        
        bs = init_feature.shape[0]
        
        # 1. Position calculation
        _, depth_init = disp_to_depth(disp_init, self.min_depth, self.max_depth)
        depth_init = depth_init.clamp(self.min_depth, self.max_depth)
        position = self.backprojector(depth_init, inv_K)
        position = position.permute(0,2,1).contiguous()
        position = position.view(bs, -1, 3)
        
        # 2. Rotation parameters
        rotation = F.normalize(self.rotation_convs(init_feature), dim=1, p=2)
        rotation = rotation.permute(0,2,3,1).contiguous()
        rotation = rotation.view(bs, -1, 4)
        
        # 3. Scale parameters
        scale = torch.abs(self.scale_convs(init_feature))
        scale = scale.permute(0,2,3,1).contiguous()
        scale = scale.view(bs, -1, 3)
        
        # 4. Opacity parameters
        opacity = torch.sigmoid(self.opacity_convs(init_feature))
        opacity = opacity.permute(0,2,3,1).contiguous()
        opacity = opacity.view(bs, -1, 1)
        
        # 5. Feature generation
        gs_feature = self.feature_leve_convs(init_feature)
        gs_feature = gs_feature.permute(0,2,3,1).contiguous()
        gs_feature = gs_feature.view(bs, -1, gs_feature.shape[-1])
        
        # 6. Rasterization
        leveraged_feature = self.rasterizer(position, rotation, scale, opacity, gs_feature, inv_K, K)
        
        leveraged_feature = self.feature_resume_conv(leveraged_feature)
        
        return leveraged_feature
    
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