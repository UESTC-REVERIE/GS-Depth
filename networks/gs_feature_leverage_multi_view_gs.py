from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from networks.init_depth_decode_resnet import InitDepthDecoder

# 根据统一的分辨率高斯光栅化不同的分辨率特征，
# 使用se注意力机制，
# 使用一个head多通道预测高斯参数，同时预测position-offset
class GaussianFeatureLeverage(nn.Module):

    def __init__(self, num_ch_in, scales, height=192, width=640, leveraged_feat_ch=64, min_depth=0.1, max_depth=100.0, num_ch_concat=0, gs_scale=0, split_dimensions = [3,1,3,4], gs_num_pixel=2):
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
            split_dimensions: 高斯参数预测头的通道划分[offset_s, opacity_s, scale_s, rotation_s]
        """
        super().__init__()
        self.num_ch_in = num_ch_in[-4:] # np.array([18, 36, 72, 144])
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
        self.gs_num_pixel =gs_num_pixel
        
        self.convs = OrderedDict()
        self.backprojector = OrderedDict()
        self.rasterizer = OrderedDict()
        total_ch_in = sum(self.num_ch_in) + num_ch_concat
        
        # print(f"using option: predict {self.gs_num_pixel} gs per pixel")
        # rotation scale opacity depth-offset
        for i in range(gs_num_pixel):
            self.convs[("gs_depth_bias_conv", i)] = InitDepthDecoder(
                num_ch_enc=self.num_ch_in,
                scales=self.scales
            ) if i != 0 else None
            # print(f"total_ch_in is {total_ch_in}")
            self.convs[("gs_head_conv", i)] = self.create_gaussian_head(total_ch_in, sum(self.split_dimensions))
            self.convs[("gs_feature_leve_conv", i)] = nn.Sequential(
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
            self.rasterizer[i] = Rasterize_Gaussian_Feature_FiT3D_v2(
                image_height = self.height // scale,
                image_width = self.width // scale,
                min_depth = self.min_depth,
                max_depth = self.max_depth)
            self.rasterizer[i].to("cuda")
            _num_ch_in = leveraged_feat_ch * self.gs_num_pixel + self.num_ch_in[i] + 1 + 3
            self.convs[("gs_feature_resume_conv", i)] = nn.Sequential(
                # SEBlock(_num_ch_in), # 添加SE注意力机制，选择融合特征中更重要的特征
                ConvBlock(_num_ch_in, self.num_ch_out[i])
            )
        self.gs_leverage = nn.ModuleList(list(self.convs.values()))

    def forward(self, init_features, colors, init_disps, inv_K, K, T=None):
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
        gs_features = {}
        # 基本特征融合
        for i in range(4):
            features = features + [F.interpolate(init_features[i], [h, w], mode="bilinear", align_corners=False)]
        
        # init_disp融合
        for i in range(4):
            features = features + [F.interpolate(init_disps[i], [h, w], mode="bilinear", align_corners=False)]
        # color融合
        features = features + [colors[self.gs_scale]]

        feature = torch.cat(features, 1)
        
        position, rotation, scale, opacity, gs_feature = (
            list(range(self.gs_num_pixel)) for _ in range(5)
        )
        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        depth = depth.clamp(self.min_depth, self.max_depth)
        
        for i in range(self.gs_num_pixel):
            depth = [depth]
            if i!=0:
                _, _, depth_bias_o = self.convs[("gs_depth_bias_conv", i)](init_features)
                depth_bias = torch.exp(torch.clamp(depth_bias_o[0], min=-10.0, max=6.0))
                depth_bias = F.interpolate(disp, [h, w], mode="bilinear", align_corners=False)
                depth += [depth_bias]
            depth = torch.cat(depth, 1)
            depth = torch.sum(depth, 1, keepdim=True)
            gs_params = self.convs[("gs_head_conv", i)](feature)
            position[i], rotation[i], scale[i], opacity[i] = self.gs_generator(depth,bs,inv_K[self.gs_scale],gs_params)
            
            gs_feature[i] = self.convs[("gs_feature_leve_conv", i)](feature)
            gs_feature[i] = gs_feature[i].permute(0,2,3,1).contiguous()
            gs_feature[i] = gs_feature[i].view(bs, -1, gs_feature[i].shape[-1])
        
        for i in range(4): # different scales feature
            gs_feat_stack = self.rasterize_per_pixel(position, rotation, scale, opacity, gs_feature, K, i)
            if not T is None:
                gs_features[(0, i)] = torch.cat(gs_feat_stack, 1)
                for frame_id, transformation in T.items(): # 渲染其他视角高斯特征
                    # TODO 这里计算的是1/4~1/32的高斯特征用于计算损失，考虑是否改到全分辨率
                    gs_features[(frame_id, i)] = torch.cat(self.rasterize_per_pixel(position, rotation, scale, opacity, gs_feature, K, i, transformation), 1)
            fused_features = gs_feat_stack + [init_features[i]] + [colors[i+2]] + [init_disps[i]] # 融合其他特征
            fused_features = torch.cat(fused_features, 1)
            leveraged_features.append(self.convs[("gs_feature_resume_conv", i)](fused_features))
            
        return leveraged_features, gs_features
    
    def gs_generator(self, depth, bs, inv_k, gs_params):
        
        offset_s, opacity_s, scale_s, rotation_s = gs_params.split(self.split_dimensions, dim=1)

        # 1. Position calculation
        _position = self.backprojector(depth, inv_k)
        offset_s = offset_s.view(bs, 3, -1) # [B,3,H*W]
        _position = _position + offset_s
        _position = _position.permute(0,2,1).contiguous()
        _position = _position.view(bs, -1, 3)

        # 2. Rotation parameters
        _rotation = F.normalize(rotation_s, dim=1, p=2)
        _rotation = _rotation.permute(0,2,3,1).contiguous()
        _rotation = _rotation.view(bs, -1, 4)
        
        # 3. Scale parameters
        # 改为exp \times lambda
        # s = torch.abs(self.convs["gs_scale_conv"](feature))
        _scale = torch.exp(scale_s) * 0.01
        _scale = _scale.permute(0,2,3,1).contiguous()
        _scale = _scale.view(bs, -1, 3)
        
        # 4. Opacity parameters
        _opacity = torch.sigmoid(opacity_s)
        _opacity = _opacity.permute(0,2,3,1).contiguous()
        _opacity = _opacity.view(bs, -1, 1)
        
        return _position, _rotation, _scale, _opacity
    def create_gaussian_head(self, in_ch: int, out_ch: int, expand_ratio=2) -> nn.Sequential:
        """
        高斯参数预测头工厂函数
        :param in_ch: 输入通道数
        :param out_ch: 目标参数维度 
        :param expand_ratio: 中间层通道扩展倍数
        """
        return nn.Sequential(
            # SEBlock(in_ch), # 添加SE避免cat的不同特征数值差异过大
            nn.Conv2d(in_ch, out_ch * expand_ratio, 3, 1, 1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv2d(out_ch * expand_ratio, out_ch, 3, 1, 1, padding_mode='replicate')
            
            # ConvBlock(in_ch, in_ch)
            # Conv3x3(in_ch, out_ch)
        )
    def rasterize_per_pixel(self, position, rotation, scale, opacity, gs_feature, K, index, T=None):
        gs_feat_stack = []
        for gs_index in range(self.gs_num_pixel): # multi-gs per pixel
            leveraged_feature = self.rasterizer[index](position[gs_index], rotation[gs_index], scale[gs_index], opacity[gs_index], gs_feature[gs_index],  K[index+2], T)
            gs_feat_stack += [leveraged_feature]
        return gs_feat_stack
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
