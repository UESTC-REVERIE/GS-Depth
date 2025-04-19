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
        
        self.num_depth_bins =  96
        # self.num_depth_bins =  0
        self.matching_height = self.height // (2 ** self.gs_scale)
        self.matching_width = self.width // (2 ** self.gs_scale)
        
        self.convs = OrderedDict()
        self.backprojector = OrderedDict()
        self.rasterizer = OrderedDict()
        # total_ch_in = sum(self.num_ch_in) + num_ch_concat + self.num_depth_bins
        # 融合lookup_features
        total_ch_in = sum(self.num_ch_in) * 2 + num_ch_concat + self.num_depth_bins
        
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
        # setup cost volume 
        self.feature_backprojector = BackprojectDepth_removedxy(
            batch_size=self.num_depth_bins,
            height=self.matching_height,
            width=self.matching_width)
        self.feature_projector = Project3D_removedxy(
            batch_size=self.num_depth_bins,
            height=self.matching_height,
            width=self.matching_width)
        self.feature_backprojector.to("cuda")
        self.feature_projector.to("cuda")
        self.compute_depth_bins(self.min_depth, self.max_depth)
        
        for i in range(4):
            # Up-Rasterization 4, 8, 16, 32
            scale = 2 ** ( i + 2 )
            self.rasterizer[i] = Rasterize_Gaussian_Feature_FiT3D_v2(
                image_height = self.height // scale,
                image_width = self.width // scale,
                min_depth = self.min_depth,
                max_depth = self.max_depth)
            self.rasterizer[i].to("cuda")
            # _num_ch_in = leveraged_feat_ch * self.gs_num_pixel + self.num_ch_in[i] + 1 + 3
            if i == 0:
                _num_ch_in = leveraged_feat_ch * self.gs_num_pixel + self.num_ch_in[i] + 1 + 3 + self.num_depth_bins
            else:
                _num_ch_in = leveraged_feat_ch * self.gs_num_pixel + self.num_ch_in[i] + 1 + 3
                
            self.convs[("gs_feature_resume_conv", i)] = nn.Sequential(
                SEBlock(_num_ch_in), # 添加SE注意力机制，选择融合特征中更重要的特征
                ConvBlock(_num_ch_in, self.num_ch_out[i])
            )
        self.gs_leverage = nn.ModuleList(list(self.convs.values()))

    def forward(self, init_features, colors, init_disps, inv_K, K, T=None, lookup_features=None, lookup_T=None):
        init_features = init_features[-4:]
        lookup_features = lookup_features[-4:] if lookup_features is not None else None
        leveraged_features = []
        # init_disps: 1/4~1/32
        # colors: 1~1/32
        disp = init_disps[0]
        bs = disp.shape[0]
        # h, w = disp[2:]
        h = self.height // (2 ** self.gs_scale)
        w = self.width // (2 ** self.gs_scale)
        disp = F.interpolate(disp, [h, w], mode="bilinear", align_corners=False)
        _ , depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        depth = depth.clamp(self.min_depth, self.max_depth)
        
        features = []
        gs_features = {}
        # 基本特征融合
        # TODO 是否融合lookup_features?
        for i in range(4):
            features = features + [F.interpolate(init_features[i], [h, w], mode="bilinear", align_corners=False)]
            features = features + [F.interpolate(lookup_features[i], [h, w], mode="bilinear", align_corners=False)]
        
        # init_disp融合
        for i in range(4):
            features = features + [F.interpolate(init_disps[i], [h, w], mode="bilinear", align_corners=False)]
        # color融合
        features = features + [colors[self.gs_scale]]

        # 构建cost volume与其它features融合，输入高斯模块
        if lookup_features is not None: # 有可用的multi-frames
            with torch.no_grad():
                cost_volume, missing_mask = \
                    self.match_features(init_features[0], lookup_features[0], lookup_T, K[2], inv_K[2])
                confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                            (1 - missing_mask.detach()))
            # mask the cost volume based on the confidence
            cost_volume *= confidence_mask.unsqueeze(1)
            # cost volume 融合
            features = features +  [cost_volume]
        
        feature = torch.cat(features, 1)
        # print("fused feature shape is ", feature.shape)
        position, rotation, scale, opacity, gs_feature = (
            list(range(self.gs_num_pixel)) for _ in range(5)
        )
        
        
        for i in range(self.gs_num_pixel):
            depth = [depth]
            if i!=0: # 每个像素预测多个高斯时，添加深度偏置，激励高斯填补被遮挡的非表面部分(following flash3D)
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
            # 1/4~1/32的高斯特征
            gs_features[(0, i)] = torch.cat(gs_feat_stack, 1)
            if not T is None: # 传入已知视角变化时渲染其他视角高斯特征
                for frame_id, transformation in T.items(): 
                    # TODO 这里计算的是1/4~1/32的高斯特征用于计算损失，考虑是否改到全分辨率
                    gs_features[(frame_id, i)] = torch.cat(self.rasterize_per_pixel(position, rotation, scale, opacity, gs_feature, K, i, transformation), 1)
            # TODO 融合cost volume输入最终的深度回归解码器
            if i == 0:
                fused_features = gs_feat_stack + [init_features[i]] + [colors[i+2]] + [init_disps[i]] + [cost_volume] # 加入cost volume
            else:    
                fused_features = gs_feat_stack + [init_features[i]] + [colors[i+2]] + [init_disps[i]] # 融合其他特征
            # fused_features = gs_feat_stack + [init_features[i]] + [colors[i+2]] + [init_disps[i]]
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
            SEBlock(in_ch), # 添加SE避免cat的不同特征数值差异过大
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
    
    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask
    def compute_depth_bins(self, min_depth_bin, max_depth_bin):

        self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        self.warp_depths = self.warp_depths.cuda()
            
    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it
        
        ONLY 1/4 Resolution features are used for cost volume computation
        following ManyDepth
        """
        # current_feats: 12 x C x H x W
        # lookup_feats: 12 x C x H x W
        # relative_poses: 12 x 4 x 4
        # K: 12 x 4 x 4
        # invK: 12 x 4 x 4
        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence
        
        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            lookup_feat = lookup_feats[batch_idx:batch_idx + 1] # 1 x C x H x W
            lookup_pose = relative_poses[batch_idx:batch_idx + 1]
            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            
            world_points = self.feature_backprojector(self.warp_depths, _invK)


            # ignore missing images
            if lookup_pose.sum() == 0:
                continue

            lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
            pix_locs = self.feature_projector(world_points, _K, lookup_pose)
            warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                    align_corners=True)

            # mask values landing outside the image (and near the border)
            # we want to ignore edge pixels of the lookup images and the current image
            # because of zero padding in ResNet
            # Masking of ref image border
            x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                self.matching_width - 1)  # convert from (-1, 1) to pixel values
            y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

            edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                        (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
            edge_mask = edge_mask.float()

            # masking of current image
            current_mask = torch.zeros_like(edge_mask)
            current_mask[:, 2:-2, 2:-2] = 1.0
            edge_mask = edge_mask * current_mask

            diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(
                1) * edge_mask

            # integrate into cost volume
            cost_volume = cost_volume + diffs

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            # if self.set_missing_to_max:
            cost_volume = cost_volume * (1 - missing_val_mask) + \
                cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks
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
