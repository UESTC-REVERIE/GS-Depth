# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function


import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from mmcv.cnn import constant_init, kaiming_init
from torch.autograd import Variable
from copy import deepcopy
# from skimage.segmentation import all_felzenszwalb as felz_seg
# from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
# from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from diff_feature_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

eval_metrics = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1', 'd2', 'd3']



# class Rasterize_Gaussian_Feature_PGSR(nn.Module):
#     def __init__(self, image_height, image_width, scaling_modifier=1.0):
#         super(Rasterize_Gaussian_Feature_PGSR, self).__init__()
#         # Set up rasterization configuration
#         self.image_height = image_height
#         self.image_width = image_width
#         self.scaling_modifier = scaling_modifier


#     def forward(self, position, rotation, scale, opacity, feature, inv_K, K):
#         bs = position.shape[0]
#         gs_features = torch.zeros(bs, feature.shape[2] - 5, self.image_height, self.image_width).to(device=feature.device)
#         # print(self.image_height, self.image_width)
#         # gs_features = torch.zeros(bs, 256, self.image_height, self.image_width).to(device=feature.device)
#         # Rasterize visible Gaussians to image, obtain their radii (on screen).
#         for index in range(bs):
#             tanfovx = math.tan(K[index, 0, 0] * 0.5)
#             tanfovy = math.tan(K[index, 1, 1] * 0.5)

#             world_view_transform = torch.zeros(4, 4).to(device=feature.device)
#             world_view_transform[0, 0] = 1
#             world_view_transform[1, 1] = 1
#             world_view_transform[2, 2] = 1
#             world_view_transform[3, 3] = 1
#             world_view_transform = world_view_transform.transpose(0, 1)
#             world_view_transform = world_view_transform.contiguous()

#             full_proj_transform = K[index]

#             # camera_center = world_view_transform.inverse()[3, :3]
#             camera_center = torch.zeros(1, 3).to(device=feature.device)
#             bg_color = [1, 1, 1]
#             background = torch.tensor(bg_color, dtype=torch.float32, device=feature.device) 
#             raster_settings = PlaneGaussianRasterizationSettings(
#                 image_height=self.image_height,
#                 image_width=self.image_width,
#                 tanfovx=tanfovx,
#                 tanfovy=tanfovy,
#                 bg=background,
#                 scale_modifier=self.scaling_modifier,
#                 viewmatrix=world_view_transform,
#                 projmatrix=full_proj_transform,
#                 sh_degree=1,
#                 campos=camera_center,
#                 prefiltered=False,
#                 render_geo=True,
#                 debug=False
#             )
#             rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)
#             means2D_per  = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=True, device=feature.device) + 0
#             means2D_abs_per  = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=True, device=feature.device) + 0
#             color_per = torch.ones(feature.shape[1], 3).to(device=feature.device)
#             position_per  = position[index]
#             semantic_feature_per = feature[index]
#             # semantic_feature_per = semantic_feature_per.unsqueeze(1)
#             opacity_per = opacity[index]
#             scale_per = scale[index]
#             rotation_per = rotation[index]
#             rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
#             means3D = position_per,
#             means2D = means2D_per,
#             means2D_abs = means2D_abs_per,
#             shs = None,
#             colors_precomp = color_per,
#             opacities = opacity_per,
#             scales = scale_per,
#             rotations = rotation_per,
#             all_map = semantic_feature_per,
#             cov3D_precomp = None)
#             # print(out_all_map.shape)
#             gs_features[index:index+1] = out_all_map[5:].unsqueeze(0)    
#         # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#         # They will be excluded from value updates used in the splitting criteria.
#         return gs_features.contiguous()

class Rasterize_Gaussian_Feature_FiT3D(nn.Module):
    def __init__(self, image_height, image_width, scaling_modifier=1.0):
        super(Rasterize_Gaussian_Feature_FiT3D, self).__init__()
        # Set up rasterization configuration
        # 定义光栅化后生成的2D特征图的高度和宽度
        self.image_height = image_height
        self.image_width = image_width
        self.scaling_modifier = scaling_modifier

    def forward(self, position, rotation, scale, opacity, feature, inv_K, K):
        bs = position.shape[0]
        gs_features = torch.zeros(bs, feature.shape[2], self.image_height, self.image_width).to(device=feature.device)
        # print(self.image_height, self.image_width)
        # gs_features = torch.zeros(bs, 256, self.image_height, self.image_width).to(device=feature.device)
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        for index in range(bs):
            tanfovx = math.tan(K[index, 0, 0] * 0.5)
            tanfovy = math.tan(K[index, 1, 1] * 0.5)

            world_view_transform = torch.zeros(4, 4).to(device=feature.device)
            world_view_transform[0, 0] = 1
            world_view_transform[1, 1] = 1
            world_view_transform[2, 2] = 1
            world_view_transform[3, 3] = 1
            world_view_transform = world_view_transform.transpose(0, 1)
            world_view_transform = world_view_transform.contiguous()

            full_proj_transform = K[index]

            # camera_center = world_view_transform.inverse()[3, :3]
            camera_center = torch.zeros(1, 3).to(device=feature.device)
            bg_color = [1, 1, 1]
            background = torch.tensor(bg_color, dtype=torch.float32, device=feature.device) 
            raster_settings = GaussianRasterizationSettings(
                image_height=self.image_height,
                image_width=self.image_width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=self.scaling_modifier,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=1,
                campos=camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            means2D_per = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=False, device=feature.device) + 0
            # means2D_abs_per  = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=True, device=feature.device) + 0
            color_per = torch.ones(feature.shape[1], 3).to(device=feature.device)
            position_per  = position[index]
            semantic_feature_per = feature[index]
            # semantic_feature_per = semantic_feature_per.unsqueeze(1)
            opacity_per = opacity[index]
            scale_per = scale[index]
            rotation_per = rotation[index]
            rendered_image, rendered_featmap, radii = rasterizer(
            means3D = position_per,
            means2D = means2D_per,
            shs = None,
            colors_precomp = color_per,
            opacities = opacity_per,
            scales = scale_per,
            rotations = rotation_per,
            sem = semantic_feature_per,
            cov3D_precomp = None)
            # print(out_all_map.shape)
            gs_features[index:index+1] = rendered_featmap.unsqueeze(0)    
            
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return gs_features.contiguous()

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Rasterize_Gaussian_Feature_FiT3D_v1(nn.Module):
    def __init__(self, image_height, image_width, min_depth=0.1, max_depth=100.0, scaling_modifier=1.0):
        super(Rasterize_Gaussian_Feature_FiT3D_v1, self).__init__()
        # Set up rasterization configuration
        self.image_height = image_height
        self.image_width = image_width
        self.scaling_modifier = scaling_modifier
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, position, rotation, scale, opacity, feature, inv_K, K):
        bs = position.shape[0]
        gs_features = torch.zeros(bs, feature.shape[2], self.image_height, self.image_width).to(device=feature.device)
        # print(self.image_height, self.image_width)
        # gs_features = torch.zeros(bs, 256, self.image_height, self.image_width).to(device=feature.device)
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        for index in range(bs):
            fx, fy = K[index, 0, 0], K[index, 1, 1]
            FoVx, FoVy = focal2fov(fx, self.image_width), focal2fov(fy, self.image_height)

            tanfovx = math.tan(FoVx* 0.5)
            tanfovy = math.tan(FoVy * 0.5)

            # 单位矩阵，相机坐标系和世界坐标系的变换矩阵，数据集中以camera_0为基准，故默认无变化
            world_view_transform = torch.zeros(4, 4).to(device=feature.device)
            world_view_transform[0, 0] = 1
            world_view_transform[1, 1] = 1
            world_view_transform[2, 2] = 1
            world_view_transform[3, 3] = 1
            world_view_transform = world_view_transform.transpose(0,1)

            full_proj_transform = getProjectionMatrix(znear=0.01, zfar=self.max_depth, fovX=FoVx, fovY=FoVy).transpose(0,1).to(device=feature.device)
            # camera_center = world_view_transform.inverse()[3, :3]
            camera_center = torch.zeros(1, 3).to(device=feature.device)
            bg_color = [1, 1, 1]
            background = torch.tensor(bg_color, dtype=torch.float32, device=feature.device) 
            raster_settings = GaussianRasterizationSettings(
                image_height=self.image_height,
                image_width=self.image_width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=self.scaling_modifier,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=1,
                campos=camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            means2D_per = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=False, device=feature.device) + 0
            # means2D_abs_per  = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=True, device=feature.device) + 0
            color_per = torch.ones(feature.shape[1], 3).to(device=feature.device)
            position_per  = position[index]
            semantic_feature_per = feature[index]
            # semantic_feature_per = semantic_feature_per.unsqueeze(1)
            opacity_per = opacity[index]
            scale_per = scale[index]
            rotation_per = rotation[index]
            rendered_image, rendered_featmap, radii = rasterizer(
                means3D = position_per,
                means2D = means2D_per,
                shs = None,
                colors_precomp = color_per,
                opacities = opacity_per,
                scales = scale_per,
                rotations = rotation_per,
                sem = semantic_feature_per,
                cov3D_precomp = None)
            # print(out_all_map.shape)
            gs_features[index:index+1] = rendered_featmap.unsqueeze(0)    
            
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return gs_features.contiguous()
    
class Rasterize_Gaussian_Feature_FiT3D_v2(nn.Module):
    def __init__(self, image_height, image_width, min_depth=0.1, max_depth=100.0, scaling_modifier=1.0):
        super(Rasterize_Gaussian_Feature_FiT3D_v2, self).__init__()
        # Set up rasterization configuration
        self.image_height = image_height
        self.image_width = image_width
        self.scaling_modifier = scaling_modifier
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, position, rotation, scale, opacity, feature, K, T=None):
        bs = position.shape[0]
        gs_features = torch.zeros(bs, feature.shape[2], self.image_height, self.image_width).to(device=feature.device)
        # print(self.image_height, self.image_width)
        # gs_features = torch.zeros(bs, 256, self.image_height, self.image_width).to(device=feature.device)
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        for index in range(bs):
            fx, fy = K[index, 0, 0], K[index, 1, 1]
            FoVx, FoVy = focal2fov(fx, self.image_width), focal2fov(fy, self.image_height)

            tanfovx = math.tan(FoVx* 0.5)
            tanfovy = math.tan(FoVy * 0.5)
            if T is None:
                world_view_transform = torch.zeros(4, 4).to(device=feature.device)
                world_view_transform[0, 0] = 1
                world_view_transform[1, 1] = 1
                world_view_transform[2, 2] = 1
                world_view_transform[3, 3] = 1
                world_view_transform = world_view_transform.transpose(0,1)
                camera_center = torch.zeros(1, 3).to(device=feature.device)
            else:
                # 转换为列优先存储
                world_view_transform = T[index].transpose(0,1).to(feature.device)
                camera_center = world_view_transform.inverse()[3, :3]
            
            full_proj_transform = getProjectionMatrix(znear=0.01, zfar=self.max_depth, fovX=FoVx, fovY=FoVy).transpose(0,1).to(device=feature.device)
            bg_color = [1, 1, 1]
            background = torch.tensor(bg_color, dtype=torch.float32, device=feature.device) 
            raster_settings = GaussianRasterizationSettings(
                image_height=self.image_height,
                image_width=self.image_width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=self.scaling_modifier,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=1,
                campos=camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            means2D_per = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=False, device=feature.device) + 0
            # means2D_abs_per  = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=True, device=feature.device) + 0
            color_per = torch.ones(feature.shape[1], 3).to(device=feature.device)
            position_per  = position[index]
            semantic_feature_per = feature[index]
            # semantic_feature_per = semantic_feature_per.unsqueeze(1)
            opacity_per = opacity[index]
            scale_per = scale[index]
            rotation_per = rotation[index]
            rendered_image, rendered_featmap, radii = rasterizer(
                means3D = position_per,
                means2D = means2D_per,
                shs = None,
                colors_precomp = color_per,
                opacities = opacity_per,
                scales = scale_per,
                rotations = rotation_per,
                sem = semantic_feature_per,
                cov3D_precomp = None)
            # print(out_all_map.shape)
            gs_features[index:index+1] = rendered_featmap.unsqueeze(0)    
            
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return gs_features.contiguous()

class Rasterize_Gaussian_Feature_FiT3D_batch(nn.Module):
    def __init__(self, image_height, image_width, scaling_modifier=1.0):
        super(Rasterize_Gaussian_Feature_FiT3D_batch, self).__init__()
        # Set up rasterization configuration
        self.image_height = image_height
        self.image_width = image_width
        self.scaling_modifier = scaling_modifier


    def forward(self, position, rotation, scale, opacity, feature, inv_K, K):
        bs, N, C = feature.shape
        position = position.view(-1, 3)
        rotation = rotation.view(-1, 4)
        scale = scale.view(-1, 3)
        opacity = opacity.view(-1, 1)
        feature = feature.view(-1, C)
        print(position.shape, rotation.shape, scale.shape, opacity.shape, feature.shape)
        # gs_features = torch.zeros(bs * feature.shape[2], self.image_height, self.image_width).to(device=feature.device)
        # print(self.image_height, self.image_width)
        # gs_features = torch.zeros(bs, 256, self.image_height, self.image_width).to(device=feature.device)
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        tanfovx = math.tan(K[0, 0, 0] * 0.5)
        tanfovy = math.tan(K[0, 1, 1] * 0.5)

        world_view_transform = torch.zeros(4, 4).to(device=feature.device)
        world_view_transform[0, 0] = 1
        world_view_transform[1, 1] = 1
        world_view_transform[2, 2] = 1
        world_view_transform[3, 3] = 1
        world_view_transform = world_view_transform.transpose(0, 1)
        world_view_transform = world_view_transform.contiguous()

        full_proj_transform = K[0]

        # camera_center = world_view_transform.inverse()[3, :3]
        camera_center = torch.zeros(1, 3).to(device=feature.device)
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device=feature.device) 
        raster_settings = GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=self.scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=1,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means2D_per  = torch.zeros_like(position, dtype=position.dtype, requires_grad=False, device=feature.device) + 0
        # means2D_abs_per  = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=True, device=feature.device) + 0
        color_per = torch.ones(feature.shape[0], 3).to(device=feature.device)
        position_per  = position
        semantic_feature_per = feature
        # semantic_feature_per = semantic_feature_per.unsqueeze(1)
        opacity_per = opacity
        scale_per = scale
        rotation_per = rotation
        rendered_image, rendered_featmap, radii = rasterizer(
        means3D = position_per,
        means2D = means2D_per,
        shs = None,
        colors_precomp = color_per,
        opacities = opacity_per,
        scales = scale_per,
        rotations = rotation_per,
        sem = semantic_feature_per,
        cov3D_precomp = None)

        print(rendered_featmap.shape)
            
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return rendered_featmap.reshape(bs, C, self.image_height, self.image_width)


# class Rasterize_Gaussian_Feature(nn.Module):
#     def __init__(self, image_height, image_width, scaling_modifier=1.0):
#         super(Rasterize_Gaussian_Feature, self).__init__()
#         # Set up rasterization configuration
#         self.image_height = image_height
#         self.image_width = image_width
#         self.scaling_modifier = scaling_modifier


#     def forward(self, position, rotation, scale, opacity, feature, inv_K, K):
#         bs = position.shape[0]
#         gs_features = torch.zeros(bs, feature.shape[2], self.image_height, self.image_width).to(device=feature.device)
#         # print(self.image_height, self.image_width)
#         # gs_features = torch.zeros(bs, 256, self.image_height, self.image_width).to(device=feature.device)
#         # Rasterize visible Gaussians to image, obtain their radii (on screen).
#         for index in range(bs):
#             tanfovx = math.tan(K[index, 0, 0] * 0.5)
#             tanfovy = math.tan(K[index, 1, 1] * 0.5)

#             world_view_transform = torch.zeros(4, 4).to(device=feature.device)
#             world_view_transform[0, 0] = 1
#             world_view_transform[1, 1] = 1
#             world_view_transform[2, 2] = 1
#             world_view_transform[3, 3] = 1
#             world_view_transform = world_view_transform.transpose(0, 1)
#             world_view_transform = world_view_transform.contiguous()

#             full_proj_transform = K[index]

#             # camera_center = world_view_transform.inverse()[3, :3]
#             camera_center = torch.zeros(1, 3).to(device=feature.device)
#             bg_color = [1, 1, 1]
#             background = torch.tensor(bg_color, dtype=torch.float32, device=feature.device) 
#             raster_settings = GaussianRasterizationSettings(
#                 image_height=self.image_height,
#                 image_width=self.image_width,
#                 tanfovx=tanfovx,
#                 tanfovy=tanfovy,
#                 bg=background,
#                 scale_modifier=self.scaling_modifier,
#                 viewmatrix=world_view_transform,
#                 projmatrix=full_proj_transform,
#                 sh_degree=1,
#                 campos=camera_center,
#                 prefiltered=False,
#                 debug=False
#             )
#             rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#             means2D_per  = torch.zeros_like(position[index], dtype=position.dtype, requires_grad=True, device=feature.device) + 1
#             # color_per = torch.ones(feature.shape[1], 3).to(device=feature.device)
#             color_per = feature[index]
#             position_per  = position[index]
#             semantic_feature_per = feature[index]
#             # print(semantic_feature_per.shape)
#             semantic_feature_per = torch.concat((semantic_feature_per, semantic_feature_per, semantic_feature_per[:, :2]), dim=1)
#             opacity_per = opacity[index]
#             scale_per = scale[index]
#             rotation_per = rotation[index]
#             rendered_image, feature_map, radii, depth = rasterizer(
#                 means3D = position_per,
#                 means2D = means2D_per,
#                 shs = None,
#                 colors_precomp = color_per, 
#                 semantic_feature = semantic_feature_per.unsqueeze(1), 
#                 opacities = opacity_per,
#                 scales = scale_per,
#                 rotations = rotation_per,
#                 cov3D_precomp = None)
#             # print(rendered_image)
#             gs_features[index:index+1] = rendered_image.unsqueeze(0)    
#         # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#         # They will be excluded from value updates used in the splitting criteria.
#         return gs_features.contiguous()



class BackprojectDepth_PointCloud(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, height, width):
        super(BackprojectDepth_PointCloud, self).__init__()

        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.repeat(depth.shape[0], 1, 1))
        cam_points = depth.view(depth.shape[0], 1, -1) * cam_points # [B, 3, N]

        return cam_points






class DN_to_Depth_Feature(nn.Module):
    """Layer to transform depth and normal into distance
    """
    def __init__(self, batch_size=1, height=192, width=640, min_depth=0.01, max_depth=100, inv_K=None):
        super(DN_to_Depth_Feature, self).__init__()

        # self.batch_size = batch_size
        # self.height = height
        # self.width = width
        # self.min_depth = min_depth 
        # self.max_depth = max_depth

        # U_coord = torch.arange(start=0, end=self.width).unsqueeze(0).repeat(self.height, 1).float().cuda()
        # V_coord = torch.arange(start=0, end=self.height).unsqueeze(1).repeat(1, self.width).float().cuda()

        # self.pix_coords = torch.stack([U_coord, V_coord], dim=0) # [2, H, W]
        # # self.pix_coords = self.pix_coords.unsqueeze(0).permute(0, 2, 3, 1)
        # # self.pix_coords[..., 0] /= self.width - 1
        # # self.pix_coords[..., 1] /= self.height - 1
        # self.pix_coords = self.pix_coords.unsqueeze(0)
        # self.pix_coords[:, 0:1] /= self.width - 1
        # self.pix_coords[:, 1:2] /= self.height - 1
        # self.pix_coords = (self.pix_coords - 0.5) * 2
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.inv_K = inv_K

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False).cuda() # 2, H, W  

        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False).cuda() # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

        self.normalized_points = torch.matmul(self.inv_K[:, :3, :3], self.pix_coords) # 1, 3, L

    def forward(self, norm_normal, distance):
        # norm_normal: B, 3*C, H, W
        # distance: B, C, H, W
        # self.batch_size = distance.shape[0]
        B, C, H, W = distance.shape
        N  = B * C
        norm_normal_new = norm_normal.reshape(N, 3, H, W)
        norm_normal_new = F.normalize(norm_normal_new, dim=1, p=2)
        distance_new = distance.reshape(N, 1, H, W)
            
        points = self.normalized_points.repeat(N, 1, 1).cuda()
        points = points.reshape(N, 3, self.height, self.width)
        normal_points = (norm_normal_new * points).sum(1, keepdim=True)
        depth = distance_new / (normal_points + 1e-7)
        return depth.reshape(B, C, H, W) # B, C, H, W

class DN_to_distance(nn.Module):
    """Layer to transform depth and normal into distance
    """
    def __init__(self, batch_size, height, width):
        super(DN_to_distance, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, depth, norm_normal, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        distance = (norm_normal.view(self.batch_size, 3, -1) * cam_points).sum(1, keepdim=True)
        distance = distance.reshape(self.batch_size, 1, self.height, self.width)
        return distance.abs()
    
class DN_to_depth(nn.Module):
    """Layer to transform distance and normal into depth
    """
    def __init__(self, batch_size, height, width, min_depth, max_depth):
        super(DN_to_depth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False).cuda() # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False).cuda() # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, norm_normal, distance, inv_K):
        _, distance = disp_to_depth(distance, min_depth=self.min_depth, max_depth=self.max_depth)
        distance = distance.clamp(self.min_depth, self.max_depth)
        normalized_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        normalized_points = normalized_points.reshape(self.batch_size, 3, self.height, self.width)
        normal_points = (norm_normal * normalized_points).sum(1, keepdim=True)
        depth = distance / (normal_points + 1e-7)
        return depth.abs()


class DN_to_depth_v1(nn.Module):
    """Layer to transform distance and normal into depth
    """
    def __init__(self, batch_size, height, width, min_depth, max_depth):
        super(DN_to_depth_v1, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = torch.from_numpy(self.id_coords) # 2, H, W  

        self.ones = torch.ones(self.batch_size, 1, self.height * self.width) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, norm_normal, distance, inv_K, min_value, max_value):
        b = norm_normal.shape[0]
        coords = self.pix_coords.repeat(b, 1, 1)
        coords = nn.Parameter(coords, requires_grad=False).to(norm_normal.device)

        _, distance = disp_to_depth(distance, min_depth=self.min_depth, max_depth=self.max_depth)
        distance = distance.clamp(min_value, max_value)
        normalized_points = torch.matmul(inv_K[:, :3, :3], coords)
        normalized_points = normalized_points.reshape(b, 3, self.height, self.width)
        normal_points = (norm_normal * normalized_points).sum(1, keepdim=True)
        depth = distance / (normal_points + 1e-7)
        return depth.abs()
    
    
class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()
        self.kernel = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.pad = nn.ReplicationPad2d(1)
    def forward(self, x):
        x = self.pad(x)
        y = F.conv2d(x, self.kernel.repeat(1, x.shape[1], 1, 1).to(x.device), stride=1).abs().clamp(min=0., max=1.0)
        return y
    
class Canny(nn.Module):
    def __init__(self):
        super(Canny, self).__init__()
        self.kernel = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.pad = nn.ReplicationPad2d(1)
    def forward(self, x):
        x = self.pad(x)
        y = F.conv2d(x, self.kernel.repeat(1, x.shape[1], 1, 1).to(x.device), stride=1).abs().clamp(min=0., max=1.0)
        return y
    
class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.kernel = torch.tensor([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
        self.pad = nn.ReplicationPad2d(1)
    def forward(self, x):
        x = self.pad(x)
        y = F.conv2d(x, self.kernel.repeat(1, x.shape[1], 1, 1).to(x.device), stride=1).abs().clamp(min=0., max=1.0)
        return y


def update_sample(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, depth_num, min_depth, max_depth, uncertainty_range):
    
    with torch.no_grad():    
        b, _, h, w = bin_edges.shape

        mode = 'direct'
        if mode == 'direct':
            depth_range = uncertainty_range
            depth_start_update = torch.clamp_min(depth_r - 0.5 * depth_range, min_depth)
        else:
            depth_range = uncertainty_range + (target_bin_right - target_bin_left).abs()
            depth_start_update = torch.clamp_min(target_bin_left - 0.5 * uncertainty_range, min_depth)

        interval = depth_range / depth_num
        interval = interval.repeat(1, depth_num, 1, 1)
        interval = torch.concat([torch.ones([b, 1, h, w], device=bin_edges.device) * depth_start_update, interval], 1)

        bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
        curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
    return bin_edges.detach(), curr_depth.detach()

def get_label(gt_depth_img, bin_edges, depth_num):

    with torch.no_grad():
        gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device)
        for i in range(depth_num):
            bin_mask = torch.ge(gt_depth_img, bin_edges[:, i])
            bin_mask = torch.logical_and(bin_mask, 
                torch.lt(gt_depth_img, bin_edges[:, i + 1]))
            gt_label[bin_mask] = i
        
        return gt_label


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def dist_to_distance(dist, min_dist, max_dist):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_dist
    max_disp = 1 / min_dist
    scaled_dist = min_disp * (dist / dist.abs()) + (max_disp - min_disp) * dist
    distance = 1 / scaled_dist
    return scaled_dist, distance




def normalization(d, mode):
    if mode == 'mean':
        m = torch.mean(d, [2,3], keepdim=True)
    elif mode == 'median':
        m = torch.median(d.view([d.shape[0], -1]), 1, keepdim=True)
        m = m[..., None, None]
    else:
        raise ValueError('Unknown normalization mode: %s'%mode)
    return d / m.clamp(1e-6)

def depth_to_disp(depth, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    r_depth = 1 / depth
    disp = (r_depth - min_disp) / (max_disp - min_disp)
    return disp

def distance_to_depth(disp, min_distance, max_distance):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_distance
    max_disp = 1 / min_distance
    scaled_distance = min_disp + (max_disp - min_disp) * disp
    distance = 1 / scaled_distance
    return scaled_distance, distance

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp



def right_from_parameters(axisangle, translation0, translation1):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t0 = translation0.clone()
    t1 = translation1.clone()

    t1 = t1 * -1
    t = t0 + t1

    T = get_translation_matrix(t)

    M = torch.matmul(R, T)

    return M


def left_from_parameters(translation):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    t = translation.clone()

    t = t * -1

    T = get_translation_matrix(t)

    return T


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    axisangle: 相机旋转偏移（角）
    translation: 相机位置便宜
    return 视图矩阵
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t = t * -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)


    # print(M.shape)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    # print(translation_vector.shape)

    t = translation_vector.contiguous().view(-1, 3, 1)
    # print(t.shape)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock_down(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_down, self).__init__()

        self.conv = Conv3x3_down(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock1x3_3x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x3_3x1, self).__init__()

        self.conv = Conv1x3_3x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class Conv1x3_3x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv1x3_3x1, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv1x3 = nn.Conv2d(int(in_channels), int(out_channels), (1, 3))
        self.conv3x1 = nn.Conv2d(int(out_channels), int(out_channels), (3, 1))
        # self.elu1 = nn.ELU(inplace=False)
        # self.elu2 = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv1x3(out)
        # out = self.elu1(out)
        out = self.conv3x1(out)
        # out = self.elu2(out)
        return out


class Conv3x3_down(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3_down, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, 2)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

"""
Use Euler method, which is the stand ResNet
1 block = 2layers para and flops
"""
"""
For shortcut option, I only tested A and identity.
If you want use option B, I guess BN in B shall be removed.
"""

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):

        out = self.conv2(F.relu(self.bn2(self.conv1(F.relu(self.bn1(x))))))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Project_depth(nn.Module):
    def __init__(self,B,H,W):
        super(Project_depth,self).__init__()
        self.B, self.H, self.W = B, H, W

    def forward(self, points, K, T):
        P = torch.matmul(K,T)[:,:3,:]
        cam_points = torch.matmul(P, points) #(B,3,-1)
        cam_points = cam_points.permute(0,2,1)
        uv = cam_points[:, :, :2]
        z_c = cam_points[:, :, 2]
        # uv = uv / cam_points[:, :, 2:3].clamp(min=1e-4)#1e-3
        # device = uv.device
        # uv = uv.long()
        # depth = torch.zeros((self.B, self.H, self.W), dtype=torch.float32).to(device)

        # for b in range(self.B):
        #     # in_view = (uv[b, :, 0] > 0) & (uv[b, :, 1] > 0) & (uv[b, :, 0] < self.W) & \
        #     #           (uv[b, :, 1] < self.H) & (z_c[b, :] > 1e-3)#1e-2
        #     # compute outside of image
        #     px = uv[b, :, 0]
        #     py = uv[b, :, 1]
        #     x = (px + 1.0) * (self.W - 1.0) / 2.0
        #     y = (py + 1.0) * (self.H - 1.0) / 2.0
        #     # Compute the coordinates of the 4 pixels to sample from.
        #     x0 = x
        #     x1 = x0 + 1
        #     y0 = y
        #     y1 = y0 + 1
        #     in_view = (x0 > 0) & (y0 > 0) & (x1 < (self.W - 1)) & \
        #               (y1 < (self.H - 1)) & (z_c[b, :] > 1e-3)#1e-2
        #     in_view = in_view.to(device)
        #     uv_b, z_c_b = uv[b, in_view], z_c[b, in_view]
        #     depth[b, uv_b[:, 1], uv_b[:, 0]] = z_c_b
        # depth = depth.unsqueeze(dim=1)
        # # depth = depth
        depth = z_c.view(self.B, self.H, self.W).unsqueeze(dim=1)
        return depth



class DetailGuide(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(DetailGuide, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

    def forward(self, height_re, width_re, dxy):

        points_all = []
        for i in range(self.batch_size):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = nn.Parameter(torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0), requires_grad=False).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]

            points_all.append(pix_coords)

        points_all = torch.concat(points_all, 0)   #[12,2,122880]
        points_all = points_all.view(self.batch_size, 2, self.height, self.width)
        points_all = points_all.permute(0, 2, 3, 1) #[12,192,640,2]
        points_all[..., 0] = points_all[..., 0] * (self.width * 1.0 / width_re)
        points_all[..., 1] = points_all[..., 1] * (self.height * 1.0 / height_re)
        points_all[..., 0] = points_all[..., 0]  / self.width - 1
        points_all[..., 1] = points_all[..., 1] / self.height - 1
        points_all = (points_all - 0.5) * 2

        return points_all



class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
        #                              requires_grad=False)   #[1,1,122880]

        # meshgrid = np.meshgrid(range(dx, dx+640), range(dy, dy+192), indexing='xy')
        # self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
        #                               requires_grad=False)  #[2,192,640]

        # self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
        #                          requires_grad=False)   #[12,1,122880]

        # self.pix_coords = torch.unsqueeze(torch.stack(
        #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)    #[1,2,122880]
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)  #[12,2,122880]
        # self.pix_coords = nn.Parameter(torch.concat([self.pix_coords, self.ones], 1),
        #                                requires_grad=False)   #[12,3,122880]

    def forward(self, depth, inv_K, dxy):

        cam_points_all = []
        for i in range(self.batch_size):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]
            pix_coords = nn.Parameter(torch.concat([pix_coords, ones], 1),
                                           requires_grad=False).cuda()   #[1,3,122880]

            cam_points = torch.matmul(inv_K[i, :3, :3], pix_coords)   #[1,3,122880]
            cam_points = depth[i,0,:,:].view(1, 1, -1) * cam_points   #[1,3,122880]
            cam_points = torch.concat([cam_points, ones], 1)   #[1,4,122880]
            cam_points_all.append(cam_points)

        cam_points_all = torch.concat(cam_points_all, 0)

        return cam_points_all

class BackprojectDepth_v1(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, height, width):
        super(BackprojectDepth_v1, self).__init__()

        self.height = height
        self.width = width

        # self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
        #                              requires_grad=False)   #[1,1,122880]

        # meshgrid = np.meshgrid(range(dx, dx+640), range(dy, dy+192), indexing='xy')
        # self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
        #                               requires_grad=False)  #[2,192,640]

        # self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
        #                          requires_grad=False)   #[12,1,122880]

        # self.pix_coords = torch.unsqueeze(torch.stack(
        #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)    #[1,2,122880]
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)  #[12,2,122880]
        # self.pix_coords = nn.Parameter(torch.concat([self.pix_coords, self.ones], 1),
        #                                requires_grad=False)   #[12,3,122880]

    def forward(self, depth, inv_K, dxy):

        cam_points_all = []
        for i in range(depth.shape[0]):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]
            pix_coords = nn.Parameter(torch.concat([pix_coords, ones], 1),
                                           requires_grad=False).cuda()   #[1,3,122880]

            cam_points = torch.matmul(inv_K[i, :3, :3], pix_coords)   #[1,3,122880]
            cam_points = depth[i,0,:,:].view(1, 1, -1) * cam_points   #[1,3,122880]
            cam_points_all.append(cam_points)

        cam_points_all = torch.concat(cam_points_all, 0)

        return cam_points_all


class BackprojectDepth_removedxy(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth_removedxy, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords) # [batch_size, 3, height*width]
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points



class BackprojectDepth_v2(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth_v2, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)
        
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # [B, 1, L]
        
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # [1, 2, L]
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1).cuda()  # [B, 2, L]

    def forward(self, depth, inv_K, dxy):
        dxy = dxy.unsqueeze(2)
        dxy = dxy.repeat(1, 1, self.pix_coords.shape[2])
        pix_coords = self.pix_coords + dxy
        pix_coords = nn.Parameter(torch.concat([pix_coords, self.ones], 1),
                                        requires_grad=False).cuda()   #[1,3,122880]
        
        cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords)   #[1,3,122880]
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points   #[1,3,122880]
        cam_points = torch.concat([cam_points, self.ones], 1)   #[1,4,122880]

        return cam_points


class BackprojectDepth_Enhance(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth_Enhance, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
        #                              requires_grad=False)   #[1,1,122880]

        # meshgrid = np.meshgrid(range(dx, dx+640), range(dy, dy+192), indexing='xy')
        # self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
        #                               requires_grad=False)  #[2,192,640]

        # self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
        #                          requires_grad=False)   #[12,1,122880]

        # self.pix_coords = torch.unsqueeze(torch.stack(
        #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)    #[1,2,122880]
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)  #[12,2,122880]
        # self.pix_coords = nn.Parameter(torch.concat([self.pix_coords, self.ones], 1),
        #                                requires_grad=False)   #[12,3,122880]

    def forward(self, depth, inv_K, dxy):

        cam_points_all = []
        for i in range(self.batch_size):
            meshgrid = np.meshgrid(range(int(dxy[i,0]), int(dxy[i,0]+self.width)), range(int(dxy[i,1]), int(dxy[i,1]+self.height)), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = nn.Parameter(torch.from_numpy(id_coords),
                                          requires_grad=False)  #[2,192,640]

            ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                     requires_grad=False).cuda()   #[1,1,122880]

            pix_coords = torch.unsqueeze(torch.stack(
                [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0).cuda()    #[1,2,122880]
            # self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)  #[12,2,122880]
            pix_coords = nn.Parameter(torch.concat([pix_coords, ones], 1),
                                           requires_grad=False).cuda()   #[1,3,122880]
            # pix_coords = torch.concat([pix_coords, ones], 1)   #[1,4,122880]

            cam_points = torch.matmul(inv_K[i, :3, :3], pix_coords)   #[1,3,122880]
            cam_points = depth[i,0,:,:].view(1, 1, -1) * cam_points   #[1,3,122880]
            cam_points = torch.concat([cam_points, ones], 1)   #[1,4,122880]
            cam_points_all.append(cam_points)

        cam_points_all = torch.concat(cam_points_all, 0)

        return cam_points_all


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, dxy):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        dxy = dxy.unsqueeze(1).unsqueeze(2).expand(-1, self.height, self.width, -1)
        pix_coords -= dxy
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        # print(pix_coords)
        return pix_coords
    
    
class Project3D_removedxy(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_removedxy, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class Project3D_Occlusion(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, dxy):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        dxy = dxy.unsqueeze(1).unsqueeze(2).expand(-1, self.height, self.width, -1)
        pix_coords -= dxy
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        # print(pix_coords)
        return pix_coords, cam_points[:, 2].clamp(min=1e-3).view(self.batch_size, 1, self.height, self.width)


class Project3D_res(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_res, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, dxy, t_res):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        cam_points = cam_points
        # t_res =  t_res.view(t_res.shape[0], 3, -1)
        # t_ones = torch.zeros(t_res.shape[0], 1, t_res.shpe[2]).to(device=t_res.device)
        # t_res = torch.concat((t_res, t_ones), dim=1)

        xyz_res = torch.matmul(K[:, :3, :3], t_res.view(t_res.shape[0], 3, -1))
        # print(cam_points.shape, xyz_res.shape)
        cam_points = cam_points + xyz_res

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        dxy = dxy.unsqueeze(1).unsqueeze(2).expand(-1, self.height, self.width, -1)
        pix_coords -= dxy
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class Project3D_poseconsis(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D_poseconsis, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, T):
        # P = torch.matmul(K, T)[:, :3, :] #P:[12,3,4]   T:[12,4,4]    points:[12,4,1]

        # cam_points = torch.matmul(P, points)

        cam1 = torch.matmul(T, points)  #[12,4,1]

        return cam1


def updown_sample(x, scale_fac, mode='nearest'):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode=mode)


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def downsample(x):
    """Downsample input tensor by a factor of 1/2
    """
    return F.interpolate(x, scale_factor=1.0/2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()



def get_smooth_ND(normal, distance, planar_mask):
    
    """Computes the smoothness loss for normal and distance
    """
    grad_normal_x = torch.mean(torch.abs(normal[:, :, :, :-1] - normal[:, :, :, 1:]), 1, keepdim=True)
    grad_normal_y = torch.mean(torch.abs(normal[:, :, :-1, :] - normal[:, :, 1:, :]), 1, keepdim=True)

    grad_distance_x = torch.abs(distance[:, :, :, :-1] - distance[:, :, :, 1:])
    grad_distance_y = torch.abs(distance[:, :, :-1, :] - distance[:, :, 1:, :])

    planar_mask_x = planar_mask[:, :, :, :-1]
    planar_mask_y = planar_mask[:, :, :-1, :]
    
    loss_grad_normal = (grad_normal_x * planar_mask_x).sum() / (planar_mask_x.sum() + 1e-7) + (grad_normal_y * planar_mask_y).sum() / (planar_mask_y.sum() + 1e-7)
    loss_grad_distance = (grad_distance_x * planar_mask_x).sum() / (planar_mask_x.sum() + + 1e-7) + (grad_distance_y * planar_mask_y).sum() / (planar_mask_y.sum() + 1e-7)
    
    return loss_grad_normal, loss_grad_distance



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    # print("eva:, ", abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

    # return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]


################################################ SENet ######################

class SEModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super(SEModule, self).__init__()

        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        # self.sigmoid = nn.Sigmoid()


    def forward(self, features):

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # y = self.sigmoid(y) * 2.0
        features = features + y.expand_as(features)

        return features




class DepthToNormal(nn.Module):
    """Layer to transform depth to normal
    """
    def __init__(self, batch_size, height, width):
        super(DepthToNormal, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False).cuda() # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False).cuda() # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, depth, inv_K, nei):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        # normalized_points = normalized_points.reshape(self.batch_size, 3, self.height, self.width)
        # normal_points = (norm_normal * normalized_points).sum(1, keepdim=True)
        # disp = normal_points / (distance + 1e-7)
        # return disp.abs()
    
        pts_3d_map = cam_points[:, :3, :].permute(0,2,1).view(-1, self.height, self.width, 3)
    
        ## shift the 3d pts map by nei along 8 directions
        pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
        pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
        pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
        pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
        pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
        pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
        pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
        pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
        pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

        ## generate difference between the central pixel and one of 8 neighboring pixels
        diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
        diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
        diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
        diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
        diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
        diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
        diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
        diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

        diff_x0 = diff_x0.reshape(-1, 3)
        diff_y0 = diff_y0.reshape(-1, 3)
        diff_x1 = diff_x1.reshape(-1, 3)
        diff_y1 = diff_y1.reshape(-1, 3)
        diff_x0y0 = diff_x0y0.reshape(-1, 3)
        diff_x0y1 = diff_x0y1.reshape(-1, 3)
        diff_x1y0 = diff_x1y0.reshape(-1, 3)
        diff_x1y1 = diff_x1y1.reshape(-1, 3)

        ## calculate normal by cross product of two vectors
        normals0 = torch.cross(diff_x1, diff_y1)
        normals1 =  torch.cross(diff_x0, diff_y0)
        normals2 = torch.cross(diff_x0y1, diff_x0y0)
        normals3 = torch.cross(diff_x1y0, diff_x1y1)

        normal_vector = normals0 + normals1 + normals2 + normals3
        normal_vectorl2 = torch.norm(normal_vector, p=2, dim = 1)
        normal_vector = torch.div(normal_vector.permute(1,0), normal_vectorl2)
        normal_vector = normal_vector.permute(1,0).view(pts_3d_map_ctr.shape).permute(0,3,1,2)
        # normal_map = F.pad( normal_vector, (0,2*nei,2*nei,0),"constant",value=0)
        # normal_map = F.pad(normal_vector, [1] * 4, mode="replicate")  # SB, 3, H+2, W+2
        normal_map = F.pad(normal_vector, [nei] * 4, mode="replicate")  # SB, 3, H+2, W+2
        normal =  F.normalize(normal_map, dim=1, p=2)

        return normal


class DepthToNormal_v1(nn.Module):
    """Layer to transform distance and normal into depth
    """
    def __init__(self, batch_size, height, width):
        super(DepthToNormal_v1, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False).cuda() # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False).cuda() # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, depth, inv_K, nei):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        # normalized_points = normalized_points.reshape(self.batch_size, 3, self.height, self.width)
        # normal_points = (norm_normal * normalized_points).sum(1, keepdim=True)
        # disp = normal_points / (distance + 1e-7)
        # return disp.abs()
    
        pts_3d_map = cam_points[:, :3, :].permute(0,2,1).view(-1, self.height, self.width, 3)
    
        ## shift the 3d pts map by nei along 8 directions
        pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
        pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
        pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
        pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
        pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
        pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
        pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
        pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
        pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

        ## generate difference between the central pixel and one of 8 neighboring pixels
        diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
        diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
        diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
        diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
        diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
        diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
        diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
        diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

        diff_x0 = diff_x0.reshape(-1, 3)
        diff_y0 = diff_y0.reshape(-1, 3)
        diff_x1 = diff_x1.reshape(-1, 3)
        diff_y1 = diff_y1.reshape(-1, 3)
        diff_x0y0 = diff_x0y0.reshape(-1, 3)
        diff_x0y1 = diff_x0y1.reshape(-1, 3)
        diff_x1y0 = diff_x1y0.reshape(-1, 3)
        diff_x1y1 = diff_x1y1.reshape(-1, 3)

        ## calculate normal by cross product of two vectors
        normals0 = torch.cross(diff_x1, diff_y1)
        normals1 =  torch.cross(diff_x0, diff_y0)
        normals2 = torch.cross(diff_x0y1, diff_x0y0)
        normals3 = torch.cross(diff_x1y0, diff_x1y1)

        normal_vector = normals0 + normals1 + normals2 + normals3
        normal_vectorl2 = torch.norm(normal_vector, p=2, dim = 1)
        normal_vector = torch.div(normal_vector.permute(1,0), normal_vectorl2)
        normal_vector = normal_vector.permute(1,0).view(pts_3d_map_ctr.shape).permute(0,3,1,2)
        # normal_map = F.pad( normal_vector, (0,2*nei,2*nei,0),"constant",value=0)
        normal_map = F.pad(normal_vector, [1] * 4, mode="replicate")  # SB, 3, H+2, W+2
        normal =  -F.normalize(normal_map, dim=1, p=2)

        return normal



# DINER: Depth-aware Image-based NEural Radiance fields
# Official PyTorch implementation of the CVPR 2023 paper 
import torch
from torch.nn.functional import pad
import matplotlib.pyplot as plt


@torch.no_grad()
def depth2normal_DINER(dmap, K):
    """
    calculating normal maps from depth map via central difference
    Parameters
    ----------
    dmap  (N, 1, H, W)
    K (N, 3, 3)

    Returns
    -------
    normal (N, 3, H, W)

    """
    N, _, H, W = dmap.shape
    device = dmap.device

    # reprojecting dmap to pointcloud
    image_rays = torch.stack(torch.meshgrid(torch.arange(0.5, H, 1., device=device),
                                            torch.arange(0.5, W, 1., device=device))[::-1],
                             dim=-1).reshape(-1, 2)  # H, W, 2
    image_rays = image_rays.unsqueeze(0).expand(N, -1, -1).clone()  # N, H*W, 2
    image_rays -= K[:, [0, 1], -1].unsqueeze(-2)
    image_rays /= K[:, [0, 1], [0, 1]].unsqueeze(-2)
    image_rays = torch.concat((image_rays, torch.ones_like(image_rays[..., -1:])), dim=-1)  # SB, H*W, 3
    image_pts = image_rays.view(N, H, W, 3) * dmap.view(N, H, W, 1)  # SB, H, W, 3
    image_pts = image_pts.permute(0, 3, 1, 2)  # SB, 3, H, W
    image_pts = pad(image_pts, [1] * 4, mode="replicate")  # SB, 3, H+2, W+2

    # calculating normals
    image_pts_offset_down = image_pts[:, :, 2:, 1:-1]  # SB, 3, H, W
    image_pts_offset_up = image_pts[:, :, :-2, 1:-1]  # SB, 3, H, W
    image_pts_offset_right = image_pts[:, :, 1:-1, 2:]  # SB, 3, H, W
    image_pts_offset_left = image_pts[:, :, 1:-1, :-2]  # SB, 3, H, W

    vdiff = image_pts_offset_down - image_pts_offset_up  # SB, 3, H, W
    hdiff = image_pts_offset_right - image_pts_offset_left  # SB, 3, H, W
    normal = torch.cross(vdiff.permute(0, 2, 3, 1), hdiff.permute(0, 2, 3, 1))  # SB, H, W, 3
    normal /= torch.norm(normal, p=2, dim=-1, keepdim=True)  # SB, H, W, 3

    # cleaning normal map
    idx_map = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(H), torch.arange(W)),
                          dim=-1).to(device)  # SB, H, W, 3
    offset_map = torch.zeros_like(idx_map)
    helper = (image_pts_offset_down[:, 0] == 0)[..., None] & \
             torch.tensor([False, True, False], device=device).view(1, 1, 1, 3)
    offset_map[helper] += -1
    helper = (image_pts_offset_up[:, 0] == 0)[..., None] & \
             torch.tensor([False, True, False], device=device).view(1, 1, 1, 3)
    offset_map[helper] += 1
    helper = (image_pts_offset_right[:, 0] == 0)[..., None] & \
             torch.tensor([False, False, True], device=device).view(1, 1, 1, 3)
    offset_map[helper] += -1
    helper = (image_pts_offset_left[:, 0] == 0)[..., None] & \
             torch.tensor([False, False, True], device=device).view(1, 1, 1, 3)
    offset_map[helper] += 1

    offset_mask = torch.any(offset_map != 0, dim=-1)
    new_idcs = idx_map[offset_mask] + offset_map[offset_mask]
    new_idcs[:, 1] = new_idcs[:, 1].clip(min=0, max=H - 1)
    new_idcs[:, 2] = new_idcs[:, 2].clip(min=0, max=W - 1)
    normal[offset_mask] = normal[new_idcs[:, 0], new_idcs[:, 1], new_idcs[:, 2]]
    normal[dmap[:, 0] == 0] = 0

    normal = -normal.permute(0, 3, 1, 2)

    return normal


class AbsLoss(object):
    r"""An abstract class for loss functions. 
    """
    def __init__(self):
        self.record = []
        self.bs = []
    
    def compute_loss(self, pred, gt):
        r"""Calculate the loss.
        
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.

        Return:
            torch.Tensor: The loss.
        """
        pass
    
    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss
    
    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def _reinit(self):
        self.record = []
        self.bs = []

# class NormalLoss(AbsLoss):
#     def __init__(self):
#         super(NormalLoss, self).__init__()
        
#     def compute_loss(self, pred, gt):
#         # gt has been normalized on the NYUv2 dataset
#         pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
#         binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
#         loss = 1 - torch.sum((pred*gt)*binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
#         return loss
    
# class NormalLoss(AbsLoss):
#     def __init__(self):
#         super(NormalLoss, self).__init__()
        
def normalLoss(pred, gt):
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # gt = F.normalize(gt, dim=1, p=2)
    # norm_loss_score = cos(pred, gt)
    # gt has been normalized on the NYUv2 dataset
    # pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
    # gt = F.normalize(gt, dim=1, p=2)
    binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
    loss = 1 - torch.sum((pred*gt)*binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    # loss = 1 - torch.sum(pred * gt)
    # loss = 1 - norm_loss_score

    return loss


def normalize(a):
    # print(a.shape)
    return (a - a.min())/(a.max() - a.min() + 1e-8)


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

# def compute_seg(rgb, aligned_norm, D, planar_area_thresh=400):
        
#         """
#         inputs:
#             rgb                 b, 3, H, W
#             aligned_norm        b, 3, H, W
#             D                   b, H, W

#         outputs:
#             segment                b, 1, H, W
#             planar mask        b, 1, H, W
#         """
#         # print(rgb.shape, aligned_norm.shape, D.shape)
#         b, _, h, w  = rgb.shape
#         device = rgb.device

#         # compute cost
#         pdist = nn.PairwiseDistance(p=2)

#         D_down = abs(D[:, 1:] - D[:, :-1])
#         D_right = abs(D[:, :, 1:] - D[:, :, :-1])

#         norm_down = pdist(aligned_norm[:, :, 1:], aligned_norm[:, :, :-1])
#         # norm_right = pdist(aligned_norm[:, :, :, 1:], aligned_norm[:, :, :, :-1])
#         norm_down = pdist(aligned_norm[:, :, 1:].permute(0, 2, 3, 1), aligned_norm[:, :, :-1].permute(0, 2, 3, 1))
#         norm_right = pdist(aligned_norm[:, :, :, 1:].permute(0, 2, 3, 1), aligned_norm[:, :, :, :-1].permute(0, 2, 3, 1))

#         D_down = torch.stack([normalize(D_down[i]) for i in range(b)])
#         norm_down = torch.stack([normalize(norm_down[i]) for i in range(b)])

#         D_right = torch.stack([normalize(D_right[i]) for i in range(b)])
#         norm_right = torch.stack([normalize(norm_right[i]) for i in range(b)])

#         # print(D_down.shape, norm_down.shape, D_right.shape, norm_right.shape)
#         normD_down = D_down + norm_down
#         normD_right = D_right + norm_right

#         normD_down = torch.stack([normalize(normD_down[i]) for i in range(b)])
#         normD_right = torch.stack([normalize(normD_right[i]) for i in range(b)])

#         cost_down = normD_down
#         cost_right = normD_right
        
#         # get dissimilarity map visualization
#         # dst = cost_down[:,  :,  : -1] + cost_right[ :, :-1, :]
        
#         # felz_seg
#         cost_down_np = cost_down.detach().cpu().numpy()
#         cost_right_np = cost_right.detach().cpu().numpy()
#         segment = torch.stack([torch.from_numpy(felz_seg(normalize(cost_down_np[i]), normalize(cost_right_np[i]), 0, 0, h, w, scale=1, min_size=50)).to(device) for i in range(b)])
#         segment += 1
#         segment = segment.unsqueeze(1)
        
#         # generate mask for segment with area larger than 200
#         max_num = segment.max().item() + 1

#         area = torch.zeros((b, max_num)).to(device)
#         area.scatter_add_(1, segment.view(b, -1), torch.ones((b, 1, h, w)).view(b, -1).to(device))

#         planar_area_thresh = planar_area_thresh
#         valid_mask = (area > planar_area_thresh).float()
#         planar_mask = torch.gather(valid_mask, 1, segment.view(b, -1))
#         planar_mask = planar_mask.view(b, 1, h, w)

#         planar_mask[:, :, :8, :] = 0
#         planar_mask[:, :, -8:, :] = 0
#         planar_mask[:, :, :, :8] = 0
#         planar_mask[:, :, :, -8:] = 0

#         # return segment, planar_mask, dst.unsqueeze(1)
#         return planar_mask




# structDepth
# ----- vps
def depth2norm(cam_points, height, width, nei=3):
    pts_3d_map = cam_points[:, :3, :].permute(0,2,1).view(-1, height, width, 3)
    
    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    diff_x0 = diff_x0.reshape(-1, 3)
    diff_y0 = diff_y0.reshape(-1, 3)
    diff_x1 = diff_x1.reshape(-1, 3)
    diff_y1 = diff_y1.reshape(-1, 3)
    diff_x0y0 = diff_x0y0.reshape(-1, 3)
    diff_x0y1 = diff_x0y1.reshape(-1, 3)
    diff_x1y0 = diff_x1y0.reshape(-1, 3)
    diff_x1y1 = diff_x1y1.reshape(-1, 3)

    ## calculate normal by cross product of two vectors
    normals0 = torch.cross(diff_x1, diff_y1)
    normals1 =  torch.cross(diff_x0, diff_y0)
    normals2 = torch.cross(diff_x0y1, diff_x0y0)
    normals3 = torch.cross(diff_x1y0, diff_x1y1)

    normal_vector = normals0 + normals1 + normals2 + normals3
    normal_vectorl2 = torch.norm(normal_vector, p=2, dim = 1)
    normal_vector = torch.div(normal_vector.permute(1,0), normal_vectorl2)
    normal_vector = normal_vector.permute(1,0).view(pts_3d_map_ctr.shape).permute(0,3,1,2)
    normal_map = F.pad( normal_vector, (0,2*nei,2*nei,0),"constant",value=0)
    normal = - F.normalize(normal_map, dim=1, p=2)

    return normal

class SE_block(nn.Module):
    def __init__(self, in_channel, visual_weights = False, reduction = 16 ):
        super(SE_block, self).__init__()
        reduction = reduction
        in_channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channel // reduction, in_channel, bias = False)
            )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace = True)
        self.vis = False
    
    def forward(self, in_feature):

        b,c,_,_ = in_feature.size()
        output_weights_avg = self.avg_pool(in_feature).view(b,c)
        output_weights_max = self.max_pool(in_feature).view(b,c)
        output_weights_avg = self.fc(output_weights_avg).view(b,c,1,1)
        output_weights_max = self.fc(output_weights_max).view(b,c,1,1)
        output_weights = output_weights_avg + output_weights_max
        output_weights = self.sigmoid(output_weights)
        return output_weights.expand_as(in_feature) * in_feature

## ChannelAttetion
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_planes // ratio, in_planes, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b,c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature

## SpatialAttetion

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, in_feature):
        x = in_feature
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.concat([avg_out, max_out], dim=1)
        #x = avg_out
        #x = max_out
        x = self.conv1(x)
        return self.sigmoid(x).expand_as(in_feature) * in_feature


#CS means channel-spatial  
class CS_Block(nn.Module):
    def __init__(self, in_channel, reduction = 16 ):
        super(CS_Block, self).__init__()
        
        reduction = reduction
        in_channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channel // reduction, in_channel, bias = False)
            )
        self.sigmoid = nn.Sigmoid()
        ## Spatial_Block
        self.conv = nn.Conv2d(2,1,kernel_size = 1, bias = False)
        #self.conv = nn.Conv2d(1,1,kernel_size = 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, in_feature):

        b,c,_,_ = in_feature.size()
        
        
        output_weights_avg = self.avg_pool(in_feature).view(b,c)
        output_weights_max = self.max_pool(in_feature).view(b,c)
         
        output_weights_avg = self.fc(output_weights_avg).view(b,c,1,1)
        output_weights_max = self.fc(output_weights_max).view(b,c,1,1)
        
        output_weights = output_weights_avg + output_weights_max
        
        output_weights = self.sigmoid(output_weights)
        out_feature_1 = output_weights.expand_as(in_feature) * in_feature
        
        ## Spatial_Block
        in_feature_avg = torch.mean(out_feature_1,1,True)
        in_feature_max,_ = torch.max(out_feature_1,1,True)
        mixed_feature = torch.concat([in_feature_avg,in_feature_max],1)
        spatial_attention = self.sigmoid(self.conv(mixed_feature))
        out_feature = spatial_attention.expand_as(out_feature_1) * out_feature_1
        #########################
        
        return out_feature
        
class Attention_Module(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel = None):
        super(Attention_Module, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        #self.sa = SpatialAttention()
        #self.cs = CS_Block(channel)
        self.conv_se = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1 )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, high_features, low_features):

        features = [upsample(high_features)]
        features.append(low_features)
        features = torch.concat(features, 1)

        # features = [upsample(high_features)]
        # print(upsample(high_features).shape, low_features.shape)
        # features += low_features
        # features = torch.concat(features, 1)

        # features = torch.concat((upsample(high_features), low_features), dim=1)

        features = self.ca(features)
        #features = self.sa(features)
        #features = self.cs(features)
        
        return self.relu(self.conv_se(features))

    def hr_forward(self, high_features, low_features):

        features = [high_features]
        features += low_features

        features = torch.concat(features, 1)

        features = self.ca(features)
        #features = self.sa(features)
        #features = self.cs(features)
        
        return self.relu(self.conv_se(features))

    def no_relu_forward(self, high_features, low_features):

        features = [upsample(high_features)]
        features += low_features
        features = torch.concat(features, 1)

        features = self.ca(features)
        #features = self.sa(features)
        #features = self.cs(features)
        
        return self.conv_se(features)







# 2021 RAL: Three-Filters-to-Normal: An Accurate and Ultrafast Surface Normal Estimator
# class SNE(nn.Module):
#     """Our SNE takes depth and camera intrinsic parameters as input,
#     and outputs normal estimations.
#     """
#     def __init__(self):
#         super(SNE, self).__init__()

#     def forward(self, depth, camParam):
#         h,w = depth.size()
#         v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))
#         v_map = v_map.type(torch.float32)
#         u_map = u_map.type(torch.float32)

#         Z = depth   # h, w
#         Y = Z.mul((v_map - camParam[1,2])) / camParam[0,0]  # h, w
#         X = Z.mul((u_map - camParam[0,2])) / camParam[0,0]  # h, w
#         Z[Y <= 0] = 0
#         Y[Y <= 0] = 0
#         Z[torch.isnan(Z)] = 0
#         D = torch.div(torch.ones(h, w), Z)  # h, w

#         Gx = torch.tensor([[0,0,0],[-1,0,1],[0,0,0]], dtype=torch.float32)
#         Gy = torch.tensor([[0,-1,0],[0,0,0],[0,1,0]], dtype=torch.float32)

#         Gu = F.conv2d(D.view(1,1,h,w), Gx.view(1,1,3,3), padding=1)
#         Gv = F.conv2d(D.view(1,1,h,w), Gy.view(1,1,3,3), padding=1)

#         nx_t = Gu * camParam[0,0]   # 1, 1, h, w
#         ny_t = Gv * camParam[1,1]   # 1, 1, h, w

#         phi = torch.atan(torch.div(ny_t, nx_t)) + torch.ones([1,1,h,w])*3.141592657
#         a = torch.cos(phi)
#         b = torch.sin(phi)

#         diffKernelArray = torch.tensor([[-1, 0, 0, 0, 1, 0, 0, 0, 0],
#                                         [ 0,-1, 0, 0, 1, 0, 0, 0, 0],
#                                         [ 0, 0,-1, 0, 1, 0, 0, 0, 0],
#                                         [ 0, 0, 0,-1, 1, 0, 0, 0, 0],
#                                         [ 0, 0, 0, 0, 1,-1, 0, 0, 0],
#                                         [ 0, 0, 0, 0, 1, 0,-1, 0, 0],
#                                         [ 0, 0, 0, 0, 1, 0, 0,-1, 0],
#                                         [ 0, 0, 0, 0, 1, 0, 0, 0,-1]], dtype=torch.float32)

#         sum_nx = torch.zeros((1,1,h,w), dtype=torch.float32)
#         sum_ny = torch.zeros((1,1,h,w), dtype=torch.float32)
#         sum_nz = torch.zeros((1,1,h,w), dtype=torch.float32)

#         for i in range(8):
#             diffKernel = diffKernelArray[i].view(1,1,3,3)
#             X_d = F.conv2d(X.view(1,1,h,w), diffKernel, padding=1)
#             Y_d = F.conv2d(Y.view(1,1,h,w), diffKernel, padding=1)
#             Z_d = F.conv2d(Z.view(1,1,h,w), diffKernel, padding=1)

#             nz_i = torch.div((torch.mul(nx_t, X_d) + torch.mul(ny_t, Y_d)), Z_d)
#             norm = torch.sqrt(torch.mul(nx_t, nx_t) + torch.mul(ny_t, ny_t) + torch.mul(nz_i, nz_i))
#             nx_t_i = torch.div(nx_t, norm)
#             ny_t_i = torch.div(ny_t, norm)
#             nz_t_i = torch.div(nz_i, norm)

#             nx_t_i[torch.isnan(nx_t_i)] = 0
#             ny_t_i[torch.isnan(ny_t_i)] = 0
#             nz_t_i[torch.isnan(nz_t_i)] = 0

#             sum_nx = sum_nx + nx_t_i
#             sum_ny = sum_ny + ny_t_i
#             sum_nz = sum_nz + nz_t_i

#         theta = -torch.atan(torch.div((torch.mul(sum_nx, a) + torch.mul(sum_ny, b)), sum_nz))
#         nx = torch.mul(torch.sin(theta), torch.cos(phi))
#         ny = torch.mul(torch.sin(theta), torch.sin(phi))
#         nz = torch.cos(theta)

#         nx[torch.isnan(nz)] = 0
#         ny[torch.isnan(nz)] = 0
#         nz[torch.isnan(nz)] = -1

#         sign = torch.ones((1,1,h,w), dtype=torch.float32)
#         sign[ny > 0] = -1

#         nx = torch.mul(nx, sign).squeeze(dim=0)
#         ny = torch.mul(ny, sign).squeeze(dim=0)
#         nz = torch.mul(nz, sign).squeeze(dim=0)

#         return torch.concat([nx, ny, nz], dim=0)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import os, cv2
# import matplotlib.pyplot as plt
# from util import uv2coords, uv2xyz, coords2uv

# # This repository hosts the original implementation of CVPR 2022 (oral) paper "OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion "

# def depth2normal(depth):
#     device = depth.device
#     bs, _, h, w = depth.shape
#     depth = depth.reshape(bs, h, w)
#     coords = np.stack(np.meshgrid(range(w), range(h)), -1)
#     coords = np.reshape(coords, [-1, 2])
#     coords += 1
#     uv = coords2uv(coords, w, h)          
#     xyz = uv2xyz(uv) 
#     xyz = torch.from_numpy(xyz).to(device)
#     xyz = xyz.unsqueeze(0).repeat(bs, 1, 1)
    
#     depth_reshape = depth.reshape(bs, h*w, 1)
#     newxyz = xyz * depth_reshape
#     newxyz_reshape = newxyz.reshape(bs, h, w, 3).permute(0, 3, 1, 2)  #bs x 3 x h x w
#     kernel_size = 5
#     point_matrix = F.unfold(newxyz_reshape, kernel_size=kernel_size, stride=1, padding=kernel_size-1, dilation=2)
    
#     # An = b 
#     matrix_a = point_matrix.view(bs, 3, kernel_size*kernel_size, h, w)  # (B, 3, 25, H, W)
#     matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, H, W, 25, 3)
#     matrix_a_trans = matrix_a.transpose(3, 4) # (b, h, w, 3, 25)
#     matrix_b = torch.ones([bs, h, w, kernel_size*kernel_size, 1], device=device)
    
#     #normal = torch.linalg.lstsq(matrix_a, matrix_b)
#     #normal = normal.solution.squeeze(-1)
#     #norm_normalize = F.normalize(normal, p=2, dim=-1)
#     #norm_normalize = norm_normalize.permute(0, 3, 1, 2)
    
    
#     # dot(A.T, A)
#     point_multi = torch.matmul(matrix_a_trans, matrix_a)  # (b, h, w, 3, 3)
#     matrix_deter = torch.det(point_multi)
#     # make inversible
#     inverse_condition = torch.ge(matrix_deter, 1e-5)
#     inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
#     inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
#     # diag matrix to update uninverse
#     diag_constant = torch.ones([3], dtype=torch.float32, device=device)
#     diag_element = torch.diag(diag_constant)
#     diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
#     diag_matrix = diag_element.repeat(bs, h, w, 1, 1)
#     # inversible matrix

#     inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
#     inv_matrix = torch.linalg.inv(inversible_matrix)
    
#     generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
  
#     norm_normalize = F.normalize(generated_norm, p=2, dim=3)
#     norm_normalize = norm_normalize.squeeze(-1)
#     norm_normalize = norm_normalize.permute(0, 3, 1, 2)
    
#     return norm_normalize
    


# # Occlusion-Aware Depth Estimation with Adaptive Normal Constraints 2020 ECCV
# def normal2color(normal_map):
#     """
#     colorize normal map
#     :param normal_map: range(-1, 1)
#     :return:
#     """
#     tmp = normal_map / 2. + 0.5  # mapping to (0, 1)
#     color_normal = (tmp * 255).astype(np.uint8)

#     return color_normal


# def depth2color(depth, MIN_DEPTH=0.3, MAX_depth=8.0):
#     """
#     colorize depth map
#     :param depth:
#     :return:
#     """
#     depth_clip = deepcopy(depth)
#     depth_clip[depth_clip < MIN_DEPTH] = 0
#     depth_clip[depth_clip > MAX_depth] = 0
#     normalized = (depth_clip - MIN_DEPTH) / (MAX_depth - MIN_DEPTH) * 255.0
#     normalized = [normalized, normalized, normalized]
#     normalized = np.stack(normalized, axis=0)
#     normalized = np.transpose(normalized, (1, 2, 0))
#     normalized = normalized.astype(np.uint8)

#     return cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)


# class Depth2normal(nn.Module):
#     def __init__(self, k_size=9):
#         """
#         convert depth map to point cloud first, and then calculate normal map
#         :param k_size: the kernel size for neighbor points
#         """
#         super(Depth2normal, self).__init__()
#         self.k_size = k_size

#     def forward(self, depth, intrinsic_inv, instance_segs=None, planes_num=None):
#         """

#         :param depth: [B, H, W]
#         :param intrinsic_inv: [B, 3, 3]
#         :param instance_segs: [B, 20, h, w] stores "planes_num" plane instance seg (bool map)
#         :param planes_num: [B]
#         :return:
#         """
#         device = depth.get_device()
#         b, h, w = depth.shape
#         points = pixel2cam(depth, intrinsic_inv)  # (b, c, h, w)

#         valid_condition = ((depth > 0) & (depth < 10.0)).type(torch.FloatTensor)
#         valid_condition = valid_condition.unsqueeze(1)  # (b, 1, h, w)

#         unford = torch.nn.Unfold(kernel_size=(self.k_size, self.k_size), padding=self.k_size // 2, stride=(1, 1))
#         torch_patches = unford(points)  # (N,C×∏(kernel_size),L)
#         matrix_a = torch_patches.view(-1, 3, self.k_size * self.k_size, h, w)
#         matrix_a = matrix_a.permute(0, 3, 4, 2, 1)  # (b, h, w, self.k_size*self.k_size, 3)

#         valid_condition = unford(valid_condition)
#         valid_condition = valid_condition.view(-1, 1, self.k_size * self.k_size, h, w)
#         valid_condition = valid_condition.permute(0, 3, 4, 2, 1)  # (b, h, w, self.k_size*self.k_size, 1)
#         valid_condition_all = valid_condition.repeat([1, 1, 1, 1, 3])
#         valid_condition_all = (valid_condition_all > 0.5).to(device)

#         matrix_a_zero = torch.zeros_like(matrix_a)
#         matrix_a_valid = torch.where(valid_condition_all, matrix_a, matrix_a_zero)
#         matrix_a_trans = torch.transpose(matrix_a_valid, 3, 4).view(-1, 3, self.k_size * self.k_size).to(device)
#         matrix_a_valid = matrix_a_valid.view(-1, self.k_size * self.k_size, 3).to(device)
#         matrix_b = torch.ones([b, h, w, self.k_size * self.k_size, 1]).view([-1, self.k_size * self.k_size, 1]).to(
#             device)

#         point_multi = torch.bmm(matrix_a_trans, matrix_a_valid).to(device)

#         matrix_det = point_multi.det()

#         inverse_condition_invalid = torch.isnan(matrix_det) | (matrix_det < 1e-5)
#         inverse_condition_valid = (inverse_condition_invalid == False)
#         inverse_condition_valid = inverse_condition_valid.unsqueeze(1).unsqueeze(2)
#         inverse_condition_all = inverse_condition_valid.repeat([1, 3, 3]).to(device)

#         diag_constant = torch.ones([3])
#         diag_element = torch.diag(diag_constant)
#         diag_element = diag_element.unsqueeze(0).to(device)
#         diag_matrix = diag_element.repeat([inverse_condition_all.shape[0], 1, 1])

#         inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
#         inv_matrix = torch.inverse(inversible_matrix)

#         generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b).squeeze(-1)  # [-1, 3]
#         generated_norm_normalize = generated_norm / (torch.norm(generated_norm, dim=1, keepdim=True) + 1e-5)

#         generated_norm_normalize = generated_norm_normalize.view(b, h, w, 3).type(torch.FloatTensor).to(device)

#         if planes_num is not None:

#             instance_segs = instance_segs.unsqueeze(-1).repeat(1, 1, 1, 1, 3)  # [b, 20, h, w, 3]
#             generated_norm_normalize_new = []

#             loss = 0

#             for b_i in range(len(planes_num)):

#                 generated_norm_normalize_bi = generated_norm_normalize[b_i:b_i + 1]
#                 zeros_tensor = torch.zeros_like(generated_norm_normalize_bi)
#                 # use plane segs to regularize the normal values
#                 for i in range(planes_num[b_i]):
#                     instance_seg = instance_segs[b_i:b_i + 1, i, :, :, :]  # [1, h, w, 3]
#                     nominator = torch.sum(
#                         torch.mul(generated_norm_normalize_bi,
#                                   instance_seg.type(torch.FloatTensor).to(device)).view(1, h * w, 3), dim=1)
#                     denominator = torch.sum(
#                         instance_seg[:, :, :, 0].view(1, h * w), dim=1, keepdim=True).type(torch.FloatTensor).to(device)

#                     normal_regularized = (nominator / denominator).unsqueeze(1).unsqueeze(1).repeat(
#                         1, h, w, 1)
#                     normal_original = torch.where(instance_seg, generated_norm_normalize_bi, zeros_tensor)

#                     similarity = torch.nn.functional.cosine_similarity(
#                         normal_regularized.view(-1, 3), normal_original.view(-1, 3), dim=1)

#                     loss += torch.mean(1 - similarity)

#                     generated_norm_normalize_bi = torch.where(instance_seg, normal_regularized,
#                                                               generated_norm_normalize_bi)
#                 generated_norm_normalize_new.append(generated_norm_normalize_bi)
#             generated_norm_normalize = torch.stack(generated_norm_normalize_new, dim=0).squeeze(1)
#             return generated_norm_normalize.permute(0, 3, 1, 2), loss, points

#         return generated_norm_normalize.permute(0, 3, 1, 2), points


# pixel_coords = None


# def set_id_grid(depth):
#     global pixel_coords
#     b, h, w = depth.size()
#     i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w)).type_as(depth)  # [1, H, W]
#     j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w)).type_as(depth)  # [1, H, W]
#     ones = Variable(torch.ones(1, h, w)).type_as(depth)

#     pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


# def check_sizes(input, input_name, expected):
#     condition = [input.ndimension() == len(expected)]
#     for i, size in enumerate(expected):
#         if size.isdigit():
#             condition.append(input.size(i) == int(size))
#     assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected),
#                                                                               list(input.size()))


# def pixel2cam(depth, intrinsics_inv):
#     global pixel_coords
#     """Transform coordinates in the pixel frame to the camera frame.
#     Args:
#         depth: depth maps -- [B, H, W]
#         intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
#     Returns:
#         array of (u,v,1) cam coordinates -- [B, 3, H, W]
#     """
#     device = depth.get_device()
#     b, h, w = depth.size()
#     if (pixel_coords is None) or pixel_coords.size(2) < h:
#         set_id_grid(depth)
#     current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b, 3,
#                                                                                            -1).to(device)  # [B, 3, H*W]
#     cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
#     return cam_coords * depth.unsqueeze(1)


# def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
#     """Transform coordinates in the camera frame to the pixel frame.
#     Args:
#         cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
#         proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
#         proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
#     Returns:
#         array of [-1,1] coordinates -- [B, 2, H, W]
#     """
#     b, _, h, w = cam_coords.size()
#     cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
#     if proj_c2p_rot is not None:
#         pcoords = proj_c2p_rot.bmm(cam_coords_flat)
#     else:
#         pcoords = cam_coords_flat
#     max_value = torch.max(cam_coords_flat)

#     if proj_c2p_tr is not None:
#         pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
#     X = pcoords[:, 0]
#     Y = pcoords[:, 1]
#     Z = pcoords[:, 2].clamp(min=1e-3)

#     X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
#     Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
#     if padding_mode == 'zeros':
#         X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
#         X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
#         Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
#         Y_norm[Y_mask] = 2

#     pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
#     return pixel_coords.view(b, h, w, 2)


# def inverse_warp(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
#     """
#     Inverse warp a source(right) image to the target(reference, left) image plane.

#     Args:
#         feat: the source(right) feature (where to sample pixels) -- [B, CH, H, W]
#         depth: depth map of the target(reference, left) image -- [B, H, W]
#         pose: 6DoF pose parameters from target(reference, left) to source(right) -- [B, 6]
#               right2left = pose_src @ np.linalg.inv(pose_tgt)
#         intrinsics: source(right) camera intrinsic matrix -- [B, 3, 3]
#         intrinsics_inv: inverse of the target(reference) intrinsic matrix -- [B, 3, 3]
#     Returns:
#         Source(right) image warped to the target(reference, left) image plane  -- [B, CH, H, W]
#     """
#     check_sizes(depth, 'depth', 'BHW')
#     check_sizes(pose, 'pose', 'B34')
#     check_sizes(intrinsics, 'intrinsics', 'B33')
#     check_sizes(intrinsics_inv, 'intrinsics', 'B33')

#     assert (intrinsics_inv.size() == intrinsics.size())
    
#     device = intrinsics_inv.get_device()

#     batch_size, _, feat_height, feat_width = feat.size()

#     cam_coords = pixel2cam(depth, intrinsics_inv)

#     pose_mat = pose
#     pose_mat = pose_mat.to(device)

#     # Get projection matrix for tgt camera frame to source pixel frame
#     proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

#     src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:],
#                                  padding_mode)  # [B,H,W,2]
#     projected_feat = torch.nn.functional.grid_sample(feat, src_pixel_coords, padding_mode=padding_mode)

#     return projected_feat


def vis_normal(normal):
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    @para normal: surface normal, [h, w, 3], numpy.array
    """
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])

def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))
    
def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused