# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # 721/1242=0.58, 721/375=1.92
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # P_rect(Projection Matrix after Rectification)
        # self.cam_k = {
        #     '2011_09_26' : np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], 
        #                              [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        #                              [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
        #                              [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),

        #     '2011_09_28' : np.array([[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01], 
        #                     [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01], 
        #                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03],
        #                              [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
        #     '2011_09_29' : np.array([[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], 
        #                     [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
        #                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03],
        #                              [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
        #     '2011_09_30' : np.array([[7.070912e+02, 0.000000e+00, 6.018873e+02, 4.688783e+01], 
        #                     [0.000000e+00, 7.070912e+02, 1.831104e+02, 1.178601e-01], 
        #                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 6.203223e-03],
        #                              [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
        #     '2011_10_03' : np.array([[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01], 
        #                     [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01], 
        #                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03],
        #                              [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
        # }
        
        # S_rect(Rectified Size)
        self.full_res_shape = (1242, 375)
        # image_00是左侧灰度，image_01右侧灰度，image_02左侧彩色，image_03右侧彩色
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    
    def index_to_folder_and_frame_idx(self, index):
        """
        Convert dataset index to folder name, frame index, and metadata
            
        Parses the string format of `self.filenames[index]` to extract:
        - Folder path (e.g., "2011_09_26/2011_09_26_drive_0022_sync")
        - Frame index (e.g., 473)
        - indicate direction (e.g., "r" or "l")
        
        Example input format:
        "2011_09_26/2011_09_26_drive_0022_sync 473 r"
        
        Args:
            index (int): Dataset sample index
        
        Returns:
            tuple: A tuple containing:
                - folder_name (str): Path to data folder
                - frame_idx (int): Frame sequence number
                - metadata (str): Additional metadata (e.g., direction flag)
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIStereoDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIStereoDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        if '/' in folder:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        else:
            f_str = "{:06d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                "/pcalab/tmp/KittiOdometry/color",
                "sequences/{:02d}".format(int(folder)),
                "image_{}".format(self.side_map[side]),
                f_str)
        return image_path


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        # f_str = "frame_{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            # self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
