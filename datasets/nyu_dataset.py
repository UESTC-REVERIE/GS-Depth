from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class NYUDataset(MonoDataset):
    """NYU dataset loaders
    """

    edge_crop = 16
    full_res_shape = (640 - 2*edge_crop, 480 - 2*edge_crop)
    default_crop = [40 - edge_crop, 601 - edge_crop, 44 - edge_crop, 471 - edge_crop]
    min_depth = 0.01
    max_depth = 10.0

    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size

        w, h = self.full_res_shape

        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        cx = (3.2558244941119034e+02 - self.edge_crop) / w
        cy = (2.5373616633400465e+02 - self.edge_crop) / h

        self.K = np.array([[fx, 0, cx, 0],
                           [0, fy, cy, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = line[1]
        if not self.is_train:
            depth_filename = os.path.join(self.data_path, scene_name,
                                          "{:05d}".format(int(frame_index)) + "_depth.png")
        else:
            depth_filename = os.path.join(self.data_path, scene_name, frame_index+".png")
        return os.path.isfile(depth_filename)
    
    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) >= 2:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side


    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        color = color.crop((self.edge_crop, self.edge_crop, 640-self.edge_crop, 480-self.edge_crop))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    


class NYURAWDataset(NYUDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYURAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        if self.is_test:
            image_path = os.path.join(
                self.data_path, folder, "{:05d}".format(frame_index)+"_colors.png")
        else:
            image_path = os.path.join(
                self.data_path, folder, str(frame_index) + ".jpg")
        return image_path
    
    def get_depth(self, folder, frame_index, side, do_flip):
        if self.is_test:
            depth_path = os.path.join(
                self.data_path, folder, "{:05d}".format(frame_index) + "_depth.png")

            depth_gt = pil.open(depth_path)
            depth_gt = depth_gt.crop((self.edge_crop, self.edge_crop, 640 - self.edge_crop, 480 - self.edge_crop))

            depth_gt = np.array(depth_gt).astype(np.float32) / 1000
        # elif self.is_val:
        #     depth_path = os.path.join(
        #         self.data_path, folder, str(frame_index) + ".png")

        #     depth_gt = pil.open(depth_path)
        #     depth_gt = depth_gt.crop((self.edge_crop, self.edge_crop, 640 - self.edge_crop, 480 - self.edge_crop))

        #     depth_gt = np.array(depth_gt).astype(np.float32) / 1000
        else:
            depth_path = os.path.join(
                self.data_path, folder, str(frame_index) + ".png")

            depth_gt = pil.open(depth_path)
            depth_gt = depth_gt.crop((self.edge_crop, self.edge_crop, 640 - self.edge_crop, 480 - self.edge_crop))

            depth_gt = np.array(depth_gt).astype(np.float32) / 25.6

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
