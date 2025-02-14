# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from utils import readlines
import matplotlib.pyplot as plt
import cv2
from layers import *

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--data_path', type=str, default='assets/test_image.jpg',
                        help='path to a test image or folder of images')
    parser.add_argument('--save_path', type=str, default='results/GS-Depth_FiT3DAlter',
                        help='path to a test image or folder of images')
    parser.add_argument('--model_name', type=str, default='RA-Depth',
                        help='name of a pretrained model to use',
                        )
                        # choices = [
                        #     "RA-Depth"]
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--eval_split",
                        type=str,
                        help="which training split to use",
                        choices=["eigen", "eigen_wu", "eigen_small", "eigen_zhou", "eigen_full", "odom", "benchmark", "tmp", "eigen_and_odom", "nyu"],
                        default="eigen")
    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=192)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)
    parser.add_argument("--load_weights_folder",
                        type=str,
                        help="name of model to load")
    parser.add_argument("--use_gs",
                        help="if set, uses 3d gs",
                        action="store_true")
    parser.add_argument("--gs_scale",
                        type=int,
                        help="use gs at one scale",
                        default=32)

    return parser.parse_args()


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

def get_image_path(data_path, folder, frame_index, side, img_ext):
        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        f_str = "{:010d}.{}".format(int(frame_index), img_ext)
        # f_str = "frame_{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            # self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
            data_path, folder, "image_0{}/data".format(side_map[side]), f_str)
        return image_path

def simple(args):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    K = np.array([[0.58, 0, 0.5, 0],
                [0, 1.92, 0.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

    encoder = networks.hrnet18(False)
    depth_decoder = networks.DepthDecoder_MSF_GS_FiTAlter(encoder.num_ch_enc, 
                                                    [0], 
                                                    num_output_channels=1,
                                                    use_gs=args.use_gs, 
                                                    gs_scale=args.gs_scale, 
                                                    min_depth=args.min_depth, 
                                                    max_depth=args.max_depth, 
                                                    height=args.height, 
                                                    width=args.width)

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    # encoder = networks.ResnetEncoder(18, False)


    model_path = args.load_weights_folder
    print("-> Loading model from ", model_path)
    checkpoint = torch.load(args.load_weights_folder)
    encoder_dict = checkpoint["encoder"]
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_dcit = checkpoint["depth"]
    depth_decoder.load_state_dict(depth_dcit)

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()
    # encoder.to(device)
    # encoder.eval()
    # depth_decoder.to(device)
    # depth_decoder.eval()

    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))

    feed_width = args.width
    feed_height = args.height

    # K[0, :] *= args.width
    # K[1, :] *= args.height

    # inv_K = np.linalg.pinv(K)
    # K = torch.from_numpy(K)
    # inv_K = torch.from_numpy(inv_K)
    # K = K.unsqueeze(0)
    # inv_K = inv_K.unsqueeze(0)
    # # print(inv_K.shape)
    # K = K.to(device)
    # inv_K = inv_K.to(device)


    inv_K_data = []
    K_data = []
    if args.gs_scale == 0:
        for scale in range(1, 6):
            K_new = K
            K_new[0, :] *= args.width // (2 ** scale)
            K_new[1, :] *= args.height // (2 ** scale)
            inv_K_new = np.linalg.pinv(K_new)
            K_new = torch.from_numpy(K_new).unsqueeze(0)
            inv_K_new = torch.from_numpy(inv_K_new).unsqueeze(0)
            K_new = K_new.to(device)
            inv_K_new = inv_K_new.to(device)
            K_data.append(K_new)
            inv_K_data.append(inv_K_new)


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # FINDING INPUT IMAGES
    if os.path.isfile(args.data_path):
        # Only testing on a single image
        paths = [args.data_path]
        output_directory = os.path.dirname(args.data_path)
    elif os.path.isdir(args.data_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.data_path, '*.{}'.format(args.ext)))
        output_directory = args.data_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.data_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # for idx, image_path in enumerate(paths):
        for idx in range(len(filenames)):
            folder, frame_index, side = filenames[idx].split()
            image_path = get_image_path(args.data_path, folder, frame_index, side, args.ext)
            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            target_ = input_image
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)

            # input_image = input_image.resize((640, 192), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            # outputs = depth_decoder(features)
            outputs = depth_decoder(features, inv_K_data, K_data)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            
            
            feat_before = outputs[("gs_feature_before", 2)]
            feat_after = outputs[("gs_feature_after", 2)]

            # # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            result_save_path = os.path.join(args.save_path, folder)
            if not os.path.exists(result_save_path):
                os.makedirs(result_save_path)
            
            target_.save(os.path.join(result_save_path, "{}_image.png".format(output_name)))

            feat_before = feat_before[0, :3].squeeze().cpu().numpy() * 255
            feat_before = np.transpose(feat_before, (1, 2, 0)).astype(np.uint8)
            feat_before = Image.fromarray(feat_before)
            name_dest_im = os.path.join(result_save_path, "{}_feat_before.png".format(output_name))
            feat_before.save(name_dest_im)
            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))
            
            feat_after = feat_after[0, :3].squeeze().cpu().numpy() * 255
            feat_after = np.transpose(feat_after, (1, 2, 0)).astype(np.uint8)
            feat_after = Image.fromarray(feat_after)
            name_dest_im = os.path.join(result_save_path, "{}_feat_after.png".format(output_name))
            feat_after.save(name_dest_im)
            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma') #magma
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join(result_save_path, "{}_disp.png".format(output_name))
            im.save(name_dest_im)
            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')

if __name__ == '__main__':
    args = parse_args()
    simple(args)


#CUDA_VISIBLE_DEVICES=0 python simpleTest.py --image_path /test/monodepth2-master/assets/test.png --model_name RA-Depth


# CUDA_VISIBLE_DEVICES=6 python simpleTest.py --image_path /data0/wuhaifeng/PytorchCode/DepthEstimation/RA-Depth-main/assets/2011_09_26_drive_0002_sync/0000000000.png --model_name /data0/wuhaifeng/PytorchCode/DepthEstimation/RA-Depth-main/models/RA-Depth_multi_gpu_MultiTask_TwoDepth_NormalSmoothLoss_NDDepth/models/weights_19
