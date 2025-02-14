from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from ptflops import get_model_complexity_info
from thop import profile
import time

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

# data_len = 0

#CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /models/RA-Depth/ --eval_mono --height 192 --width 640 --scales 0 --data_path /datasets/Kitti/Kitti_raw_data --png

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

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # MIN_DEPTH = 40
    # MAX_DEPTH = 50

    device = torch.device("cuda")

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isfile(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        # encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        # decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        # encoder_dict = torch.load(encoder_path, map_location=device)

        checkpoint = torch.load(opt.load_weights_folder)

        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                     encoder_dict['height'], encoder_dict['width'],
                                                     [0], 4,
                                                     is_train=False)
        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False, img_ext='.png', gs_scale=opt.gs_scale)

        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # encoder = networks.ResnetEncoder(opt.num_layers, False)
        encoder = networks.hrnet18(False)

        # encoder = networks.ResnetEncoder(opt.num_layers, False)
        # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales)
        
        # depth_decoder = networks.DepthDecoder_MSF_BS_ND2(encoder.num_ch_enc, [0], 
        #                                                 num_output_channels=1, 
        #                                                 min_depth=opt.min_depth, 
        #                                                 max_depth=opt.max_depth)
        # depth_decoder = networks.DepthDecoder_MSF_BS_ND_INDL_SNL_1conv1x1_32(encoder.num_ch_enc, [0], 
        # depth_decoder = networks.DepthDecoder_MSF_BS_ND_INDL_SNL_64_32(encoder.num_ch_enc, [0], 
        # depth_decoder = networks.DepthDecoder_MSF_BS_ND_Inter_Intra(encoder.num_ch_enc, [0], 
        #                                                 num_output_channels=1, 
        #                                                 min_depth=opt.min_depth, 
        #                                                 max_depth=opt.max_depth,
        #                                                 height=opt.height,
        #                                                 width=opt.width)
        depth_decoder = networks.DepthDecoder_MSF_GS_FiT_Inter(encoder.num_ch_enc, 
                                                    [0], 
                                                    num_output_channels=1,
                                                    use_gs=opt.use_gs, 
                                                    gs_scale=opt.gs_scale, 
                                                    min_depth=opt.min_depth, 
                                                    max_depth=opt.max_depth, 
                                                    height=opt.height, 
                                                    width=opt.width)
        # depth_decoder = networks.DepthDecoder_MSF(encoder.num_ch_enc, 
        #                                                     [0], 
        #                                                     num_output_channels=1, 
        #                                                     min_depth=opt.min_depth, 
        #                                                     max_depth=opt.max_depth
        #                                                     )

        # model_dict = encoder.state_dict()
        # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        # depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        # encoder.to(device)
        # encoder.eval()
        # depth_decoder.to(device)
        # depth_decoder.eval()
        encoder_dict = checkpoint["encoder"]
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_dcit = checkpoint["depth"]
        depth_decoder.load_state_dict(depth_dcit)

        encoder.to(device)
        encoder.eval()
        depth_decoder.to(device)
        depth_decoder.eval()

        print("data length: ", len(dataset))

        encoder_total_params = sum(p.numel() for p in encoder.parameters())
        print('Total parameters: %.6fM (%d)' % (encoder_total_params / 1e6, encoder_total_params))
        encoder_total_trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad)
        print('Total_trainable_params: %.6fM (%d)' % (encoder_total_trainable_params / 1e6, encoder_total_trainable_params))

        decoder_total_params = sum(p.numel() for p in depth_decoder.parameters())
        print('Total parameters: %.6fM (%d)' % (decoder_total_params / 1e6, decoder_total_params))
        decoder_total_trainable_params = sum(
            p.numel() for p in depth_decoder.parameters() if p.requires_grad)
        print('Total_trainable_params: %.6fM (%d)' % (decoder_total_trainable_params / 1e6, decoder_total_trainable_params))

        network_total_params = encoder_total_params + decoder_total_params
        network_total_trainable_params = encoder_total_trainable_params + decoder_total_trainable_params
        print('Network total parameters: %.6fM (%d)' % (network_total_params / 1e6, network_total_params))
        print('Network total_trainable_params: %.6fM (%d)' % (network_total_trainable_params / 1e6, network_total_trainable_params))


        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            opt.width, opt.height))

        with torch.no_grad():
            for data in dataloader:
                # input_color = data[("color", 0, 0)].to(device)
                input_color = data[("color", 0, 0)].to(device)
                inv_K = []
                K = []
                if opt.gs_scale != 0:
                    inv_K.append(data[("inv_K_gs", opt.gs_scale)].to(device))
                    K.append(data[("K_gs", opt.gs_scale)].to(device))
                else:
                    inv_K.append(data[("inv_K_gs", 1)].to(device))
                    inv_K.append(data[("inv_K_gs", 2)].to(device))
                    inv_K.append(data[("inv_K_gs", 3)].to(device))
                    inv_K.append(data[("inv_K_gs", 4)].to(device))
                    inv_K.append(data[("inv_K_gs", 5)].to(device))

                    K.append(data[("K_gs", 1)].to(device))
                    K.append(data[("K_gs", 2)].to(device))
                    K.append(data[("K_gs", 3)].to(device))
                    K.append(data[("K_gs", 4)].to(device))
                    K.append(data[("K_gs", 5)].to(device))

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color), inv_K, K, 0)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu().numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        if gt_depth is None:
            continue
        gt_height, gt_width = gt_depth.shape[:2] #[375, 1242]

        pred_disp = pred_disps[i]
        # print(pred_disp.shape)
        pred_disp = cv2.resize(pred_disp.squeeze(), (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            # print(mask.shape)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        if gt_depth.shape[0] == 0:
            continue

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        ratio_max = ratios.max()
        ratio_min = ratios.min()
        ratio_mean = ratios.mean()
        print("Scaling ratios | mean: {:0.3f} | min: {:0.3f} | max: {:0.3f}".format(ratio_mean, ratio_min, ratio_max))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    begain_time = time.time()
    options = MonodepthOptions()
    evaluate(options.parse())
    end_time = time.time()
    print("Average time consumption: ", (end_time - begain_time) / 697)
    # print("data len: ", data_len)

# python evaluate_depth.py --load_weights_folder models/RA-Depth/ --eval_mono --height 192 --width 640 --scales 0 --data_path /data/home/wuhaifeng/dataset/KITTI/raw_data --png
# python evaluate_depth.py --load_weights_folder models/RA-Depth/ --eval_mono

# python evaluate_depth.py --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/RA-Depth-main/models/RA-Depth-retrain/models/weights_19 --eval_mono --height 192 --width 640 --scales 0 --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --png --eval_split eigen

# python evaluate_depth.py --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/RA-Depth-main/models/RA-Depth-retrain_eigen_wu_5frame/models/weights_19 --eval_mono --height 192 --width 640 --scales 0 --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --png --eval_split eigen_wu

#  python evaluate_depth.py --load_weights_folder /data0/wuhaifeng/PytorchCode/DepthEstimation/RA-Depth-main/models/RA-Depth-retrain_ResNetEncoder/models/weights_19 --eval_mono --height 192 --width 640 --scales 0 --data_path /data0/wuhaifeng/dataset/KITT_dataset/raw_data --png --eval_split eigen