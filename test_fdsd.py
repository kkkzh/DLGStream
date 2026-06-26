#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import random
import os
import sys
import copy
from time import time
from random import randint
import pickle
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import cv2
import ffmpeg
import matplotlib.pyplot as plt

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from scene import Scene
from scene.fdsd_gaussian_model import GaussianModel
from gaussian_renderer import stream_render_fisd
from scene.dataset import MultiEpochsDataLoader
from gaussian_renderer import fps_render, stream_render_eval, stream_render_eval_v1, stream_lang_render_eval

from utils.loader_utils import FineSampler, get_stamp_list
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from utils.image_utils import psnr
from utils.timer import Timer
from lpipsPyTorch import lpips

from compression.compression_exp import run_compressions, run_decompressions
from compression.compression_exp import decompress_attr, decompress_attr_2, decompress_attr_3
from compression.jpeg_xl import JpegXlCodec
from compression.png import PNGCodec

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def measure_decode_latency(input_file):
    try:
        probe = ffmpeg.probe(input_file)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        # 启动 ffmpeg 进程
        process = (
            ffmpeg
            .input(input_file)
            .output('pipe:', format='rawvideo', pix_fmt='gray16be')
            .run_async(pipe_stdout=True, quiet=True)
        )

        width = int(video_info['width'])
        height = int(video_info['height'])

        decode_start_time = time()
        frames = []
        while True:
            frame_data = process.stdout.read(width * height * 2)

            if not frame_data:
                break

            frame = np.frombuffer(frame_data, dtype=np.uint16)
            frame = frame.newbyteorder('>')
            frame = frame.reshape((width, height))
            frames.append(frame)

        decode_end_time = time()
        print(f"Decoding latency: {(decode_end_time - decode_start_time) / 60:.6f}s")
        return frames

    except Exception as e:
        print(f"Error: {e}")

def get_file_size_in_kB(file_path):
    """Return the file size in kilobytes (kB)."""
    size_in_bytes = os.path.getsize(file_path)
    # Divide by 1024 to convert from bytes to kilobytes
    size_in_kB = size_in_bytes / 1024
    return round(size_in_kB, 5)

def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def opacity_vis(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene: Scene, save_path, gopid, **kwargs):
    image_output_path = os.path.join(save_path, 'opacity_vis')
    os.makedirs(image_output_path, exist_ok=True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if gopid == 0:
        stage = "refine"
    elif gopid > 0:
        stage = "following"
    else:
        raise ValueError

    assert kwargs['include_feature']

    # precompute keyframe
    if not kwargs['include_feature']:
        xyz_keyframes = []
        rot_keyframes = []
        sca_keyframes = []
        for idx in range(gaussians.keyframe_num):
            feat = gaussians._point_feats[:, idx, :]
            dy_xyz = gaussians.get_deform_mlp(feat[:, :12])
            # dy_xyz = gaussians.get_deform_mlp(feat)
            xyz_keyframes.append(dy_xyz)
            dcov = gaussians.get_cov_mlp(feat[:, :12])
            # dcov = gaussians.get_cov_mlp(feat)
            dy_rot = gaussians.get_rotation_dynamic_ori + dcov[:, 3:]
            rot_keyframes.append(dy_rot)
            if gopid > 0:
                dy_sca = gaussians.get_scaling_dynamic_ori + dcov[:, :3]
                sca_keyframes.append(dy_sca)
        gaussians._xyz_keyframe = torch.stack(xyz_keyframes, dim=1)
        gaussians._rotation_keyframe = torch.stack(rot_keyframes, dim=1)
        if gopid > 0:
            gaussians._scaling_keyframe = torch.stack(sca_keyframes, dim=1)

    # save static opacity difference
    if gopid == 0:
        opacity = gaussians.get_opacity
        opacity = opacity.detach().cpu()
        grid_sidelen = int(math.sqrt(opacity.shape[0]))
        opacity = opacity.reshape((grid_sidelen, grid_sidelen, -1))
        image = torch.abs(opacity[:, :, 0] - opacity[:, :, 1])
        image = image.squeeze()
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.colorbar(label='Difference Value')
        plt.tight_layout()
        plt.imsave(os.path.join(image_output_path, "diff_heatmap_norm.png"), image, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.close()

        opacity = gaussians._opacity
        opacity = opacity.detach().cpu()
        opacity = opacity.reshape((grid_sidelen, grid_sidelen, -1))
        image = torch.abs(opacity[:, :, 0] - opacity[:, :, 1])
        image = image.squeeze()
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.colorbar(label='Difference Value')
        plt.tight_layout()
        plt.imsave(os.path.join(image_output_path, "diff_heatmap.png"), image, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.close()

        opacity = torch.cat([gaussians.get_opacity_ori, gaussians.get_opacity_dynamic_ori], dim=0)
        opacity = gaussians.opacity_activation(opacity)
        opacity = opacity.detach().cpu()
        opacity_color = opacity[:, 0:1].flatten().numpy()
        opacity_feat = opacity[:, 1:2].flatten().numpy()

        bins = np.linspace(0, 1, 11)

        plt.figure(figsize=(10, 6))
        weights_color = np.ones_like(opacity_color) / len(opacity_color)
        weights_feat = np.ones_like(opacity_feat) / len(opacity_feat)
        counts, _, _ = plt.hist([opacity_color, opacity_feat], bins=bins, weights=[weights_color, weights_feat], label=['color opacity', 'language opacity'],
                                color=['skyblue', 'green'], edgecolor='black', alpha=0.8)
        plt.xticks(bins, fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')
        plt.xlabel('Opacity range', fontsize=16, fontweight='bold')
        plt.ylabel('Ratio', fontsize=16, fontweight='bold')
        plt.legend(fontsize=16)
        plt.savefig(os.path.join(image_output_path, "opacity_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print('draw static opacity difference image finished!')

    test_cams = scene.getTestCameras()
    test_cams.dataset.read_clip_features(False)
    test_view_num = len(test_cams)
    viewpoint_stack_loader = MultiEpochsDataLoader(test_cams, batch_size=1, shuffle=False, num_workers=6, collate_fn=list)

    progress_bar = tqdm(range(0, test_view_num), desc=f"Evaluating gop{gopid}")

    diff_list = []
    for idx, viewpoint_cams in enumerate(viewpoint_stack_loader):
        viewpoint_cam = viewpoint_cams[0]
        viewpoint_cam.cuda()
        if kwargs['include_feature']:
            outputs = stream_render_fisd(viewpoint_cam, gaussians, pipe, background, stage=stage, evaluation=True, include_feature=True, return_opacity=True)
        else:
            assert AssertionError

        diff = torch.abs(outputs[:,0:1] - outputs[:,1:2]).mean().double().item()
        diff_list.append(diff)
        bar_pd = {"diff": f"{diff:.{4}f}"}
        progress_bar.set_postfix(bar_pd)
        progress_bar.update(1)

        torch.cuda.empty_cache()

    progress_bar.close()

    return diff_list

def reconstruction_testing(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene: Scene, save_path, gopid, **kwargs):
    image_output_path = os.path.join(save_path, 'images')
    # image_output_path = os.path.join('/home/kzh/3DGS/work4-results/lang_so', 'sear_steak')
    os.makedirs(image_output_path, exist_ok=True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if gopid == 0:
        stage = "refine"
    elif gopid > 0:
        stage = "following"
    else:
        raise ValueError

    # precompute keyframe
    if not kwargs['include_feature']:
        xyz_keyframes = []
        rot_keyframes = []
        sca_keyframes = []
        for idx in range(gaussians.keyframe_num):
            feat = gaussians._point_feats[:, idx, :]
            dy_xyz = gaussians.get_deform_mlp(feat[:, :12])
            # dy_xyz = gaussians.get_deform_mlp(feat)
            xyz_keyframes.append(dy_xyz)
            dcov = gaussians.get_cov_mlp(feat[:, :12])
            # dcov = gaussians.get_cov_mlp(feat)
            dy_rot = gaussians.get_rotation_dynamic_ori + dcov[:, 3:]
            rot_keyframes.append(dy_rot)
            if gopid > 0:
                dy_sca = gaussians.get_scaling_dynamic_ori + dcov[:, :3]
                sca_keyframes.append(dy_sca)
        gaussians._xyz_keyframe = torch.stack(xyz_keyframes, dim=1)
        gaussians._rotation_keyframe = torch.stack(rot_keyframes, dim=1)
        if gopid > 0:
            gaussians._scaling_keyframe = torch.stack(sca_keyframes, dim=1)

    test_cams = scene.getTestCameras()
    test_cams.dataset.read_clip_features(False)
    test_view_num = len(test_cams)
    viewpoint_stack_loader = MultiEpochsDataLoader(test_cams, batch_size=1, shuffle=False, num_workers=6, collate_fn=list)

    progress_bar = tqdm(range(0, test_view_num), desc=f"Evaluating gop{gopid}")

    psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
    psnr_list = []
    for idx, viewpoint_cams in enumerate(viewpoint_stack_loader):
        viewpoint_cam = viewpoint_cams[0]
        viewpoint_cam.cuda()
        if kwargs['include_feature']:
            outputs = stream_render_fisd(viewpoint_cam, gaussians, pipe, background, stage=stage, evaluation=True, include_feature=True)
            image = torch.clamp(outputs["render"], 0.0, 1.0)
        else:
            outputs = stream_render_eval(viewpoint_cam, gaussians, pipe, background, stage=stage, evaluation=True)
            # outputs = stream_render_eval_v1(viewpoint_cam, gaussians, pipe, background, stage=stage, evaluation=True)
            image = torch.clamp(outputs[0], 0.0, 1.0)

        gt_image = torch.clamp(viewpoint_cam.original_image, 0.0, 1.0)

        psnr_ = psnr(image, gt_image, mask=None).mean().double().item()
        psnr_test += psnr_
        psnr_list.append(psnr_)
        bar_pd = {"psnr": f"{psnr_:.{2}f}"}
        if not args.disable_ssim:
            ssim_ = ssim(image, gt_image).double().item()
            ssim_test += ssim_
            bar_pd["ssim"] = f"{ssim_:.{2}f}"
        if not args.disable_lpips:
            lpips_ = lpips(image, gt_image, net_type='vgg').double().item()
            lpips_test += lpips_
            bar_pd["lpips"] = f"{lpips_:.{2}f}"
        progress_bar.set_postfix(bar_pd)
        progress_bar.update(1)

        timestamp = viewpoint_cam.time * args.gop
        img_idx = int(timestamp) + gopid * args.gop
        saved_idx = [0, 1, 10, 20, 30, 40, 60, 65, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 267, 280, 299]
        if args.write_images:
            save_images = []
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)
            save_images.append(image)

            if kwargs['include_feature'] and args.write_lang:
                lang = outputs["language_feature"].permute(1, 2, 0).detach().cpu().numpy()
                for level in range(0, lang.shape[2], 3):
                    _lang = (lang[:, :, level: level+3] * 65535).astype(np.uint16)
                    save_images.append(_lang)

            for idx, image in enumerate(save_images):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(image_output_path, '{0:05d}'.format(img_idx) + f"_{idx}" + ".png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        if kwargs['include_feature'] and img_idx in saved_idx:
            if not os.path.exists(os.path.join(image_output_path, '{0:05d}'.format(img_idx) + ".npy")):
                lang = outputs["language_feature"].permute(1, 2, 0).detach().cpu().numpy()
                np.save(os.path.join(image_output_path, '{0:05d}'.format(img_idx) + ".npy"), lang)

        torch.cuda.empty_cache()

    progress_bar.close()
    psnr_test /= test_view_num
    ssim_test /= test_view_num
    lpips_test /= test_view_num
    print(f"PSNR={psnr_test:.5f}, SSIM={ssim_test:.5f}, LPIPS={lpips_test:.5f}")
    return psnr_test, ssim_test, lpips_test, psnr_list


def render_nogt(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene: Scene, gop, gopid, **kwargs):
    print("point nums:", gaussians._xyz.shape[0])

    if gopid == 0:
        stage = "refine"
    elif gopid > 0:
        stage = "following"
    else:
        raise ValueError

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # precompute keyframe
    pre_times = []
    for iteration in range(300):
        t0 = time()
        torch.cuda.synchronize()
        xyz_keyframes = []
        rot_keyframes = []
        sca_keyframes = []
        for idx in range(gaussians.keyframe_num):
            feat = gaussians._point_feats[:, idx, :]
            dy_xyz = gaussians.get_deform_mlp(feat[:, :12])
            xyz_keyframes.append(dy_xyz)
            dcov = gaussians.get_cov_mlp(feat[:, :12])
            dy_rot = gaussians.get_rotation_dynamic_ori + dcov[:, 3:]
            rot_keyframes.append(dy_rot)
            if gopid > 0:
                dy_sca = gaussians.get_scaling_dynamic_ori + dcov[:, :3]
                sca_keyframes.append(dy_sca)
        gaussians._xyz_keyframe = torch.stack(xyz_keyframes, dim=1)
        gaussians._rotation_keyframe = torch.stack(rot_keyframes, dim=1)
        if gopid > 0:
            gaussians._scaling_keyframe = torch.stack(sca_keyframes, dim=1).cuda()
            stage = "following"
        torch.cuda.synchronize()
        t1 = time()
        duration = t1 - t0
        if iteration > 100:  # warm up
            pre_times.append(duration)

    pre_delay = np.mean(np.array(pre_times)) / gop
    print("precompute keyframe latency : {:>12.7f}".format(pre_delay))

    test_cams = scene.getTestCameras()
    test_cams.dataset.read_clip_features(False)
    test_view_num = len(test_cams)
    print(f"loaded {test_view_num} images")

    viewpoint_stack_loader = MultiEpochsDataLoader(test_cams, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=list)

    # start timing
    times = []
    iteration = 0
    for _ in range(3):
        for idx, viewpoint_cams in enumerate(tqdm(viewpoint_stack_loader, desc="timing ")):
            viewpoint_cam = viewpoint_cams[0]
            viewpoint_cam.cuda()
            if not kwargs['include_feature']:
                images, duration = stream_render_eval(viewpoint_cam, gaussians, pipe, background, stage=stage)  # fps_render
            else:
                images, duration = stream_lang_render_eval(viewpoint_cam, gaussians, pipe, background, stage=stage)
            if iteration > 30:  # warm up
                times.append(duration)
            iteration += 1
    delay = np.mean(np.array(times))
    print("render_latency : {:>12.7f}".format(delay))
    return delay + pre_delay


def render_video(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene: Scene, save_path, gopid):
    image_output_path = os.path.join(save_path, 'general_views')
    os.makedirs(image_output_path, exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    stage = "refine"

    # precompute keyframe
    xyz_keyframes = []
    rot_keyframes = []
    sca_keyframes = []
    for idx in range(gaussians.keyframe_num):
        feat = gaussians._point_feats[:, idx, :]
        dy_xyz = gaussians.get_deform_mlp(feat[:, :12])
        xyz_keyframes.append(dy_xyz)
        dcov = gaussians.get_cov_mlp(feat[:, :12])
        dy_rot = gaussians.get_rotation_dynamic_ori + dcov[:, 3:]
        rot_keyframes.append(dy_rot)
        if gopid > 0:
            dy_sca = gaussians.get_scaling_dynamic_ori + dcov[:, :3]
            sca_keyframes.append(dy_sca)
    gaussians._xyz_keyframe = torch.stack(xyz_keyframes, dim=1)
    gaussians._rotation_keyframe = torch.stack(rot_keyframes, dim=1)
    if gopid > 0:
        gaussians._scaling_keyframe = torch.stack(sca_keyframes, dim=1)
        stage = "following"

    test_cams = scene.getVideoCameras()
    test_view_num = len(test_cams)
    print(f"loaded {test_view_num} images")
    viewpoint_stack_loader = MultiEpochsDataLoader(test_cams, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=list)

    for idx, viewpoint_cams in enumerate(tqdm(viewpoint_stack_loader, desc="render video")):
        viewpoint_cam = viewpoint_cams[0]
        outputs = stream_render_eval(viewpoint_cam, gaussians, pipe, background, stage=stage, evaluation=True)
        image = torch.clamp(outputs[0], 0.0, 1.0)

        image = image.permute(1, 2, 0).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        timestamp = viewpoint_cam.time * args.gop
        img_idx = int(timestamp) + gopid * args.gop
        cv2.imwrite(os.path.join(image_output_path, '{0:05d}'.format(img_idx) + ".png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def test(dataset, hyper, opt, pipe, args, checkpoint_path, save_path, gopid, pre_gaussians, gop=60, all_temporal_features=None):
    gaussians = GaussianModel(dataset.sh_degree, True, eval=True, language=args.language)

    scene = Scene(dataset, gaussians, duration=[gopid*gop, (gopid+1)*gop], timedordered=False, skip_init=True)
    time_line = scene.maxtime
    gaussians.time_line = time_line
    interval = 10
    if 'n20' in args.postfix:
        interval = 20
    elif 'n5' in args.postfix:
        interval = 5
    elif 'n2' in args.postfix:
        interval = 2
    gaussians.keyframe_num = time_line // interval + 1
    gaussians.time_embedding_num = gaussians.keyframe_num
    gaussians.interval = interval
    print(f"[INFO] gopid={gopid}, gop length={gop}, loaded time length={gaussians.time_line}, interval={gaussians.interval}, feature nums={gaussians.time_embedding_num}")

    # load compressed features
    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"
    compr_exp_config = OmegaConf.load(compression_config)
    experiment = compr_exp_config['experiments'][0]
    experiment_name = experiment['name']
    assert experiment_name == 'jxl_quant', f'config file error!'

    gaussians.active_sh_degree = dataset.sh_degree
    if args.postfix == "mu_lerp":
        gaussians.setup_interpolators('lerp', 'slerp')
    elif args.postfix == "rot_lerp":
        gaussians.setup_interpolators('chip', 'lerp')

    if not os.path.exists(os.path.join(checkpoint_path, "compression_info.csv")):
        print(f"checkpoints not found in {checkpoint_path}!")
    compr_info = pd.read_csv(os.path.join(checkpoint_path, "compression_info.csv"), index_col=0)

    if all_temporal_features is None:
        compressed_file = os.path.join(checkpoint_path, f'_point_feats.mp4' if args.qp == 6 else f'_point_feats_{args.qp}.mp4')
        if not os.path.exists(compressed_file):
            os.system(f"ffmpeg -y -framerate 30 -i {checkpoint_path}/feat_images/_point_feats_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {checkpoint_path}/_point_feats_{args.qp}.mp4")
        assert os.path.exists(compressed_file)
        temporal_features = measure_decode_latency(compressed_file)
    else:
        temporal_features = all_temporal_features[gopid * 7: gopid*7 + 7]  # testing after merge gop videos into one | only used for gop length is 60 and keyframe interval is 10

    if gopid == 0:
        for attribute in experiment['attributes']:
            attr_name = attribute["name"]
            if not args.language and 'language' in attr_name:
                continue
            compressed_file = os.path.join(checkpoint_path, compr_info.loc[attr_name, "file"])
            decompress_attr_3(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"],
                              compr_info.loc[attr_name, "max"], None, temporal_features)
    else:
        if args.pt:
            pre_gaussians.offset_mode = 0
        gaussians._features_dc_dynamic = pre_gaussians.get_feature_dc_dynamic_ori.detach()
        gaussians._features_rest_dynamic = pre_gaussians.get_feature_rest_dynamic_ori.detach()
        gaussians._opacity_dynamic = pre_gaussians.get_opacity_dynamic_ori.detach()
        gaussians._scaling_dynamic = pre_gaussians.get_scaling_dynamic_ori.detach()
        gaussians._rotation_dynamic = pre_gaussians.get_rotation_dynamic_ori.detach()

        gaussians._features_dc = pre_gaussians.get_feature_dc_ori.detach()
        gaussians._features_rest = pre_gaussians.get_feature_rest_ori.detach()
        gaussians._opacity = pre_gaussians.get_opacity_ori.detach()
        gaussians._scaling = pre_gaussians.get_scaling_ori.detach()
        gaussians._rotation = pre_gaussians.get_rotation_ori.detach()

        if args.language:
            gaussians._language_feature_dynamic = pre_gaussians.get_language_feature_dynamic.detach()
            gaussians._language_feature = pre_gaussians.get_language_feature.detach()

        # load temporal features
        for attribute in experiment['attributes']:
                attr_name = attribute["name"]
                if 'point_feat' in attr_name:
                    compressed_file = os.path.join(checkpoint_path, compr_info.loc[attr_name, "file"])
                    decompress_attr_3(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"],
                                    compr_info.loc[attr_name, "max"], None, temporal_features)

    # checkpoint = torch.load(os.path.join(checkpoint_path, "model.pth"), map_location='cuda')
    # skip = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity",
    #         "_xyz_dynamic", "_features_dc_dynamic", "_features_rest_dynamic", "_scaling_dynamic", "_rotation_dynamic", "_opacity_dynamic"]
    # for attribute in experiment['attributes']:
    #     attr_name = attribute["name"]
    #     if attr_name not in skip or gopid < 1:
    #         compressed_file = os.path.join(checkpoint_path, compr_info.loc[attr_name, "file"])
    #         decompress_attr_3(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"],
    #                           compr_info.loc[attr_name, "max"], None, temporal_features)
    #     else:
    #         setattr(gaussians, f"{attr_name}", checkpoint[f"{attr_name}"].float().cuda())

    if gopid > 0:
        gaussians.offset_mode = 22
        if args.postfix == "wo_st_refine":
            gaussians.offset_mode = 2
        gaussians.load_compressed_offset(os.path.join(checkpoint_path, "offset_compressed.pkl"))

        compress_offset = []
        xyz_dynamic_offset = experiment["dy_offset"][0]
        compress_offset.append(xyz_dynamic_offset)
        if gaussians.offset_mode == 22:
            xyz_offset = experiment["st_offset"][0]
            compress_offset.append(xyz_offset)
        for attribute in compress_offset:
            attr_name = attribute["name"]
            compressed_file = os.path.join(checkpoint_path, compr_info.loc[attr_name, "file"])
            decompress_attr(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"], None)

    gaussians.mlp_deform.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_deform.pth"), map_location="cuda"))
    gaussians.mlp_cov.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_cov.pth"), map_location="cuda"))
    gaussians.mlp_opacity.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_opacity.pth"), map_location="cuda"))
    gaussians.mlp_color.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_color.pth"), map_location="cuda"))
    if args.language:
        gaussians.mlp_lang.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_lang.pth"), map_location="cuda"))

    torch.cuda.empty_cache()

    with torch.no_grad():
        if args.fps_test:
            delay = render_nogt(dataset, opt, hyper, pipe, args, gaussians, scene, gop, gopid, include_feature=args.language)
            return delay, gaussians
        elif args.video_render:
            render_video(dataset, opt, hyper, pipe, args, gaussians, scene, save_path, gopid)
            return None, gaussians
        elif args.opacity_vis:
            dynamic_opacity_list = opacity_vis(dataset, opt, hyper, pipe, args, gaussians, scene, save_path, gopid, include_feature=args.language)
            return dynamic_opacity_list, gaussians
        else:
            # if gopid != 0:
            metrics = reconstruction_testing(dataset, opt, hyper, pipe, args, gaussians, scene, save_path, gopid, include_feature=args.language)
            return metrics, gaussians
            # return [0,0,0,[0]], gaussians


def test_all(dataset, hyper, opt, pipe, args, checkpoint_path, gop=60, gopids=None):
    if gopids is None:
        gop_nums = 300 // gop
        gop_list = range(gop_nums)
    else:
        gop_list = gopids
    print(f"Processing gop", gop_list)

    # check checkpoints
    not_exists = []
    for gopid in gop_list:
        if gopid == 0:
            checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.postfix}', 'compression', 'best', 'jxl_quant')
            if not os.path.exists(checkpoint_dir):
                checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.gop0_postfix}', 'compression', 'best', 'jxl_quant')
        else:
            checkpoint_dir = os.path.join(checkpoint_path, f'gop{gopid}' if args.postfix is None else f'gop{gopid}_{args.postfix}', 'compression', 'best', 'jxl_quant')
        if not os.path.exists(checkpoint_dir):
            not_exists.append(checkpoint_dir)
    if len(not_exists) > 0:
        print(f"checkpoint not in ", not_exists)
        sys.exit(-1)

    save_path = os.path.join(checkpoint_path, "experiments", f'results' if args.postfix is None else f'results_{args.postfix}', f'metrics_qp{args.qp}')
    os.makedirs(save_path, exist_ok=True)

    # merge all temporal feature images into one video
    temporal_features = None
    if args.all_feats:
        feat_image_list = []
        for gopid in gop_list:
            if gopid == 0:
                # if args.postfix in ["mu_lerp", "rot_lerp"]:
                #     checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.postfix}', 'compression', 'best', 'jxl_quant')
                # else:
                #     checkpoint_dir = os.path.join(checkpoint_path, 'gop0_1', 'compression', 'best', 'jxl_quant')
                checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.postfix}', 'compression', 'best', 'jxl_quant')
                if not os.path.exists(checkpoint_dir):
                    checkpoint_dir = os.path.join(checkpoint_path, 'gop0_1', 'compression', 'best', 'jxl_quant')
            else:
                checkpoint_dir = os.path.join(checkpoint_path, f'gop{gopid}' if args.postfix is None else f'gop{gopid}_{args.postfix}', 'compression', 'best', 'jxl_quant')
            feat_images_dir = os.path.join(checkpoint_dir, "feat_images")
            feat_images = sorted(os.listdir(feat_images_dir))
            for path in feat_images:
                feat_image_list.append(os.path.join(feat_images_dir, path))

        feat_images_dir = os.path.join(checkpoint_path, "experiments", f'results' if args.postfix is None else f'results_{args.postfix}', "all_feat_images")
        os.makedirs(feat_images_dir, exist_ok=True)
        for idx, path in enumerate(feat_image_list):
            image_name = f"_point_feats_{idx}.png"
            target_path = os.path.join(feat_images_dir, image_name)
            os.symlink(path, target_path)

        compressed_file = os.path.join(feat_images_dir, f'_point_feats_{args.qp}.mp4')
        if not os.path.exists(compressed_file):
            os.system(f"ffmpeg -y -framerate 30 -i {feat_images_dir}/_point_feats_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {feat_images_dir}/_point_feats_{args.qp}.mp4")
        temporal_features = measure_decode_latency(compressed_file)

    # evaluate
    psnr_ = []
    ssim_ = []
    lpips_ = []
    size_ = []
    psnr_list = []
    render_latency = []
    dy_opacity_diff = []
    pre_gaussians = None
    for gopid in gop_list:
        if gopid == 0:
            checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.postfix}', 'compression', 'best', 'jxl_quant')
            if not os.path.exists(checkpoint_dir):
                checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.gop0_postfix}', 'compression', 'best', 'jxl_quant')
        else:
            checkpoint_dir = os.path.join(checkpoint_path, f'gop{gopid}' if args.postfix is None else f'gop{gopid}_{args.postfix}', 'compression', 'best','jxl_quant')
        outputs, pre_gaussians = test(dataset, hyper, opt, pipe, args, checkpoint_dir, save_path, gopid, pre_gaussians, gop=gop, all_temporal_features=temporal_features)
        args.model_path = None
        if args.fps_test:
            render_latency.append(outputs)
        elif args.video_render:
            pass
        elif args.opacity_vis:
            dy_opacity_diff.extend(outputs)
        else:
            psnr_.append(outputs[0])
            ssim_.append(outputs[1])
            lpips_.append(outputs[2])
            psnr_list.extend(outputs[3])

            # size
            gaussian_attrs = ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest',
                              '_xyz_dynamic', '_scaling_dynamic', '_rotation_dynamic', '_opacity_dynamic', '_features_dc_dynamic', '_features_rest_dynamic']
            if args.language:
                gaussian_attrs.extend(['_language_feature', '_language_feature_dynamic'])
            gaussian_offset_attrs = ['_xyz_offset', '_xyz_dynamic_offset']
            mlps = ['mlp_deform', 'mlp_cov', 'mlp_opacity', 'mlp_color']
            if gopid == 0:
                base_dir = os.path.join(checkpoint_path, f'gop0_{args.postfix}', 'compression', 'best', 'jxl_quant')
                if not os.path.exists(base_dir):
                    base_dir = os.path.join(checkpoint_path, f'gop0_{args.gop0_postfix}', 'compression', 'best', 'jxl_quant')
            else:
                base_dir = os.path.join(checkpoint_path, f'gop{gopid}' if args.postfix is None else f'gop{gopid}_{args.postfix}', 'compression', 'best', 'jxl_quant')
            gop_size = 0

            temporal_feats_path = os.path.join(base_dir, f'_point_feats.mp4' if args.qp == 6 else f'_point_feats_{args.qp}.mp4')
            file_size = get_file_size_in_kB(temporal_feats_path) / args.gop
            gop_size += file_size

            if gopid == 0:
                for file in gaussian_attrs:
                    file_path = os.path.join(base_dir, f'{file}.jxl')
                    attr_file_size = get_file_size_in_kB(file_path)
                    file_size = attr_file_size / args.gop
                    gop_size += file_size
            else:
                for file in gaussian_offset_attrs:
                    if args.postfix == "wo_st_refine" and file == '_xyz_offset':
                        continue

                    file_path = os.path.join(base_dir, f'{file}.jxl')
                    attr_file_size = get_file_size_in_kB(file_path)
                    gop_size += attr_file_size / args.gop
                file_path = os.path.join(base_dir, 'offset_compressed.pkl')
                attr_file_size = get_file_size_in_kB(file_path)
                gop_size += attr_file_size / args.gop
            for file in mlps:
                file_path = os.path.join(base_dir, f'{file}.pth')
                file_size = get_file_size_in_kB(file_path) / args.gop
                gop_size += file_size
            compress_file_path = os.path.join(base_dir, f'compression_info.csv')
            file_size = get_file_size_in_kB(compress_file_path) / args.gop
            gop_size += file_size
            size_.append(gop_size)

    if args.fps_test:
        render_latency = np.array(render_latency)
        print(f"Average render_latency={np.mean(render_latency):.5f}, FPS={1/np.mean(render_latency)}")
        np.save(os.path.join(save_path, 'render_delay.npy'), render_latency)
    elif args.video_render:
        pass
    elif args.opacity_vis:
        frame_ids = range(len(dy_opacity_diff))
        plt.figure(figsize=(10, 4))
        plt.plot(frame_ids, dy_opacity_diff, color='black', linewidth=1.5)
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')
        plt.xlabel('Frame ID', fontsize=16, fontweight='bold')
        plt.ylabel('Dual-Opacity Difference', fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "opacity_vis", "dynamic_diff.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        psnr_ = np.array(psnr_)
        ssim_ = np.array(ssim_)
        lpips_ = np.array(lpips_)
        size_ = np.mean(np.array(size_))
        psnr_list = np.array(psnr_list)
        print(f"Average PSNR={np.mean(psnr_):.5f}, Average SSIM={np.mean(ssim_):.5f}, Average LPIPS={np.mean(lpips_):.5f}, Average frame size: {size_}kB")
        np.save(os.path.join(save_path, 'psnr.npy'), psnr_)
        if not args.disable_ssim:
            np.save(os.path.join(save_path, 'ssim.npy'), ssim_)
        if not args.disable_lpips:
            np.save(os.path.join(save_path, 'lpips.npy'), lpips_)
        np.save(os.path.join(save_path, 'psnr_per_frame.npy'), psnr_list)
        np.save(os.path.join(save_path, 'size.npy'), size_)


def test_attributes(dataset, hyper, opt, pipe, args, checkpoint_path, gopid, gop=60):
    args.model_path = os.path.join(checkpoint_path, 'test')
    os.makedirs(args.model_path, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, True)
    dataset.model_path = args.model_path

    scene = Scene(dataset, gaussians, duration=[gopid*gop, (gopid+1)*gop], timedordered=False, skip_init=True)
    time_line = scene.maxtime
    gaussians.time_line = time_line
    gaussians.time_embedding_num = 6 + 1

    # load compressed features
    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"
    compr_exp_config = OmegaConf.load(compression_config)
    experiment = compr_exp_config['experiments'][0]
    experiment_name = experiment['name']
    assert experiment_name == 'jxl_quant', f'config file error!'

    assert os.path.exists(os.path.join(checkpoint_path, "model.pth")), f"checkpoint not found in {checkpoint_path}!"
    checkpoint = torch.load(os.path.join(checkpoint_path, "model.pth"), map_location='cuda')
    include_feature = True
    # previous_model_pth = "/home/kzh/mountpoint/runtime/Work4/Ours/coffee_martini/gop0_lang/compression/best/jxl_quant"
    # previous_checkpoint = torch.load(os.path.join(previous_model_pth, "model.pth"), map_location='cuda')

    # scale_bound_min = (torch.min(checkpoint["_scaling"], dim=0, keepdim=True)[0]).detach()
    # scale_bound_max = (torch.max(checkpoint["_scaling"], dim=0, keepdim=True)[0]).detach()

    # scale_bound_min_ = (torch.min(previous_checkpoint["_scaling"], dim=0, keepdim=True)[0]).detach()
    # scale_bound_max_ = (torch.max(previous_checkpoint["_scaling"], dim=0, keepdim=True)[0]).detach()


    # canonical gaussian offset
    # xyz_dynamic_offset = checkpoint["_xyz_dynamic"] - previous_checkpoint["_xyz_dynamic"]
    # local_xyz_dynamic_offset = checkpoint["_xyz_dynamic"] - checkpoint["_xyz_dynamic_offset"]
    # features_dc_dynamic_offset = checkpoint["_features_dc_dynamic"] - previous_checkpoint["_features_dc_dynamic"]
    # features_rest_dynamic_offset = checkpoint["_features_rest_dynamic"] - previous_checkpoint["_features_rest_dynamic"]
    # scaling_dynamic_offset = checkpoint["_scaling_dynamic"] - previous_checkpoint["_scaling_dynamic"]
    # rotation_dynamic_offset = checkpoint["_rotation_dynamic"] - previous_checkpoint["_rotation_dynamic"]
    # opacity_dynamic_offset = checkpoint["_opacity_dynamic"] - previous_checkpoint["_opacity_dynamic"]

    # xyz_offset = checkpoint["_xyz"] - previous_checkpoint["_xyz"]
    # f_dc_offset = checkpoint["_features_dc"] - previous_checkpoint["_features_dc"]
    # f_rest_offset = checkpoint["_features_rest"] - previous_checkpoint["_features_rest"]
    # scaling_offset = checkpoint["_scaling"] - previous_checkpoint["_scaling"]
    # rotation_offset = checkpoint["_rotation"] - previous_checkpoint["_rotation"]
    # opacity_offset = checkpoint["_opacity"] - previous_checkpoint["_opacity"]

    # with open(os.path.join(checkpoint_path, "offset_compressed.pkl"), 'rb') as f:
    #     data = pickle.load(f)
    #
    # del data["latents"]["_xyz_offset"]
    # del data["latents"]["_xyz_dynamic_offset"]
    #
    # with open(os.path.join("output", "compressed.pkl"), 'wb') as f:
    #     pickle.dump(data, f)

    # img = gaussians.reshape_as_grid_img(xyz_offset, grid_sidelen=377)
    # attr_np = img.cpu().numpy()
    # out_file = os.path.join("output", f"xyz_offset.jxl")
    # attr_params = {"level": 101}
    # codec = JpegXlCodec()
    # codec.encode_with_normalization(attr_np, "_xyz", out_file, attr_min=-4, attr_max=4, unuse_threshold=True, **attr_params)
    #
    # img = xyz_dynamic_offset.reshape((196, 196, -1))
    # attr_np = img.cpu().numpy()
    # out_file = os.path.join("output", f"xyz_dynamic_offset.jxl")
    # attr_params = {"level": 101}
    # codec = JpegXlCodec()
    # codec.encode_with_normalization(attr_np, "_xyz_dynamic", out_file, attr_min=-4, attr_max=4, unuse_threshold=True, **attr_params)

    # gaussian attribution compression testing
    # attr_tensor = checkpoint[f'_xyz'].float().cuda()
    # img = gaussians.reshape_as_grid_img(attr_tensor, grid_sidelen=358)
    # attr_np = img.cpu().numpy()
    # attr_name = "_xyz"
    # out_file = os.path.join(checkpoint_path, f"{attr_name}.jxl")
    # attr_params = {"level": 101}

    # attr_name = "_rotation"
    # attr_tensor = checkpoint[attr_name].float().cuda()
    # img = gaussians.reshape_as_grid_img(attr_tensor, grid_sidelen=393)
    # attr_np = img.cpu().numpy()
    # out_file = os.path.join(checkpoint_path, f"{attr_name}.jxl")
    # attr_params = {"level": 101}

    # quantization = 8
    # min_val = attr_np.min()
    # max_val = attr_np.max()
    # val_range = max_val - min_val
    # # no division by zero
    # if val_range == 0:
    #     val_range = 1
    # attr_np_norm = (attr_np - min_val) / val_range
    # qpow = 2 ** quantization
    # attr_np_quantized = np.round(attr_np_norm * qpow) / qpow
    # attr_np = attr_np_quantized * val_range + min_val
    # attr_np = attr_np.astype(np.float32)
    # codec = JpegXlCodec()
    # codec.encode_with_normalization(attr_np, attr_name, out_file, attr_min=-4, attr_max=4, unuse_threshold=True, **attr_params)
    # codec.encode(attr_np, out_file, **attr_params)

    # codec = PNGCodec()
    # temporal_feats = checkpoint[f'_point_feats'].float().cuda()
    # for idx in range(gaussians.time_embedding_num):
    #     attr_tensor = temporal_feats[:, idx, :]
    #     img = gaussians.reshape_as_grid_img(attr_tensor, grid_sidelen=203)
    #     img = gaussians.frame_maker(img, 203, 203)
    #     attr_np = img.cpu().numpy()
    #     attr_name = f'_point_feat_{idx}'
    #     out_file = os.path.join(checkpoint_path, "feat_images_quat", f"{attr_name}.png")
    #     attr_params = {"dtype": "uint16"}
    #
    #     codec.encode_with_normalization(attr_np, attr_name, out_file, attr_min=-1, attr_max=1, unuse_threshold=False, **attr_params)

    # sys.exit(0)
    torch.cuda.empty_cache()

    gaussians.active_sh_degree = dataset.sh_degree
    if not os.path.exists(os.path.join(checkpoint_path, "compression_info.csv")):
        print(f"checkpoints not found in {checkpoint_path}!")
    compr_info = pd.read_csv(os.path.join(checkpoint_path, "compression_info.csv"), index_col=0)
    # compr_info.index = compr_info.index.str.replace('_point_feat', '_point_feats')

    # skip = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity",]
    # skip = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity",
    #         "_xyz_dynamic", "_features_dc_dynamic", "_features_rest_dynamic", "_scaling_dynamic", "_rotation_dynamic", "_opacity_dynamic", "_point_feats"]
    skip = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity",
            "_xyz_dynamic", "_features_dc_dynamic", "_features_rest_dynamic", "_scaling_dynamic", "_rotation_dynamic", "_opacity_dynamic", "_point_feats",
            "_language_feature", "_language_feature_dynamic"]
    for attribute in experiment['attributes']:
        attr_name = attribute["name"]
        if attr_name not in skip:
            compressed_file = os.path.join(checkpoint_path, compr_info.loc[attr_name, "file"])
            decompress_attr_2(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"],
                              compr_info.loc[attr_name, "max"], None, f'feat_images' if args.qp == 12 else f'feat_images_{args.qp}')
        else:
            setattr(gaussians, f"{attr_name}", checkpoint[attr_name].float().cuda())

    if os.path.exists(os.path.join(checkpoint_path, "offset_compressed.pkl")):
        gaussians.offset_mode = 22
        gaussians.load_compressed_offset(os.path.join(checkpoint_path, "offset_compressed.pkl"))

        compress_offset = []
        xyz_dynamic_offset = experiment["dy_offset"][0]
        compress_offset.append(xyz_dynamic_offset)
        xyz_offset = experiment["st_offset"][0]
        compress_offset.append(xyz_offset)

        for attribute in compress_offset:
            attr_name = attribute["name"]
            compressed_file = os.path.join(checkpoint_path, compr_info.loc[attr_name, "file"])
            decompress_attr(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"], None)

    gaussians.mlp_lang.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_lang.pth"), map_location="cuda"))
    gaussians.mlp_deform.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_deform.pth"), map_location="cuda"))
    gaussians.mlp_cov.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_cov.pth"), map_location="cuda"))
    gaussians.mlp_opacity.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_opacity.pth"), map_location="cuda"))
    gaussians.mlp_color.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_color.pth"), map_location="cuda"))

    torch.cuda.empty_cache()

    with torch.no_grad():
        reconstruction_testing(dataset, opt, hyper, pipe, args, gaussians, scene, args.model_path, gopid, include_feature=include_feature)


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()

    # setup_seed
    seed = 6666
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--gop0_postfix", type=str, default=None)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--compre_config", type=str, default="")
    parser.add_argument("--qp", type=int, default=6)
    parser.add_argument("--all_feats", action="store_true")
    parser.add_argument("--gopids", nargs="+", type=int, default=[])
    parser.add_argument("--gopid", type=int, default=0)
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--disable_ssim", action="store_true")
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--write_images", action="store_true")
    parser.add_argument("--write_lang", action="store_true")
    parser.add_argument("--language", action="store_true")
    parser.add_argument("--pt", action="store_true")
    parser.add_argument("--fps_test", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--video_render", action="store_true")
    parser.add_argument("--opacity_vis", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if os.path.exists(args.checkpoint_path):
        if len(args.gopids) == 0:
            args.gopids = None
        test_all(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args, args.checkpoint_path, args.gop, args.gopids)

        # if args.all:
        #     if len(args.gopids) == 0:
        #         args.gopids = None
        #     test_all(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args, args.checkpoint_path, args.gop, args.gopids)
        # else:
        #     test_attributes(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args, args.checkpoint_path, args.gopid, args.gop)
    else:
        raise NotADirectoryError
