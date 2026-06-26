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
import numpy as np
import random
import os
import sys
import copy
from time import time
from random import randint
import pickle

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

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from scene import Scene
from scene.scaffold_gaussian_model import GaussianModel
from gaussian_renderer import hac_render, prefilter_voxel
from scene.dataset import MultiEpochsDataLoader
from gaussian_renderer import fps_render, stream_render_eval

from utils.loader_utils import FineSampler, get_stamp_list
from utils.general_utils import safe_state, inverse_sigmoid
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

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
        print('\033[32m' + f"Decoding latency: {(decode_end_time - decode_start_time) / 60:.6f}s" + '\033[0m')
        return frames

    except Exception as e:
        print(f"Error: {e}")


def get_file_size_in_kB(file_path):
    """Return the file size in kilobytes (kB)."""
    size_in_bytes = os.path.getsize(file_path)
    # Divide by 1024 to convert from bytes to kilobytes
    size_in_kB = size_in_bytes / 1024
    return round(size_in_kB, 5)

def get_directory_size_in_kB(dir_path):
    file_sizes = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            file_sizes += size
    file_sizes = file_sizes / 1024
    return round(file_sizes, 5)


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


def reconstruction_testing(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene: Scene, save_path, gopid, stage, **kwargs):
    image_output_path = os.path.join(save_path, 'images')
    os.makedirs(image_output_path, exist_ok=True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        outputs = hac_render(viewpoint_cam, gaussians, pipe, background, stage, visible_mask=voxel_visible_mask)
        image = torch.clamp(outputs["render"], 0.0, 1.0)

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

            if args.language and args.write_lang:
                lang = outputs["language_feature"].permute(1, 2, 0).detach().cpu().numpy()
                for level in range(0, lang.shape[2], 3):
                    _lang = (lang[:, :, level: level+3] * 65535).astype(np.uint16)
                    save_images.append(_lang)

            for idx, image in enumerate(save_images):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(image_output_path, '{0:05d}'.format(img_idx) + f"_{idx}" + ".png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        if args.language and img_idx in saved_idx:
            lang = outputs["language_feature"].permute(1, 2, 0).detach().cpu().numpy()
            np.save(os.path.join(image_output_path, '{0:05d}'.format(img_idx) + ".npy"), lang)

        torch.cuda.empty_cache()

    progress_bar.close()
    psnr_test /= test_view_num
    ssim_test /= test_view_num
    lpips_test /= test_view_num
    print('\033[32m' + f"PSNR={psnr_test:.5f}, SSIM={ssim_test:.5f}, LPIPS={lpips_test:.5f}" + '\033[0m')
    return psnr_test, ssim_test, lpips_test, psnr_list


def render_nogt(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene: Scene, gop, gopid):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if gopid == 0:
        stage = "refine"
    elif gopid > 0:
        d_feat, d_offsets, _ = gaussians.get_ntc(gaussians.get_anchor)
        gaussians._anchor_feat += d_feat[:gaussians.get_static_anchor_num]
        gaussians._offset += d_offsets[:gaussians.get_static_anchor_num]
        gaussians._anchor_feat_dynamic += d_feat[gaussians.get_static_anchor_num:]
        gaussians._offset_dynamic += d_offsets[gaussians.get_static_anchor_num:]
        stage = "eval"
    else:
        raise ValueError

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

            torch.cuda.synchronize()
            t0 = time()
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            duration = time() - t0
            # duration = 0
            outputs = hac_render(viewpoint_cam, gaussians, pipe, background, stage, visible_mask=voxel_visible_mask)
            duration += outputs['time_sub']

            if iteration > 30:  # warm up
                times.append(duration)
            iteration += 1
    delay = np.mean(np.array(times))
    print("render_latency : {:>12.7f}".format(delay))
    return delay


def render_video(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene: Scene, save_path, gopid):
    image_output_path = os.path.join(save_path, 'general_views')
    os.makedirs(image_output_path, exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if gopid == 0:
        stage = "refine"
    elif gopid > 0:
        stage = "following"
    else:
        raise ValueError

    test_cams = scene.getVideoCameras()
    test_view_num = len(test_cams)
    print(f"loaded {test_view_num} images")
    viewpoint_stack_loader = MultiEpochsDataLoader(test_cams, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=list)

    for idx, viewpoint_cams in enumerate(tqdm(viewpoint_stack_loader, desc="render video")):
        viewpoint_cam = viewpoint_cams[0]
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        outputs = hac_render(viewpoint_cam, gaussians, pipe, background, stage, visible_mask=voxel_visible_mask)
        image = torch.clamp(outputs['render'], 0.0, 1.0)

        image = image.permute(1, 2, 0).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        timestamp = viewpoint_cam.time * args.gop
        img_idx = int(timestamp) + gopid * args.gop
        cv2.imwrite(os.path.join(image_output_path, '{0:05d}'.format(img_idx) + ".png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def test(dataset, hyper, opt, pipe, args, checkpoint_path, save_path, gopid, pre_gaussians, gop=60, all_temporal_features=None):
    if gopid == 0:
        stage = "eval"
    elif gopid > 0:
        stage = "following"
    else:
        raise ValueError
    gaussians = GaussianModel(dataset.feat_dim,
                              dataset.n_offsets,
                              dataset.voxel_size,
                              dataset.update_depth,
                              dataset.update_init_factor,
                              dataset.update_hierachy_factor,
                              n_features_per_level=args.n_features,
                              log2_hashmap_size=args.log2,
                              log2_hashmap_size_2D=args.log2_2D,
                              mode='hybrid',
                              enable_filter=dataset.enable_filter,
                              stage=stage,
                              decoded_version=True,
                              language=args.language,
                              print_log=False)
    gaussians.eval()

    scene = Scene(dataset, gaussians, duration=[gopid*gop, (gopid+1)*gop], load_memory=False, timedordered=False, skip_init=True)
    time_line = scene.maxtime
    gaussians.time_line = time_line
    interval = 10
    if 'i20' in args.postfix:
        interval = 20
    elif 'i5' in args.postfix:
        interval = 5
    elif 'i2' in args.postfix:
        interval = 2
    keyframe_num = time_line // interval + gaussians.expand_time
    gaussians.keyframe_num = keyframe_num
    gaussians.interval = interval
    print(f"[INFO] gopid={gopid}, gop length={gop}, loaded time length={gaussians.time_line}, interval={gaussians.interval}, feature nums={gaussians.keyframe_num}")

    # load compressed features
    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"
    compr_exp_config = OmegaConf.load(compression_config)
    experiment = compr_exp_config['experiments'][0]
    experiment_name = experiment['name']
    assert experiment_name == 'png_quant', f'config file error!'

    if not os.path.exists(os.path.join(checkpoint_path, "compression_info.csv")):
        print(f"checkpoints not found in {checkpoint_path}!")
    compr_info = pd.read_csv(os.path.join(checkpoint_path, "compression_info.csv"), index_col=0)

    if all_temporal_features is None:
        compressed_file = os.path.join(checkpoint_path, f'_temporal_feat.mp4' if args.qp == 6 else f'_temporal_feat_{args.qp}.mp4')
        if not os.path.exists(compressed_file):
            os.system(f"ffmpeg -y -framerate 30 -i {checkpoint_path}/feat_images/_temporal_feat_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {checkpoint_path}/_temporal_feat_{args.qp}.mp4")
        assert os.path.exists(compressed_file)
        temporal_features = measure_decode_latency(compressed_file)
    else:
        temporal_features = all_temporal_features[gopid * keyframe_num: gopid * keyframe_num + keyframe_num]  # testing after merge gop videos into one | only used for gop length is 60 and keyframe interval is 10

    bit_stream_path = os.path.join(os.path.dirname(checkpoint_path), 'bitstreams')

    # load mlps checkpoint
    mlps = torch.load(os.path.join(checkpoint_path, "mlps.pth"), map_location="cuda")
    if gopid == 0:
        print("[INFO] load attributes")
        gaussians.mlp_grid.load_state_dict(mlps['mlp_grid'])
        gaussians.conduct_decoding_from_files(pre_path_name=bit_stream_path)

        rots = torch.zeros((gaussians._anchor.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        gaussians._rotation = rots
        rots_dynamic = torch.zeros((gaussians._anchor_dynamic.shape[0], 4), device="cuda")
        rots_dynamic[:, 0] = 1
        gaussians._rotation_dynamic = rots_dynamic
        opacities = inverse_sigmoid(0.1 * torch.ones((gaussians._anchor.shape[0], 1), dtype=torch.float, device="cuda"))
        gaussians._opacity = opacities
        opacities_dynamic = inverse_sigmoid(0.1 * torch.ones((gaussians._anchor_dynamic.shape[0], 1), dtype=torch.float, device="cuda"))
        gaussians._opacity_dynamic = opacities_dynamic

        print("[INFO] load temporal features")
        # assert len(temporal_features) == gaussians.keyframe_num, f"feature nums={len(temporal_features)}, while target feature nums={gaussians.keyframe_num}"
        try:
            min_val = compr_info.loc["_temporal_feat", "min"]
            max_val = compr_info.loc["_temporal_feat", "max"]
        except KeyError:
            min_val = compr_info.loc["_temporal_feats", "min"]
            max_val = compr_info.loc["_temporal_feats", "max"]
        attr_config = experiment['attributes'][0]
        assert attr_config['name'] == '_temporal_feat'
        N_dynamic = gaussians._anchor_dynamic.shape[0]
        feat_imgs = []
        for i in range(gaussians.keyframe_num):
            if attr_config.get('normalize', False):
                decompressed_attr = temporal_features[i] / 65535 * (max_val - min_val) + min_val
            else:
                decompressed_attr = temporal_features[i] / 65535
            feat_imgs.append(decompressed_attr)
        gaussians.set_point_feat_from_grid_img(feat_imgs, gaussians.keyframe_num, N_dynamic)

        print("[INFO] load attribute mlps")
        gaussians.mlp_cov.load_state_dict(mlps['mlp_cov'])
        gaussians.mlp_color.load_state_dict(mlps['mlp_color'])
        if args.language:
            assert gaussians.mlp_language is not None
            if os.path.exists(os.path.join(checkpoint_path, "mlp_language.pth")):
                mlp_language = torch.load(os.path.join(checkpoint_path, "mlp_language.pth"), map_location='cuda')
                gaussians.mlp_language.load_state_dict(mlp_language)
            else:
                assert 'mlp_language' in mlps.keys()
                gaussians.mlp_language.load_state_dict(mlps['mlp_language'])
        gaussians.mlp_opacity.load_state_dict(mlps['mlp_opacity'])
    else:
        gaussians._anchor = pre_gaussians._anchor.detach()
        gaussians._anchor_dynamic = pre_gaussians._anchor_dynamic.detach()
        gaussians._anchor_feat = pre_gaussians._anchor_feat.detach()
        gaussians._anchor_feat_dynamic = pre_gaussians._anchor_feat_dynamic.detach()
        gaussians._offset = pre_gaussians._offset.detach()
        gaussians._offset_dynamic = pre_gaussians._offset_dynamic.detach()
        gaussians._scaling = pre_gaussians._scaling.detach()
        gaussians._scaling_dynamic = pre_gaussians._scaling_dynamic.detach()

        gaussians._rotation = pre_gaussians._rotation
        gaussians._rotation_dynamic = pre_gaussians._rotation_dynamic
        gaussians._opacity = pre_gaussians._opacity
        gaussians._opacity_dynamic = pre_gaussians._opacity_dynamic

        gaussians.x_bound_min = pre_gaussians.x_bound_min
        gaussians.x_bound_max = pre_gaussians.x_bound_max

        print("[INFO] load temporal features")
        try:
            min_val = compr_info.loc["_temporal_feat", "min"]
            max_val = compr_info.loc["_temporal_feat", "max"]
        except KeyError:
            min_val = compr_info.loc["_temporal_feats", "min"]
            max_val = compr_info.loc["_temporal_feats", "max"]
        attr_config = experiment['attributes'][0]
        assert attr_config['name'] == '_temporal_feat'
        N_dynamic = gaussians._anchor_dynamic.shape[0]
        feat_imgs = []
        for i in range(gaussians.keyframe_num):
            if attr_config.get('normalize', False):
                decompressed_attr = temporal_features[i] / 65535 * (max_val - min_val) + min_val
            else:
                decompressed_attr = temporal_features[i] / 65535
            feat_imgs.append(decompressed_attr)
        gaussians.set_point_feat_from_grid_img(feat_imgs, gaussians.keyframe_num, N_dynamic)

        print("[INFO] load attribute mlps")
        gaussians.mlp_cov.load_state_dict(pre_gaussians.mlp_cov.state_dict())
        gaussians.mlp_color.load_state_dict(pre_gaussians.mlp_color.state_dict())
        if args.language:
            assert pre_gaussians.mlp_language is not None
            gaussians.mlp_language.load_state_dict(pre_gaussians.mlp_language.state_dict())
            # gaussians.mlp_language.load_state_dict(mlps['mlp_language'])
        gaussians.mlp_opacity.load_state_dict(pre_gaussians.mlp_opacity.state_dict())

    print("[INFO] deformation mlps")
    gaussians.mlp_deform_xyz.load_state_dict(mlps['mlp_deform_xyz'])
    gaussians.mlp_deform_cov.load_state_dict(mlps['mlp_deform_cov'])
    gaussians.mlp_deform_color.load_state_dict(mlps['mlp_deform_color'])
    gaussians.mlp_deform_opacity.load_state_dict(mlps['mlp_deform_opacity'])

    # load canonical deformation
    if gopid > 0:
        gaussians.conduct_decoding_for_ntc(pre_path_name=bit_stream_path)
        gaussians.ntc_mlp.load_state_dict(mlps['ntc_mlp'])

    torch.cuda.empty_cache()

    with torch.no_grad():
        if args.fps_test:
            delay = render_nogt(dataset, opt, hyper, pipe, args, gaussians, scene, gop, gopid)
            return delay, gaussians
        elif args.video_render:
            render_video(dataset, opt, hyper, pipe, args, gaussians, scene, save_path, gopid)
            return None, gaussians
        else:
            if gopid != 0:
                metrics = reconstruction_testing(dataset, opt, hyper, pipe, args, gaussians, scene, save_path, gopid, stage)
            else:
                metrics = [0,0,0,[0]]
            if gopid > 0 and not args.pt:
                d_feat, d_offsets, _ = gaussians.get_ntc(gaussians.get_anchor)
                gaussians._anchor_feat += d_feat[:gaussians.get_static_anchor_num]
                gaussians._offset += d_offsets[:gaussians.get_static_anchor_num]
                gaussians._anchor_feat_dynamic += d_feat[gaussians.get_static_anchor_num:]
                gaussians._offset_dynamic += d_offsets[gaussians.get_static_anchor_num:]
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
        checkpoint_dir = os.path.join(checkpoint_path, f'gop{gopid}' if args.postfix is None else f'gop{gopid}_{args.postfix}', 'compression', 'best', 'png_quant')
        if gopid == 0 and args.gop0_postfix is not None:
            checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.gop0_postfix}', 'compression', 'best', 'png_quant')
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
            checkpoint_dir = os.path.join(checkpoint_path, f'gop{gopid}' if args.postfix is None else f'gop{gopid}_{args.postfix}', 'compression', 'best', 'png_quant')
            feat_images_dir = os.path.join(checkpoint_dir, "feat_images")
            feat_images = sorted(os.listdir(feat_images_dir))
            for path in feat_images:
                feat_image_list.append(os.path.join(feat_images_dir, path))

        feat_images_dir = os.path.join(checkpoint_path, "experiments", f'results' if args.postfix is None else f'results_{args.postfix}', "all_feat_images")
        os.makedirs(feat_images_dir, exist_ok=True)
        for idx, path in enumerate(feat_image_list):
            image_name = f"_temporal_feat_{idx}.png"
            target_path = os.path.join(feat_images_dir, image_name)
            os.symlink(path, target_path)

        compressed_file = os.path.join(feat_images_dir, f'_temporal_feat_{args.qp}.mp4')
        if not os.path.exists(compressed_file):
            os.system(f"ffmpeg -y -framerate 30 -i {feat_images_dir}/_temporal_feat_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} {feat_images_dir}/_temporal_feat_{args.qp}.mp4")
        temporal_features = measure_decode_latency(compressed_file)

    # evaluate
    psnr_ = []
    ssim_ = []
    lpips_ = []
    psnr_list = []
    size_list = []
    render_latency = []
    pre_gaussians = None
    for gopid in gop_list:
        checkpoint_dir = os.path.join(checkpoint_path, f'gop{gopid}' if args.postfix is None else f'gop{gopid}_{args.postfix}', 'compression', 'best','png_quant')
        if gopid == 0 and args.gop0_postfix is not None:
            checkpoint_dir = os.path.join(checkpoint_path, f'gop0_{args.gop0_postfix}', 'compression', 'best', 'png_quant')
        outputs, pre_gaussians = test(dataset, hyper, opt, pipe, args, checkpoint_dir, save_path, gopid, pre_gaussians, gop=gop, all_temporal_features=temporal_features)
        args.model_path = None
        if args.fps_test:
            render_latency.append(outputs)
        elif args.video_render:
            pass
        else:
            psnr_.append(outputs[0])
            ssim_.append(outputs[1])
            lpips_.append(outputs[2])
            psnr_list.extend(outputs[3])

            # frame size calculation
            gop_size = 0

            local_dir = os.path.join(checkpoint_path, f'gop{gopid}_{args.postfix}', 'compression', 'best')
            if gopid == 0 and args.gop0_postfix is not None:
                local_dir = os.path.join(checkpoint_path, f'gop0_{args.gop0_postfix}', 'compression', 'best')
            attr_dir = os.path.join(local_dir, 'bitstreams')
            attr_size = get_directory_size_in_kB(attr_dir) / args.gop
            gop_size += attr_size

            base_dir = os.path.join(local_dir, 'png_quant')
            temporal_feats_path = os.path.join(base_dir, f'_temporal_feat.mp4' if args.qp == 6 else f'_temporal_feat_{args.qp}.mp4')
            file_size = get_file_size_in_kB(temporal_feats_path) / args.gop
            gop_size += file_size

            temporal_feats_path = os.path.join(base_dir, f'mlps.pth')
            file_size = get_file_size_in_kB(temporal_feats_path) / args.gop
            gop_size += file_size

            compress_file_path = os.path.join(base_dir, f'compression_info.csv')
            file_size = get_file_size_in_kB(compress_file_path) / args.gop
            gop_size += file_size

            size_list.append(gop_size)

    if args.fps_test:
        render_latency = np.array(render_latency)
        print(f"Average render_latency={np.mean(render_latency):.5f}, FPS={1/np.mean(render_latency)}")
        np.save(os.path.join(save_path, 'render_delay.npy'), render_latency)
    elif args.video_render:
        pass
    else:
        psnr_ = np.array(psnr_)
        ssim_ = np.array(ssim_)
        lpips_ = np.array(lpips_)
        psnr_list = np.array(psnr_list)
        size_ = np.mean(np.array(size_list))
        print('\033[33m' + f"Average PSNR={np.mean(psnr_):.5f}, Average SSIM={np.mean(ssim_):.5f}, Average LPIPS={np.mean(lpips_):.5f}, Average frame size: {size_}kB" + '\033[0m')
        np.save(os.path.join(save_path, 'psnr.npy'), psnr_)
        if not args.disable_ssim:
            np.save(os.path.join(save_path, 'ssim.npy'), ssim_)
        if not args.disable_lpips:
            np.save(os.path.join(save_path, 'lpips.npy'), lpips_)
        np.save(os.path.join(save_path, 'psnr_per_frame.npy'), psnr_list)
        np.save(os.path.join(save_path, 'size.npy'), size_)


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
    parser.add_argument("--scenes", nargs="+", type=str, default=None)
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
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.scenes is None:
        if os.path.exists(args.checkpoint_path):
            if len(args.gopids) == 0:
                args.gopids = None
            test_all(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args, args.checkpoint_path, args.gop, args.gopids)
        else:
            raise NotADirectoryError
    else:
        checkpoint_path = args.checkpoint_path
        source_path = args.source_path
        for scene in args.scenes:
            print("Evaluating " + scene)
            args.checkpoint_path = os.path.join(checkpoint_path, scene)
            args.source_path = os.path.join(source_path, scene)

            if os.path.exists(args.checkpoint_path):
                if len(args.gopids) == 0:
                    args.gopids = None
                test_all(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args, args.checkpoint_path, args.gop, args.gopids)
            else:
                print(f"Not find directory {args.checkpoint_path}")

