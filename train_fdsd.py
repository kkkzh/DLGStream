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
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import pandas as pd
from kornia.filters import sobel, spatial_gradient

from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig, OmegaConf

from fused_ssim import fused_ssim as ssim
# from utils.loss_utils import ssim

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from scene import Scene
from scene.fdsd_gaussian_model import GaussianModel
from gaussian_renderer import stream_render_fisd, dynamic_render, stream_render_lang

from utils.loader_utils import FineSampler, get_stamp_list
from utils.general_utils import safe_state, DecayScheduler
from utils.loss_utils import l1_loss, l1_loss_mask, lpips_loss
from utils.loss_utils import loss_depth_smoothness, patch_norm_mse_loss, patch_norm_mse_loss_global
from utils.image_utils import psnr
from utils.timer import Timer
from lpipsPyTorch import lpips

from compression.compression_exp import run_compressions, run_decompressions, decompress_geo_attr, run_single_decompression
from compression.decompress import decompress_all_to_ply
from compression.entropy_models import rdloss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_report(tb_writer, iteration, loss_dict, loss, elapsed, testing_iterations, scene: Scene, gaussian: GaussianModel,
                    renderFunc, renderArgs, stage, lr_dict=None, compress=False, decom_stage=False, **kwargs):
    decom = 'refined' if decom_stage else ''
    if tb_writer and iteration > 49:
        tb_writer.add_scalar(decom + f'1_train_loss_patches/l1_loss', loss_dict['L1'].item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/ssim_loss', loss_dict['Lssim'].item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/langl1_loss', loss_dict['Ltssim'].item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/total_points', gaussian.get_xyz_all.shape[0], iteration)
        # if lr_dict is not None and 'transformer' in lr_dict.keys():
        #     tb_writer.add_scalar(decom + f'train_loss_patches/transformer_lr', lr_dict['transformer'], iteration)
        if lr_dict is not None:
            for key, value in lr_dict.items():
                tb_writer.add_scalar(decom + f'1_train_loss_patches/zzlr_{key}', value, iteration)

        for key, value in loss_dict.items():
            if 'anl' in key:
                tb_writer.add_scalar(decom + f'compression_loss/{key}', value, iteration)

    # Report test and samples of training set
    test_psnr_ = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        scene.gaussians.eval()
        #
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, len(scene.getTrainCameras()), 20)]})
        com = 'decompress' if compress else ''
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                t_list = []
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint.cuda()
                    outputs = renderFunc(viewpoint, gaussian, *renderArgs, stage, evaluation=True, include_feature=kwargs['include_feature'])

                    image = torch.clamp(outputs["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    try:
                        if tb_writer and (idx % 5 == 0) and stage != 'coarse':
                            lang_image = torch.clamp(outputs["rendered_language_feature"], 0.0, 1.0)
                            tb_writer.add_images(config['name'] + com + decom + f"_view/render_{viewpoint.image_name}", lang_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                if config['name'] == 'test':
                    tqdm.write(com + decom + f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.5f} PSNR {psnr_test:.5f}")
                    test_psnr_ = psnr_test

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + com + decom + f'_eval/l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + com + decom + f'_eval/psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

    scene.gaussians.train()
    return test_psnr_


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


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations,
                         args, gaussians: GaussianModel, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    patch_range = (5, 17)
    opt.hard_depth_start = 0
    opt.soft_depth_start = 500
    opt.error_tolerance = 0.2

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    print(f"loaded {len(train_cams)} images")
    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=opt.batch_size, shuffle=True, num_workers=6, collate_fn=list)
        loader = iter(viewpoint_stack_loader)

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, final_iter+1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # dynerf's branch
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                loader = iter(viewpoint_stack_loader)

        # Render
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        opacity_list = []

        for idx, viewpoint_cam in enumerate(viewpoint_cams):
            render_pkg = stream_render_fisd(viewpoint_cam, gaussians, pipe, background, stage)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            images.append(image.unsqueeze(0))

            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))

            opacity_list.append(render_pkg["opacity"])

        gt_image_tensor = torch.cat(gt_images, 0)

        image_tensor = torch.cat(images, 0)
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        # Loss
        Ll1, Lssim, Ltssim = torch.tensor(0), torch.tensor(0), torch.tensor(0)

        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        # Lssim = 1.0 - ssim(image_tensor, gt_image_tensor)
        # loss = 0.8 * Ll1 + 0.2 * Lssim
        loss = Ll1
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss_dict = {'L1': Ll1, 'Lssim': Lssim, 'Ltssim': Ltssim}

        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        opacity_merged = torch.mean(torch.stack(opacity_list), dim=0)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "points": f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians,
                            stream_render_fisd, [pipe, background], stage)

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > 500:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                    gaussians.update_opacity_stats(opacity_merged, visibility_filter, opacity_threshold)

                opacity_threshold = opt.opacity_threshold_coarse
                densify_threshold = opt.densify_grad_threshold_coarse
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 80000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.save(dataset.model_path, name='coarse.pth', save_geometry_only=True)


def scene_reconstruction_with_dynamic(dataset, opt, hyper, pipe, testing_iterations,
                                      args, gaussians: GaussianModel, scene, stage, tb_writer, train_iter, timer):
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    lang_iter = 1000
    begin_lang = False
    final_iter = train_iter + lang_iter
    opt.densify_until_iter = train_iter

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=opt.batch_size, prefetch_factor=1, shuffle=True, num_workers=6, collate_fn=list, pin_memory=True)
        loader = iter(viewpoint_stack_loader)

    first_iter = 1
    progress_bar = tqdm(range(first_iter, final_iter + 1), desc="Coarse progress")
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # dynerf's branch
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                loader = iter(viewpoint_stack_loader)

        if iteration > train_iter and not begin_lang:
            begin_lang = True


        # Render
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        language_features = []
        gt_language_features = []
        language_feature_masks = []

        for idx, viewpoint_cam in enumerate(viewpoint_cams):
            viewpoint_cam.cuda()
            render_pkg = stream_render_fisd(viewpoint_cam, gaussians, pipe, background, stage, include_feature=begin_lang)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg[
                "visibility_filter"], render_pkg["radii"]
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            images.append(image.unsqueeze(0))

            gt_image = viewpoint_cam.original_image
            gt_images.append(gt_image.unsqueeze(0))

            if begin_lang:
                language_features.append(render_pkg["language_feature"].view(3, 3, gt_image.shape[1], gt_image.shape[2]).unsqueeze(0))
                gt_language_features.append(viewpoint_cam.lf_map.unsqueeze(0))
                language_feature_masks.append(viewpoint_cam.seg_map.unsqueeze(0))

        gt_image_tensor = torch.cat(gt_images, 0)

        image_tensor = torch.cat(images, 0)
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        if begin_lang:
            language_feature_tensor = torch.cat(language_features,0)
            language_feature_mask_tensor = torch.cat(language_feature_masks,0)
            gt_language_feature_tensor = torch.cat(gt_language_features,0)

        # Loss
        Ll1, Lssim, Ltssim = torch.tensor(0), torch.tensor(0), torch.tensor(0)

        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        # Lssim = 1.0 - ssim(image_tensor, gt_image_tensor)
        # loss = 0.8 * Ll1 + 0.2 * Lssim
        loss = Ll1

        lang_l1 = torch.tensor(0)
        if begin_lang:
            # lang_l1 = 0.2 * l1_loss(language_feature_tensor*language_feature_mask_tensor, gt_language_feature_tensor*language_feature_mask_tensor)
            lang_l1 = 0.2 * l1_loss(language_feature_tensor, gt_language_feature_tensor)
            loss += lang_l1

        loss_dict = {'L1': Ll1, 'Lssim': Lssim, 'Ltssim': lang_l1}

        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "lang": f"{lang_l1:.{4}f}",
                                          "points": f"{total_point}"})
                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians,
                            stream_render_fisd, [pipe, background], stage, include_feature=begin_lang)

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > 500:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter, None, gaussians._xyz.shape[0])

                opacity_threshold = opt.opacity_threshold_coarse
                densify_threshold = opt.densify_grad_threshold_coarse

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < opt.coarse_gaussian_num:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_static_only(densify_threshold, densify_threshold, scene.cameras_extent, opacity_threshold, opacity_threshold)

            # Optimizer step
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.save(dataset.model_path, name='coarse_lang.pth', save_geometry_only=True)

            if iteration == train_iter:
                gaussians.save(dataset.model_path, name='coarse.pth', save_geometry_only=True)

    gaussians.create_dynamic(gaussians.get_xyz)
    gaussians.training_dynamic_setup(opt)
    dy_iterations = 3000

    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=1, shuffle=True, num_workers=4, collate_fn=list)
        loader = iter(viewpoint_stack_loader)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]).cuda())

    progress_bar = tqdm(range(0, dy_iterations), desc="Dynamic progress")
    for iteration in range(1, dy_iterations + 1):
        gaussians.update_learning_dy_rate(iteration)
        try:
            viewpoint_cams = next(loader)
        except StopIteration:
            loader = iter(viewpoint_stack_loader)

        viewpoint_cam = viewpoint_cams[0]
        render_pkg = dynamic_render(viewpoint_cam, gaussians, pipe, background)
        dynamic_map = render_pkg["dynamic_map"].squeeze(0)
        dy_mask = viewpoint_cam.mask.cuda()
        loss = criterion(dynamic_map, dy_mask.float())

        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss:.{4}f}"})
                progress_bar.update(10)
            if iteration == dy_iterations:
                progress_bar.close()

            if iteration < dy_iterations:
                gaussians.dy_optimizer.step()
                gaussians.dy_optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.save(dataset.model_path, name='coarse_sd.pth', save_geometry_only=True)


def scene_static_dynamic_split(dataset, opt, hyper, pipe, args, gaussians: GaussianModel, scene, dy_iterations):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    print(f"loaded {len(train_cams)} images")

    gaussians.create_dynamic(gaussians.get_xyz)
    gaussians.training_dynamic_setup(opt)
    # dy_iterations = 3000

    viewpoint_stack_loader = DataLoader(train_cams, batch_size=1, shuffle=True, num_workers=4, collate_fn=list)
    loader = iter(viewpoint_stack_loader)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]).cuda())

    progress_bar = tqdm(range(0, dy_iterations), desc="Dynamic progress")
    for iteration in range(1, dy_iterations + 1):
        gaussians.update_learning_dy_rate(iteration)
        try:
            viewpoint_cams = next(loader)
        except StopIteration:
            loader = iter(viewpoint_stack_loader)

        viewpoint_cam = viewpoint_cams[0]
        render_pkg = dynamic_render(viewpoint_cam, gaussians, pipe, background)
        dynamic_map = render_pkg["dynamic_map"].squeeze(0)
        dy_mask = viewpoint_cam.mask.cuda()
        loss = criterion(dynamic_map, dy_mask.float())

        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss:.{4}f}"})
                progress_bar.update(10)
            if iteration == dy_iterations:
                progress_bar.close()

            if iteration < dy_iterations:
                gaussians.dy_optimizer.step()
                gaussians.dy_optimizer.zero_grad(set_to_none=True)
            else:
                # gaussians.save(dataset.model_path, name='coarse_sd.pth')
                pass


def scene_reconstruction_refine(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                                gaussians: GaussianModel, scene, stage, tb_writer, train_iter, gopid):
    first_iter = 0
    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_temporal_regularization = True

    dynamic_densify = False
    if stage == "following" and gaussians.dynamic_grid_sidelen == 0:
        dynamic_densify = True

    opacity_threshold = opt.opacity_threshold_static
    dynamic_opacity_threshold = opt.opacity_threshold_dynamic
    densify_threshold = opt.densify_grad_threshold_static
    dynamic_densify_threshold = opt.densify_grad_threshold_dynamic

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0

    lang_iter = 2000
    begin_lang = True
    final_iter = train_iter + lang_iter
    testing_iterations.append(final_iter)

    scene_name = dataset.model_path.split('/')[-2]
    # print(f'[INFO] scene name {scene_name}')
    lang_reg = -1
    if scene_name in ['coffee_martini', 'cook_spinach', 'flame_salmon', 'cut_roasted_beef', 'flame_steak']:
        lang_reg = 0
    elif scene_name in ['sear_steak']:
        lang_reg = 1
    print(f'[INFO] lang_reg={lang_reg}')

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    batch_size = opt.batch_size
    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=list, pin_memory=True)
        loader = iter(viewpoint_stack_loader)

    neighbor_loss_reg = False
    neighbor_loss_weight = hyper.neighbor_loss_weight
    if stage != "following":
        neighbor_loss_reg = True
    if stage == "following" and gaussians.offset_mode == -1:
        neighbor_loss_weight = hyper.dynamic_neighbor_loss_weight
        neighbor_loss_reg = True
    # if stage == "following" and gaussians.offset_mode >= 10:
    #     neighbor_loss_weight = hyper.static_neighbor_loss_weight
    #     neighbor_loss_reg = False
    neighbor_loss_weight_sum = sum(neighbor_loss_weight.values())

    temporal_reg = opt.temporal_reg
    best_test_psnr = 0.0
    best_decom_test_psnr = 0.0

    compression_iterations = []
    compr_results = None
    progress_bar = tqdm(range(first_iter, final_iter + 1), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        lr_dict = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # dynerf's branch
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                loader = iter(viewpoint_stack_loader)

        # if (iteration > 2000 and gaussians.offset_mode == 0) or (gaussians.offset_mode > 0):
        # if iteration > 3000:
        #     begin_lang = True

        # Render
        images = []
        images_t = []
        gt_images = []
        language_features = []
        gt_language_features = []
        language_feature_masks = []
        camera_list = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        opacity_list = []
        scale_list = []
        time_feat_loss = []
        gaussian_importance_list = []

        for idx, viewpoint_cam in enumerate(viewpoint_cams):
            viewpoint_cam.cuda()
            render_pkg = stream_render_fisd(viewpoint_cam, gaussians, pipe, background, stage, noise=None, include_feature=begin_lang)

            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

            opacity_list.append(render_pkg["opacity"])
            gaussian_importance_list.append(render_pkg["max_weight_t"])

            images.append(image.unsqueeze(0))

            gt_image = viewpoint_cam.original_image
            gt_images.append(gt_image.unsqueeze(0))

            if begin_lang:
                language_features.append(render_pkg["language_feature"].view(3, 3, gt_image.shape[1], gt_image.shape[2]).unsqueeze(0))
                gt_language_features.append(viewpoint_cam.lf_map.unsqueeze(0))
                language_feature_masks.append(viewpoint_cam.seg_map.unsqueeze(0))

        gt_image_tensor = torch.cat(gt_images, 0)
        image_tensor = torch.cat(images, 0)

        if begin_lang:
            language_feature_tensor = torch.cat(language_features,0)
            language_feature_mask_tensor = torch.cat(language_feature_masks,0)
            gt_language_feature_tensor = torch.cat(gt_language_features,0)

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        Lssim = torch.tensor(0)

        if stage != "following":
            if iteration > opt.stage[0]:
                Lssim = 1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :])
                loss = 0.8 * Ll1 + 0.2 * Lssim
            else:
                Lssim = 1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :])
                loss = 0.9 * Ll1 + 0.1 * Lssim
        else:
            Lssim = 1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :])
            loss = 0.8 * Ll1 + 0.2 * Lssim

        lang_l1 = torch.tensor(0)
        if begin_lang and lang_reg >= 0:
            if lang_reg:
                lang_reg = int(iteration // 2000) * 0.05  # for cut | steak | sear
            else:
                lang_reg = int(iteration // 2000 + 1) * 0.1  # for coffee | cook | salmon
            #
            lang_l1 = lang_reg * l1_loss(language_feature_tensor * language_feature_mask_tensor, gt_language_feature_tensor * language_feature_mask_tensor)
            loss += lang_l1

        loss_dict = {'L1': Ll1, 'Lssim': Lssim, 'Ltssim': lang_l1}

        neighbor_loss = []
        if neighbor_loss_reg and iteration > 600 and (gaussians._xyz.shape[0] == gaussians.grid_sidelen * gaussians.grid_sidelen):
            for attr_name, weight in neighbor_loss_weight.items():
                if weight > 0:
                    attr_neighbor_loss = gaussians.neighborloss_2d(attr_name, hyper) * weight / neighbor_loss_weight_sum
                    neighbor_loss.append(attr_neighbor_loss)
                    loss_dict[f"anl_{attr_name}"] = attr_neighbor_loss
            loss += opt.neighbor_reg * sum(neighbor_loss)

        if opt.static_latent_reg > 0 and gaussians.offset_mode >= 22 and iteration > opt.static_latent_reg_start:
            for idx, latent in enumerate([gaussians._features_dc_offset, gaussians._features_rest_offset, gaussians._scaling_offset, gaussians._rotation_offset, gaussians._opacity_offset, gaussians._language_feature_offset]):
                latent_reg = torch.abs(latent).mean()
                loss_dict[f'anl_{gaussians.quat_static_attrbutes[idx]}'] = latent_reg.detach()
                loss += opt.static_latent_reg * latent_reg

        if opt.dynamic_latent_reg > 0 and gaussians.offset_mode >= 22 and iteration > opt.dynamic_latent_reg_start:
            for idx, latent in enumerate([gaussians._features_dc_dynamic_offset, gaussians._features_rest_dynamic_offset, gaussians._scaling_dynamic_offset, gaussians._rotation_dynamic_offset, gaussians._opacity_dynamic_offset, gaussians._language_feature_dynamic_offset]):
                latent_reg = torch.abs(latent).mean()
                loss_dict[f'anl_{gaussians.quat_attrbutes[idx]}'] = latent_reg.detach()
                loss += opt.dynamic_latent_reg * latent_reg

        # temporal smooth regularization
        temp_loss = []
        if use_temporal_regularization and 2000 < iteration:
            # temp_loss = gaussians._point_feats[:, :1] - gaussians._point_feats[:, 1:]
            # temp_loss = torch.abs(temp_loss).norm(dim=-1).mean()

            # small delta, large video size
            if viewpoint_cam.time == 0 and gaussians.offset_mode != 0 and hasattr(gaussians, '_previous_last_feat'):
                temp_loss = F.huber_loss(gaussians._previous_last_feat, render_pkg["feats"][0], delta=0.01)
                # temp_loss = F.l1_loss(gaussians._previous_last_feat, render_pkg["feats"][0])
                # temp_loss = F.mse_loss(gaussians._previous_last_feat, render_pkg["feats"][0])
                temp_loss = temp_loss.mean()
            else:
                temp_loss = F.huber_loss(render_pkg["feats"][0], render_pkg["feats"][1], delta=0.01)
                # temp_loss = F.l1_loss(render_pkg["feats"][0], render_pkg["feats"][1])
                # temp_loss = F.mse_loss(render_pkg["feats"][0], render_pkg["feats"][1])
                temp_loss = temp_loss.mean()
            loss_dict['anl_temporal_loss'] = temp_loss
            loss += temporal_reg * temp_loss

        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        # viewspace_point_tensor_grad = torch.mean(torch.stack([tensor.grad for tensor in viewspace_point_tensor_list]), dim=0)
        # viewspace_point_tensor_grad = torch.stack([tensor.grad for tensor in viewspace_point_tensor_list]).max(dim=0).values

        gaussian_importance = torch.stack(gaussian_importance_list).max(dim=0).values

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            total_point = gaussians._xyz.shape[0]
            dy_points = gaussians._point_feats.shape[0]
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"L1": f"{ema_loss_for_log:.{4}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "lang": f"{lang_l1:.{4}f}",
                                          "tp": f"{total_point}",
                                          "dp": f"{dy_points}"})
                progress_bar.update(10)
            if iteration == final_iter + 1:
                progress_bar.close()

            # Log and save
            test_psnr = training_report(tb_writer, iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians,
                                        stream_render_fisd, [pipe, background], stage, lr_dict, include_feature=begin_lang)

            # Compression
            # test_psnr > compres_thres or
            if (( test_psnr > best_test_psnr) and gaussians._xyz_dynamic.shape[0] > 10000) or iteration == testing_iterations[-1]:
                if test_psnr > best_test_psnr:
                    best_test_psnr = test_psnr
                    best_iteration = iteration

                compr_path = os.path.join(dataset.model_path, "compression", f"iteration_{iteration}")
                compression_iterations.append(iteration)

                # enable compression of non-sorted gaussians without affecting results
                gaussians_to_compress = copy.deepcopy(gaussians)
                assert gaussians_to_compress.grid_sidelen * gaussians_to_compress.grid_sidelen == gaussians_to_compress._xyz.shape[0]

                compr_results = run_compressions(gaussians_to_compress, compr_path, OmegaConf.load(compression_config), gopid, qp=args.qp)

                if gopid == 0:
                    skip = []
                else:
                    skip = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_language_feature",
                            "_xyz_dynamic", "_features_dc_dynamic", "_features_rest_dynamic", "_scaling_dynamic", "_rotation_dynamic", "_opacity_dynamic", "_language_feature_dynamic"]
                # print(f"skip attributes:", skip)
                if os.stat(os.path.join(compr_results['out_path'], '_point_feats.mp4')).st_size > (1 * 1024):
                    experiment_name = os.listdir(compr_path)[0]
                    decompressed_gaussians = run_single_decompression(compr_results['out_path'], compr_results[str(experiment_name)], GaussianModel, gaussians_to_compress, skip)
                    decom_test_psnr = training_report(tb_writer, iteration, loss_dict, loss, iter_start.elapsed_time(iter_end), compression_iterations,
                                                      scene, decompressed_gaussians, stream_render_fisd, [pipe, background], stage, compress=True, include_feature=False)
                    if decom_test_psnr > best_decom_test_psnr:
                        best_decom_test_psnr = decom_test_psnr
                        compr_best_path = os.path.join(dataset.model_path, "compression", f"best")
                        if os.path.exists(compr_best_path):
                            os.remove(compr_best_path)
                        os.symlink(compr_path, compr_best_path)
                else:
                    tqdm.write(f"[ERROR] temporal feature video compression failed, skip testing!")

            # Densification
            if stage != "following" and iteration <= opt.stage[3]:  # opt.densify_until_iter
                if iteration > 400:
                    static_num = gaussians._xyz.shape[0]
                    static_vis_filter = visibility_filter[:static_num]
                    static_radii = radii[:static_num]
                    dynamic_vis_filter = visibility_filter[static_num:]
                    dynamic_radii = radii[static_num:]

                    gaussians.max_radii2D[static_vis_filter] = torch.max(gaussians.max_radii2D[static_vis_filter], static_radii[static_vis_filter])
                    gaussians.dynamic_max_radii2D[dynamic_vis_filter] = torch.max(gaussians.dynamic_max_radii2D[dynamic_vis_filter], dynamic_radii[dynamic_vis_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, static_vis_filter, dynamic_vis_filter, static_num)

                    gaussians._importance = torch.max(gaussians._importance, gaussian_importance[:static_num])
                    gaussians._importance_dynamic = torch.max(gaussians._importance_dynamic, gaussian_importance[static_num:])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if iteration >= 3000 and iteration % 1200 == 0:
                        gaussians.prune_low_importance(0.02, opt.num_gaussian)

                    size_threshold = None
                    if iteration % opt.static_densification_interval == 0:
                        gaussians.densify_static(densify_threshold, dynamic_densify_threshold, opt.num_gaussian, opt.num_gaussian2,
                                                 scene.cameras_extent, opacity_threshold, dynamic_opacity_threshold)  # densify

                    gaussians.densify_dynamic(densify_threshold, dynamic_densify_threshold, opt.num_gaussian, opt.num_gaussian2,
                                              scene.cameras_extent, opacity_threshold, dynamic_opacity_threshold)

                    if hyper.sorting_enabled:
                        gaussians.sort_into_grid(hyper, False)
                        if gaussians.get_dynamic_xyz.shape[0] > 10000:
                            gaussians.sort_dynamic_into_grid(hyper, False)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration > final_iter + 1:
            break


def scene_refine_language(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                          gaussians: GaussianModel, scene, stage, tb_writer, train_iter, gopid):
    first_iter = 0
    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"

    gaussians.create_language_learnable_features()
    gaussians.training_language_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_temporal_regularization = True

    ema_loss_for_log = 0.0

    final_iter = train_iter
    testing_iterations.append(final_iter)

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    batch_size = opt.batch_size
    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=list, pin_memory=True)
        loader = iter(viewpoint_stack_loader)

    temporal_reg = opt.temporal_reg
    # begin_lang = True
    best_test_psnr = 0.0
    best_decom_test_psnr = 0.0
    best_gaussian = None

    compression_iterations = []
    compr_results = None
    progress_bar = tqdm(range(first_iter, final_iter + 1), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, final_iter + 1):
        # lr_dict = gaussians.update_learning_rate(iteration)

        # dynerf's branch
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                loader = iter(viewpoint_stack_loader)

        # Render
        images = []
        images_t = []
        gt_images = []
        language_features = []
        gt_language_features = []
        language_feature_masks = []
        camera_list = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        opacity_list = []
        scale_list = []
        time_feat_loss = []
        gaussian_importance_list = []

        for idx, viewpoint_cam in enumerate(viewpoint_cams):
            viewpoint_cam.cuda()
            render_pkg = stream_render_lang(viewpoint_cam, gaussians, pipe, background, stage, noise=None, include_feature=True)

            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            viewspace_point_tensor_list.append(viewspace_point_tensor)

            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image
            gt_images.append(gt_image.unsqueeze(0))

            language_features.append(render_pkg["language_feature"].view(3, 3, gt_image.shape[1], gt_image.shape[2]).unsqueeze(0))
            gt_language_features.append(viewpoint_cam.lf_map.unsqueeze(0))
            language_feature_masks.append(viewpoint_cam.seg_map.unsqueeze(0))

        gt_image_tensor = torch.cat(gt_images, 0)
        image_tensor = torch.cat(images, 0)

        language_feature_tensor = torch.cat(language_features,0)
        language_feature_mask_tensor = torch.cat(language_feature_masks,0)
        gt_language_feature_tensor = torch.cat(gt_language_features,0)

        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        Lssim = torch.tensor(0)

        # lang_reg = int(iteration // 2000 + 1) * 0.04
        # lang_l1 = 0.2 * l1_loss(language_feature_tensor, gt_language_feature_tensor)  # coffee_martini 0.1
        lang_l1 = 0.2 * l1_loss(language_feature_tensor * language_feature_mask_tensor, gt_language_feature_tensor * language_feature_mask_tensor)
        loss = lang_l1

        loss_dict = {'L1': Ll1, 'Lssim': Lssim, 'Ltssim': lang_l1}

        if opt.static_latent_reg > 0 and gaussians.offset_mode >= 22 and iteration > opt.static_latent_reg_start:
            for idx, latent in enumerate([gaussians._language_feature_offset, gaussians._lang_opacity]):
                latent_reg = torch.abs(latent).mean()
                loss_dict[f'anl_{gaussians.quat_static_attrbutes[idx]}'] = latent_reg.detach()
                loss += opt.dynamic_latent_reg * latent_reg

        if opt.dynamic_latent_reg > 0 and gaussians.offset_mode >= 22 and iteration > opt.dynamic_latent_reg_start:
            for idx, latent in enumerate([gaussians._language_feature_dynamic_offset, gaussians._lang_opacity_dynamic]):
                latent_reg = torch.abs(latent).mean()
                loss_dict[f'anl_{gaussians.quat_attrbutes[idx]}'] = latent_reg.detach()
                loss += opt.dynamic_latent_reg * latent_reg

        # temporal smooth regularization
        temp_loss = []
        # temporal_reg = 1e-2
        if use_temporal_regularization and 2000 < iteration:
            # temp_loss = gaussians._point_feats[:, :1] - gaussians._point_feats[:, 1:]
            # temp_loss = torch.abs(temp_loss).norm(dim=-1).mean()

            # small delta, large video size
            if viewpoint_cam.time == 0 and gaussians.offset_mode != 0 and hasattr(gaussians, '_previous_last_feat'):
                temp_loss = F.huber_loss(gaussians._previous_last_feat[:, 12:16], render_pkg["feats"][0], delta=0.01)
                # temp_loss = F.l1_loss(gaussians._previous_last_feat, render_pkg["feats"][0])
                # temp_loss = F.mse_loss(gaussians._previous_last_feat, render_pkg["feats"][0])
                temp_loss = temp_loss.mean()
            else:
                temp_loss = F.huber_loss(render_pkg["feats"][0], render_pkg["feats"][1], delta=0.01)
                # temp_loss = F.l1_loss(render_pkg["feats"][0], render_pkg["feats"][1])
                # temp_loss = F.mse_loss(render_pkg["feats"][0], render_pkg["feats"][1])
                temp_loss = temp_loss.mean()
            loss_dict['anl_temporal_loss'] = temp_loss
            loss += temporal_reg * temp_loss

        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            total_point = gaussians._xyz.shape[0]
            dy_points = gaussians._point_feats.shape[0]
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"L1": f"{ema_loss_for_log:.{4}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "lang": f"{lang_l1:.{4}f}",
                                          "tp": f"{total_point}",
                                          "dp": f"{dy_points}"})
                progress_bar.update(10)
            if iteration == final_iter + 1:
                progress_bar.close()

            # Log and save
            test_psnr = training_report(tb_writer, iteration, loss_dict, loss, 0, testing_iterations, scene, gaussians,
                                        stream_render_lang, [pipe, background], stage, None, include_feature=True)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.lang_optimizer.step()
                gaussians.lang_optimizer.zero_grad(set_to_none=True)

    compr_path = os.path.join(dataset.model_path, "compression", "best")
    gaussians._point_feats[:,:,12:16] = gaussians._lang_temp_feats
    # if gaussians.opacity_dim == 2:
    #     if gaussians.offset_mode == 0:
    #         gaussians._opacity[:,1:2] = gaussians._lang_opacity
    #         gaussians._opacity_dynamic[:,1:2] = gaussians._lang_opacity_dynamic
    #     else:
    #         gaussians._opacity_offset[:,1:2] = gaussians._lang_opacity
    #         gaussians._opacity_dynamic_offset[:,1:2] = gaussians._lang_opacity_dynamic
    print(f"saved gaussians with mode {gaussians.offset_mode}")
    compr_results = run_compressions(gaussians, compr_path, OmegaConf.load(compression_config), gopid, qp=args.qp)


def training_coarse(dataset, hyper, opt, pipe, args, expname, postfix, gop, coarse):
    tb_writer = prepare_output_and_logger(os.path.join(expname, f'coarse' if postfix is None else f'coarse_{postfix}'))
    gaussians = GaussianModel(dataset.sh_degree, True)
    dataset.model_path = args.model_path
    timer = Timer()
    if coarse:
        scene = Scene(dataset, gaussians, duration=[0, int(dataset.num_times)], timedordered=False)
    else:
        scene = Scene(dataset, gaussians, duration=[0, gop], timedordered=False)

    args.test_iterations = [i for i in range(1000, opt.coarse_iterations+1, 500)]
    opt.batch_size = 4
    timer.start()
    scene_reconstruction_with_dynamic(dataset, opt, hyper, pipe, args.test_iterations, args, gaussians, scene, 'coarse', tb_writer, opt.coarse_iterations, timer)


def training_refine(dataset, hyper, opt, pipe, compres_thres, checkpoint_path, args, expname, postfix, idx, gop=60):
    args.model_path = None
    tb_writer = prepare_output_and_logger(os.path.join(expname, f'gop{idx}' if postfix is None else f'gop{idx}_{postfix}_coarse'))
    gaussians = GaussianModel(dataset.sh_degree, True)
    dataset.model_path = args.model_path
    scene = Scene(dataset, gaussians, duration=[idx*gop, (idx+1)*gop], skip=args.skip, timedordered=False, skip_init=True)

    assert os.path.exists(os.path.join(checkpoint_path, "coarse_sd.pth")), f"checkpoint not found in {checkpoint_path}!"
    checkpoint = torch.load(os.path.join(checkpoint_path, "coarse_sd.pth"), map_location='cuda')
    gaussians.create_from_coarse(checkpoint, scene.cameras_extent, scene.maxtime)
    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.setup_interpolators(opt.position_erp, opt.rotation_erp)

    torch.cuda.empty_cache()

    testing_iterations = [i for i in range(1000, args.iterations+1, 1000)]

    scene_reconstruction_refine(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                                gaussians, scene, 'refine', tb_writer, opt.refine_iterations, gopid=idx)


def training_refine_language(dataset, hyper, opt, pipe, compres_thres, checkpoint_path, args, expname, postfix, idx, gop=60):
    args.model_path = None
    tb_writer = prepare_output_and_logger(os.path.join(expname, f'gop{idx}' if postfix is None else f'gop{idx}_{postfix}'))
    # tb_writer = prepare_output_and_logger(os.path.join(expname, f'gop{idx}' if postfix is None else f'gop{idx}_{postfix}_lang'))
    gaussians = GaussianModel(dataset.sh_degree, True)
    dataset.model_path = args.model_path

    scene = Scene(dataset, gaussians, duration=[idx*gop, (idx+1)*gop], skip=args.skip, timedordered=False, skip_init=True)

    if idx == 0:
        assert os.path.exists(os.path.join(checkpoint_path, "model.pth")), f"checkpoint not found in {checkpoint_path}!"
        checkpoint = torch.load(os.path.join(checkpoint_path, "model.pth"), map_location='cuda')
        # opacity_dim = checkpoint['_opacity'].shape[1]
        # checkpoint = compose_compressed_gs_attrs(args, checkpoint_path, gaussians, opacity_dim)
        gaussians.offset_mode = 0
        stage = 'refine'
    else:
        pre_lang_ckpt = torch.load(os.path.join(expname, f'gop{idx - 1}_{postfix}', 'compression/best/jxl_quant', 'model.pth'), map_location='cuda')
        # pre_lang_ckpt = torch.load(os.path.join(expname, f'gop{idx - 1}_{postfix}_lang', 'compression/best/jxl_quant', 'model.pth'), map_location='cuda')
        if '_xyz_offset' not in pre_lang_ckpt.keys():
            language_feature = pre_lang_ckpt['_language_feature'].float().cuda()
            dynamic_language_feature = pre_lang_ckpt['_language_feature_dynamic'].float().cuda()
        else:
            language_feature = pre_lang_ckpt['_language_feature_offset'].float().cuda()
            dynamic_language_feature = pre_lang_ckpt['_language_feature_dynamic'].float().cuda()

        gaussians._language_feature = nn.Parameter(language_feature.requires_grad_(False))
        gaussians._language_feature_dynamic = nn.Parameter(dynamic_language_feature.requires_grad_(False))
        del pre_lang_ckpt

        assert os.path.exists(os.path.join(checkpoint_path, "model_offset.pth")), f"checkpoint not found in {checkpoint_path}!"
        checkpoint = torch.load(os.path.join(checkpoint_path, "model_offset.pth"), map_location='cuda')
        gaussians.offset_mode = 22
        stage = 'following'
    # print(checkpoint.keys())

    gaussians.create_from_ckpt_lang(checkpoint, scene.cameras_extent, scene.maxtime)

    if idx > 0:
        print(f'load offset')
        skip = ['_language_feature_offset', '_language_feature_dynamic_offset']  # '_language_feature_offset', '_language_feature_dynamic_offset'
        gaussians.load_compressed_offset(os.path.join(checkpoint_path, "offset_compressed.pkl"),skip=skip)

    gaussians.mlp_deform.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_deform.pth"), map_location="cuda"))
    gaussians.mlp_cov.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_cov.pth"), map_location="cuda"))
    gaussians.mlp_opacity.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_opacity.pth"), map_location="cuda"))
    gaussians.mlp_color.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_color.pth"), map_location="cuda"))
    gaussians.mlp_lang.load_state_dict(torch.load(os.path.join(checkpoint_path, "mlp_lang.pth"), map_location="cuda"))

    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.setup_interpolators(opt.position_erp, opt.rotation_erp)
    print(f"[INFO] training model is {gaussians.offset_mode}")

    torch.cuda.empty_cache()
    testing_iterations = [i for i in range(1000, args.iterations + 1, 1000)]
    # testing_iterations = []
    scene_refine_language(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                          gaussians, scene, stage, tb_writer, 3000, gopid=idx)


def compose_compressed_gs_attrs(args, checkpoint_path, gaussians, opacity_dim):
    from compression.compression_exp import decompress_geo_attr

    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"
    compr_exp_config = OmegaConf.load(compression_config)
    experiment = compr_exp_config['experiments'][0]

    compr_info = pd.read_csv(os.path.join(checkpoint_path, "compression_info.csv"), index_col=0)

    checkpoint = OrderedDict()
    for attribute in experiment['attributes']:
        attr_name = attribute["name"]
        if 'point_feat' not in attr_name:
            compressed_file = os.path.join(checkpoint_path, compr_info.loc[attr_name, "file"])
            attr_tensor = decompress_geo_attr(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"], opacity_dim)
            checkpoint[attr_name] = attr_tensor
        else:
            compressed_file = os.path.join(checkpoint_path, f'_point_feats.mp4' if args.qp == 6 else f'_point_feats_{args.qp}.mp4')
            attr_tensor = decompress_geo_attr(gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"], opacity_dim)
            checkpoint[attr_name] = attr_tensor
    return checkpoint

def training_following(dataset, hyper, opt, pipe, compres_thres, checkpoint_path, args, expname, postfix, idx, gop=60):
    args.model_path = None
    tb_writer = prepare_output_and_logger(os.path.join(expname, f'gop{idx}' if postfix is None else f'gop{idx}_{postfix}'))
    gaussians = GaussianModel(dataset.sh_degree, True)
    dataset.model_path = args.model_path

    scene = Scene(dataset, gaussians, duration=[idx*gop, (idx+1)*gop], skip=args.skip, timedordered=False, skip_init=True)

    if idx == 1:  # >= 1 simulate parallel training, ==1 is normal mode
        assert os.path.exists(os.path.join(checkpoint_path, "model.pth")), f"checkpoint not found in {checkpoint_path}!"
        checkpoint = torch.load(os.path.join(checkpoint_path, "model.pth"), map_location='cuda')
        opacity_dim = checkpoint['_opacity'].shape[1]
        checkpoint = compose_compressed_gs_attrs(args, checkpoint_path, gaussians, opacity_dim)
    else:
        assert os.path.exists(os.path.join(checkpoint_path, "model_offset.pth")), f"checkpoint not found in {checkpoint_path}!"
        checkpoint = torch.load(os.path.join(checkpoint_path, "model_offset.pth"), map_location='cuda')
    # print(f'[INFO] checkpoint keys: {checkpoint.keys()}')
    gaussians.create_from_ckpt(checkpoint, scene.cameras_extent, scene.maxtime, preserve=2.1)  # 0.1, 1, 1.1 (freeze static Gaussian) | 2, 2.1 (refine static Gaussian)
    gaussians.active_sh_degree = dataset.sh_degree
    gaussians.setup_interpolators(opt.position_erp, opt.rotation_erp)
    print(f"[INFO] training model is {gaussians.offset_mode}")

    torch.cuda.empty_cache()

    testing_iterations = [i for i in range(1000, args.iterations+1, 1000)]

    scene_reconstruction_refine(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                                gaussians, scene, 'following', tb_writer, opt.refine_iterations, gopid)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    # setup_seed
    seed = 6666
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1500, 3000, 4500, 6000, 8000, 10000, 15000])
    parser.add_argument("--compres_thres", type=float, default=33.3)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--compre_config", type=str, default="")
    parser.add_argument("--qp", type=int, default=6)
    parser.add_argument("--compres_refine", action="store_true")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--language", action="store_true")
    parser.add_argument("--gopids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--segment", type=int, default=300, help="fixed")
    parser.add_argument("--total", action="store_true")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.coarse:
        training_coarse(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args, args.expname, args.postfix, args.gop, args.coarse)
    else:
        gop_nums = args.segment // args.gop
        for gopid in args.gopids:
            segment_id = gopid // gop_nums
            if segment_id == 0:
                if gopid == 0:
                    print(f"segment_id={segment_id}, gopid={gopid}, training_refine")
                    if not os.path.exists(os.path.join(args.expname, f"gop0_{args.postfix}_coarse")):
                        training_refine(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, args.checkpoint_path,
                                        args, args.expname, args.postfix, gopid, args.gop)

                    if not os.path.exists(os.path.join(args.expname, f"gop0_{args.postfix}")):
                        checkpoint_path = os.path.join(args.expname, f"gop0_{args.postfix}_coarse/compression/best/jxl_quant")
                        training_refine_language(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, checkpoint_path,
                                                 args, args.expname, args.postfix, gopid, args.gop)
                else:
                    print(f"segment_id={segment_id}, gopid={gopid}, training_following")
                    if not os.path.exists(os.path.join(args.expname, f"gop{gopid}_{args.postfix}")):
                        training_following(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, args.checkpoint_path,
                                           args, args.expname, args.postfix, gopid, args.gop)
                    # checkpoint_path = os.path.join(args.expname, f"gop{gopid}_{args.postfix}/compression/best/jxl_quant")
                    # training_refine_language(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, checkpoint_path,
                    #                          args, args.expname, args.postfix, gopid, args.gop)
            else:
                if gopid % gop_nums == 0:
                    print(f"segment_id={segment_id}, gopid={gopid}, training_following_segment")
                    training_following(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, args.checkpoint_path,
                                       args, args.expname, args.postfix, gopid, args.gop)
                else:
                    print(f"segment_id={segment_id}, gopid={gopid}, training_following")
                    training_following(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, args.checkpoint_path,
                                       args, args.expname, args.postfix, gopid, args.gop)


    # All done
    print("\nTraining complete.")
