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
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from omegaconf import DictConfig, OmegaConf

from fused_ssim import fused_ssim as ssim
# from utils.loss_utils import ssim

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from scene import Scene
from scene.scaffold_gaussian_model import GaussianModel
from gaussian_renderer import hac_render, prefilter_voxel, hac_dynamic_render

from utils.general_utils import safe_state, DecayScheduler
from utils.loss_utils import l1_loss, l1_loss_mask, lpips_loss
from utils.image_utils import psnr
from utils.timer import Timer
from utils.encodings import anchor_round_digits, Q_anchor, encoder_anchor, get_binary_vxl_size
from lpipsPyTorch import lpips

from compression.compression_exp import run_compressions_hac, run_single_decompression_hac


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

bit2MB_scale = 8 * 1024 * 1024

def training_report(tb_writer, logger, iteration, loss_dict, loss, testing_iterations, scene: Scene, gaussian: GaussianModel,
                    renderFunc, renderArgs, stage, lr_dict=None, compress=False, decom_stage=False, training=None, **kwargs):
    decom = 'refined' if decom_stage else ''
    if tb_writer and iteration > 49:
        tb_writer.add_scalar(decom + f'1_train_loss_patches/l1_loss', loss_dict['L1'].item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/ssim_loss', loss_dict['Lssim'].item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/langl1_loss', loss_dict['Ltssim'].item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(decom + f'1_train_loss_patches/total_points', gaussian.get_anchor.shape[0], iteration)

        for key, value in loss_dict.items():
            if 'anl' in key:
                tb_writer.add_scalar(decom + f'compression_loss/{key}', value, iteration)

    # Report test and samples of training set
    test_psnr_ = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        gaussian.eval()
        #
        if decom_stage:
            if stage == 'refine':
                with torch.no_grad():
                    log_info = gaussian.estimate_final_bits()
                    logger.info(log_info)

        if stage == 'coarse':
            # validation_configs = ({'name': 'test', 'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(0, len(scene.getTestCameras()), 100)]},
            #                       {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(1, len(scene.getTrainCameras()), 300)]})
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                                  {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(1, len(scene.getTrainCameras()), 300)]})
        else:
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                                  {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(1, len(scene.getTrainCameras()), 300)]})
        com = 'decompress' if compress else ''
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                t_list = []
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint.cuda()
                    voxel_visible_mask = prefilter_voxel(viewpoint, gaussian, *renderArgs)
                    outputs = renderFunc(viewpoint, gaussian, *renderArgs, stage, visible_mask=voxel_visible_mask, training=training)

                    image = torch.clamp(outputs["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    try:
                        if tb_writer and (idx % 30 == 0) and stage == 'coarse':
                            # lang_image = torch.clamp(outputs["rendered_language_feature"], 0.0, 1.0)
                            tb_writer.add_images(config['name'] + com + decom + f"_view/render_{viewpoint.image_name}", image[None], global_step=iteration)
                            pass
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

    gaussian.train()
    return test_psnr_


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./", unique_str)
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

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(args.model_path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    return tb_writer, logger


def release_logger():
    # release
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations,
                         args, gaussians: GaussianModel, scene, stage, tb_writer, train_iter, logger):
    first_iter = 0

    gaussians.update_anchor_bound()
    gaussians.set_steps(opt.step_flag1, opt.step_flag2)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    train_cams.dataset.read_clip_features(False)  # TODO
    print(f"loaded {len(train_cams)} images")

    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=opt.batch_size, shuffle=True, num_workers=6, collate_fn=list)
        loader = iter(viewpoint_stack_loader)

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, final_iter+1):
        gaussians.update_learning_rate(iteration)

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
            viewpoint_cam.cuda()
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            retain_grad = (opt.densify_until_iter > iteration >= opt.densify_from_iter - 100)
            render_pkg = hac_render(viewpoint_cam, gaussians, pipe, background, stage, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)

            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

            bit_per_param = render_pkg["bit_per_param"]
            bit_per_feat_param = render_pkg["bit_per_feat_param"]
            bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
            bit_per_offsets_param = render_pkg["bit_per_offsets_param"]

            # visibility_filter_list.append(visibility_filter.unsqueeze(0))
            # viewspace_point_tensor_list.append(viewspace_point_tensor)
            # images.append(image.unsqueeze(0))

            gt_image = viewpoint_cam.original_image
            # gt_images.append(gt_image.unsqueeze(0))

            # opacity_list.append(render_pkg["opacity"])

        # gt_image_tensor = torch.cat(gt_images, 0)
        if iteration % 2000 == 0 and bit_per_param is not None:
            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

            tqdm.write("----------------------------------------------------------------------------------------")
            tqdm.write("-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), gaussians.get_anchor.shape[0], ttl_size_feat_MB))
            tqdm.write("-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), gaussians.get_anchor.shape[0], ttl_size_scaling_MB))
            tqdm.write("-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), gaussians.get_anchor.shape[0], ttl_size_offsets_MB))
            tqdm.write("-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----\n".format(iteration, bit_per_param.item(), gaussians.get_anchor.shape[0], ttl_size_MB))

        # image_tensor = torch.cat(images, 0)

        # visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        # Loss
        Ll1, Lssim, Ltssim = torch.tensor(0), torch.tensor(0), torch.tensor(0)

        Ll1 = l1_loss(image, gt_image[:3, :, :])
        # Lssim = 1.0 - ssim(image_tensor, gt_image_tensor)
        # loss = 0.8 * Ll1 + 0.2 * Lssim
        loss = Ll1
        psnr_ = psnr(image, gt_image).mean().double()

        # scale loss
        scaling_reg = scaling.prod(dim=1).mean()
        loss += 0.01 * scaling_reg

        # rd loss
        if bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians._anchor.shape[0] * (gaussians.feat_dim + 6 + 3 * gaussians.n_offsets)
            loss = loss + args.lmbda * (bit_per_param + bit_hash_grid / denom)

        loss_dict = {'L1': Ll1, 'Lssim': Lssim, 'Ltssim': Ltssim}

        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        # for idx in range(0, len(viewspace_point_tensor_list)):
        #     viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        # opacity_merged = torch.mean(torch.stack(opacity_list), dim=0)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._anchor.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "anchors": f"{total_point}"})
                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, logger, iteration, loss_dict, loss, testing_iterations, scene, gaussians,
                            hac_render, [pipe, background], stage)

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > 500:
                    gaussians.training_statis_static(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                opacity_threshold = opt.opacity_threshold_coarse
                densify_threshold = opt.densify_grad_threshold_coarse
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration not in range(opt.step_flag2, opt.step_flag2+1000) and gaussians._anchor.shape[0] < 80000:
                    gaussians.adjust_anchor(check_interval=opt.densification_interval, success_threshold=opt.success_threshold, grad_threshold=densify_threshold, min_opacity=opacity_threshold)

            # Optimizer step
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.save_ply(os.path.join(dataset.model_path, "point_cloud.ply"))
                gaussians.save_mlp_checkpoints(os.path.join(dataset.model_path, "checkpoint.pth"))
                gaussians.save(os.path.join(dataset.model_path, "coarse.pth"), canonical=True)


def scene_reconstruction_with_dynamic(dataset, opt, hyper, pipe, testing_iterations,
                                      args, gaussians: GaussianModel, scene, stage, tb_writer, train_iter, logger):
    gaussians.update_anchor_bound()
    gaussians.set_steps(opt.step_flag1, opt.step_flag2)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    lang_iter = 0
    begin_lang = False
    final_iter = train_iter + lang_iter
    opt.densify_until_iter = train_iter

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    train_cams.dataset.read_clip_features(args.language)
    assert opt.batch_size == 1
    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=opt.batch_size, prefetch_factor=1, shuffle=True, num_workers=6, collate_fn=list, pin_memory=True)
        loader = iter(viewpoint_stack_loader)

    first_iter = 1
    progress_bar = tqdm(range(first_iter, final_iter + 1), desc="Coarse progress")
    for iteration in range(first_iter, final_iter + 1):

        gaussians.update_learning_rate(iteration)

        # dynerf's branch
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                loader = iter(viewpoint_stack_loader)

        if args.language and iteration > (train_iter - 1000) and not begin_lang:
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
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            retain_grad = (opt.densify_until_iter > iteration >= opt.densify_from_iter - 100)
            render_pkg = hac_render(viewpoint_cam, gaussians, pipe, background, stage, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)

            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

            bit_per_param = render_pkg["bit_per_param"]
            bit_per_feat_param = render_pkg["bit_per_feat_param"]
            bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
            bit_per_offsets_param = render_pkg["bit_per_offsets_param"]

            # radii_list.append(radii.unsqueeze(0))
            # visibility_filter_list.append(visibility_filter.unsqueeze(0))
            # viewspace_point_tensor_list.append(viewspace_point_tensor)
            # images.append(image.unsqueeze(0))

            gt_image = viewpoint_cam.original_image
            # gt_images.append(gt_image.unsqueeze(0))

            if begin_lang:
                language_features.append(render_pkg["language_feature"].view(3, 3, gt_image.shape[1], gt_image.shape[2]).unsqueeze(0))
                gt_language_features.append(viewpoint_cam.lf_map.unsqueeze(0))
                language_feature_masks.append(viewpoint_cam.seg_map.unsqueeze(0))

        if iteration % 2000 == 0 and bit_per_param is not None:
            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

            tqdm.write("----------------------------------------------------------------------------------------")
            tqdm.write("-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), gaussians.get_anchor.shape[0], ttl_size_feat_MB))
            tqdm.write("-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), gaussians.get_anchor.shape[0], ttl_size_scaling_MB))
            tqdm.write("-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), gaussians.get_anchor.shape[0], ttl_size_offsets_MB))
            tqdm.write("-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----\n".format(iteration, bit_per_param.item(), gaussians.get_anchor.shape[0], ttl_size_MB))

        # gt_image_tensor = torch.cat(gt_images, 0)
        # image_tensor = torch.cat(images, 0)
        # radii = torch.cat(radii_list, 0).max(dim=0).values
        # visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        if begin_lang:
            language_feature_tensor = torch.cat(language_features,0)
            language_feature_mask_tensor = torch.cat(language_feature_masks,0)
            gt_language_feature_tensor = torch.cat(gt_language_features,0)

        # Loss
        Ll1, Lssim, Ltssim = torch.tensor(0), torch.tensor(0), torch.tensor(0)

        Ll1 = l1_loss(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = Ll1
        Lssim = 1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        if iteration > args.step_flag2:
            loss = 0.8 * Ll1 + 0.2 * Lssim
        else:
            loss = 0.8 * Ll1 + 0.1 * Lssim

        lang_l1 = torch.tensor(0)
        if begin_lang:
            # lang_l1 = 0.2 * l1_loss(language_feature_tensor*language_feature_mask_tensor, gt_language_feature_tensor*language_feature_mask_tensor)
            lang_l1 = 0.2 * l1_loss(language_feature_tensor, gt_language_feature_tensor)
            loss += lang_l1

        # scale loss
        scaling_reg = scaling.prod(dim=1).mean()
        loss += 0.01 * scaling_reg

        # rd loss
        if bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians._anchor.shape[0] * (gaussians.feat_dim + 6 + 3 * gaussians.n_offsets)
            loss = loss + args.lmbda * (bit_per_param + bit_hash_grid / denom)

        loss_dict = {'L1': Ll1, 'Lssim': Lssim, 'Ltssim': lang_l1}

        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        # for idx in range(0, len(viewspace_point_tensor_list)):
        #     viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            psnr_ = psnr(image, gt_image).mean().double()
            total_point = gaussians._anchor.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "lang": f"{lang_l1:.{4}f}",
                                          "anchors": f"{total_point}"})
                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, logger, iteration, loss_dict, loss, testing_iterations, scene, gaussians,
                            hac_render, [pipe, background], stage, include_feature=begin_lang)

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > 500:
                    gaussians.training_statis_static(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                opacity_threshold = opt.opacity_threshold_coarse
                densify_threshold = opt.densify_grad_threshold_coarse

                if iteration > opt.densify_from_iter and iteration % opt.static_densification_interval == 0 and iteration not in range(opt.step_flag2, opt.step_flag2+1000):
                    if gaussians._anchor.shape[0] < args.coarse_gaussian_num or args.coarse_gaussian_num == -1:
                        gaussians.adjust_anchor(check_interval=opt.static_densification_interval, success_threshold=opt.success_threshold, grad_threshold=densify_threshold, min_opacity=opacity_threshold)

            # Optimizer step
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.save(os.path.join(dataset.model_path, "coarse_lang.pth"), canonical=True)

            if iteration == train_iter:
                gaussians.save(os.path.join(dataset.model_path, "coarse.pth"), canonical=True)

    gaussians.create_dynamic()
    gaussians.training_dynamic_setup(opt)
    dy_iterations = 3000

    train_cams.dataset.read_clip_features(False)
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
        viewpoint_cam.cuda()
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        render_pkg = hac_dynamic_render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask)
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
                gaussians.save(os.path.join(dataset.model_path, 'coarse_sd.pth'), canonical=True)


def scene_reconstruction_refine(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                                gaussians: GaussianModel, scene, stage, train_iter, gopid, logger, tb_writer=None):
    first_iter = 0
    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"

    if stage == 'refine' or stage == 'coarse':
        opt.step_flag1=0
        opt.step_flag2=-1000
        gaussians.set_steps(opt.step_flag1, opt.step_flag2)
        gaussians.training_setup(opt)
        gaussians.train()
    elif stage == 'following':
        gaussians.training_following_setup(opt)
        gaussians.train()
    else:
        raise NotImplementedError

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_temporal_regularization = True

    begin_lang = args.language
    final_iter = train_iter
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
    train_cams.dataset.read_clip_features(begin_lang)
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

    neighbor_loss_weight_sum = sum(neighbor_loss_weight.values())

    temporal_reg = opt.temporal_reg
    ema_loss_for_log = 0.0
    best_test_psnr = 0.0
    best_decom_test_psnr = 0.0

    compression_iterations = []
    compr_results = None
    progress_bar = tqdm(range(first_iter, final_iter + 1), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, final_iter + 1):
        if stage == 'following':
            gaussians.update_learning_rate_following(iteration)
        else:
            gaussians.update_learning_rate(iteration)
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
        offset_selection_mask_list = []
        voxel_visible_mask_list = []
        opacity_list = []
        scale_list = []
        time_batch = []
        bit_per_param_list, bit_per_feat_param_list, bit_per_scaling_param_list, bit_per_offsets_param_list = [], [], [], []

        for idx, viewpoint_cam in enumerate(viewpoint_cams):
            viewpoint_cam.cuda()

            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            retain_grad = (opt.densify_until_iter > iteration >= opt.densify_from_iter - 100)
            render_pkg = hac_render(viewpoint_cam, gaussians, pipe, background, stage, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)

            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if stage != 'following':
                offset_selection_mask = render_pkg["selection_mask"]
                offset_selection_mask_list.append(offset_selection_mask)

                opacity_list.append(render_pkg["neural_opacity"])

                bit_per_param_list.append(render_pkg["bit_per_param"])
                bit_per_feat_param_list.append(render_pkg["bit_per_feat_param"])
                bit_per_scaling_param_list.append(render_pkg["bit_per_scaling_param"])
                bit_per_offsets_param_list.append(render_pkg["bit_per_offsets_param"])

            # radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter)
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            voxel_visible_mask_list.append(voxel_visible_mask)

            scale_list.append(render_pkg["scaling"])

            time_batch.append(viewpoint_cam.time)

            images.append(image.unsqueeze(0))

            gt_image = viewpoint_cam.original_image
            gt_images.append(gt_image.unsqueeze(0))

            if begin_lang:
                language_features.append(render_pkg["language_feature"].view(3, 3, gt_image.shape[1], gt_image.shape[2]).unsqueeze(0))
                gt_language_features.append(viewpoint_cam.lf_map.unsqueeze(0))
                language_feature_masks.append(viewpoint_cam.seg_map.unsqueeze(0))

        if stage != 'following' and bit_per_param_list[0] is not None:
            bit_per_param = torch.mean(torch.stack(bit_per_param_list, dim=0), dim=0)
            bit_per_feat_param = torch.mean(torch.stack(bit_per_feat_param_list, dim=0), dim=0)
            bit_per_scaling_param = torch.mean(torch.stack(bit_per_scaling_param_list, dim=0), dim=0)
            bit_per_offsets_param = torch.mean(torch.stack(bit_per_offsets_param_list, dim=0), dim=0)
            if iteration % 2000 == 0:
                ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
                ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
                ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
                ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

                tqdm.write("----------------------------------------------------------------------------------------")
                tqdm.write("-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), gaussians.get_anchor.shape[0], ttl_size_feat_MB))
                tqdm.write("-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), gaussians.get_anchor.shape[0], ttl_size_scaling_MB))
                tqdm.write("-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), gaussians.get_anchor.shape[0], ttl_size_offsets_MB))
                tqdm.write("-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----\n".format(iteration, bit_per_param.item(), gaussians.get_anchor.shape[0], ttl_size_MB))

        gt_image_tensor = torch.cat(gt_images, 0)
        image_tensor = torch.cat(images, 0)

        if begin_lang:
            language_feature_tensor = torch.cat(language_features,0)
            language_feature_mask_tensor = torch.cat(language_feature_masks,0)
            gt_language_feature_tensor = torch.cat(gt_language_features,0)

        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        Lssim = 1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :])
        loss = 0.8 * Ll1 + 0.2 * Lssim

        scaling_reg = 0.0
        if stage == 'refine':
            for scaling in scale_list:
                scaling_reg += scaling.prod(dim=1).mean()
            loss += 0.01 * (scaling_reg / len(scale_list))

        # rd loss
        if stage != 'following' and bit_per_param_list[0] is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians.get_anchor_num * (gaussians.feat_dim + 6 + 3 * gaussians.n_offsets)
            loss = loss + args.lmbda * (bit_per_param + bit_hash_grid / denom)
        elif stage == 'following':
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_ntc_2D_params() + 1) / 2)
            denom = gaussians.get_anchor_num * (gaussians.feat_dim + 3 * gaussians.n_offsets)
            loss = loss + args.lmbda * (bit_hash_grid / denom)

        lang_l1 = torch.tensor(0)
        if begin_lang and lang_reg >= 0:
            if lang_reg:
                lang_reg = int(iteration // 2000) * 0.05  # for cut | steak | sear
            else:
                lang_reg = int(iteration // 2000 + 1) * 0.1  # for coffee | cook | salmon

            lang_l1 = lang_reg * l1_loss(language_feature_tensor * language_feature_mask_tensor, gt_language_feature_tensor * language_feature_mask_tensor)
            loss += lang_l1

        loss_dict = {'L1': Ll1, 'Lssim': Lssim, 'Ltssim': lang_l1}

        # neighbor_loss = []
        # if neighbor_loss_reg and iteration > 600 and (gaussians._xyz.shape[0] == gaussians.grid_sidelen * gaussians.grid_sidelen):
        #     for attr_name, weight in neighbor_loss_weight.items():
        #         if weight > 0:
        #             attr_neighbor_loss = gaussians.neighborloss_2d(attr_name, hyper) * weight / neighbor_loss_weight_sum
        #             neighbor_loss.append(attr_neighbor_loss)
        #             loss_dict[f"anl_{attr_name}"] = attr_neighbor_loss
        #     loss += opt.neighbor_reg * sum(neighbor_loss)

        # temporal smooth regularization
        temp_loss = []
        if use_temporal_regularization and iteration > 1000:
            # temp_loss = gaussians._point_feats[:, :1] - gaussians._point_feats[:, 1:]
            # temp_loss = torch.abs(temp_loss).norm(dim=-1).mean()
            if viewpoint_cam.time == 0 and hasattr(gaussians, '_previous_last_feat'):
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

        if iteration >= opt.densify_from_iter - 100 and retain_grad and batch_size > 1:
            # grad calc
            viewspace_point_tensor_grad = torch.zeros((gaussians.offset_gradient_accum.shape[0], 3), device='cuda')
            visibility_filter = torch.zeros_like(gaussians.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
            for idx in range(0, len(viewspace_point_tensor_list)):
                anchor_visible_mask = voxel_visible_mask_list[idx].unsqueeze(dim=1).repeat([1, gaussians.n_offsets]).view(-1)
                combined_mask = torch.zeros_like(gaussians.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
                combined_mask[anchor_visible_mask] = offset_selection_mask_list[idx]
                temp_mask = combined_mask.clone()
                combined_mask[temp_mask] = visibility_filter_list[idx]

                viewspace_point_tensor_grad[combined_mask] += viewspace_point_tensor_list[idx].grad[visibility_filter_list[idx]]
                visibility_filter = torch.logical_or(visibility_filter, combined_mask)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            if gaussians.mode == 'hybrid' or gaussians.mode == 'static':
                total_point = gaussians._anchor.shape[0]
            else:
                total_point = 0
            if gaussians.mode == 'hybrid' or gaussians.mode == 'dynamic':
                dy_points = gaussians._anchor_dynamic.shape[0]
            else:
                dy_points = 0
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
            test_psnr = training_report(tb_writer, logger, iteration, loss_dict, loss, testing_iterations, scene, gaussians,
                                        hac_render, [pipe, background], stage, lr_dict=None, include_feature=begin_lang)

            # Compression
            # test_psnr > compres_thres or
            if test_psnr > best_test_psnr:
                if test_psnr > best_test_psnr:
                    best_test_psnr = test_psnr
                    best_iteration = iteration

                compr_path = os.path.join(dataset.model_path, "compression", f"iteration_{iteration}")
                compression_iterations.append(iteration)

                gaussians_to_compress = copy.deepcopy(gaussians)

                compr_results = run_compressions_hac(gaussians_to_compress, compr_path, OmegaConf.load(compression_config), gopid, qp=args.qp)

                # print
                stacked_point_feats = gaussians_to_compress._temporal_feat.detach()
                time_bound_min = torch.min(stacked_point_feats)
                time_bound_max = torch.max(stacked_point_feats)
                tqdm.write(f'time_min={time_bound_min}, time_max={time_bound_max}')

                success = True
                if stage != 'following':
                    bit_stream_path = os.path.join(compr_path, "bitstreams")
                    os.makedirs(bit_stream_path)
                    tqdm.write('Start encoding ...')
                    try:
                        patched_infos, log_info = gaussians_to_compress.conduct_encoding_new(pre_path_name=bit_stream_path)
                        tqdm.write(log_info)
                    except RuntimeError:
                        tqdm.write("[ERROR] entropy encoding failed, skip!")
                        success = False
                else:
                    bit_stream_path = os.path.join(compr_path, "bitstreams")
                    os.makedirs(bit_stream_path)
                    tqdm.write('Start encoding ...')
                    log_info = gaussians_to_compress.conduct_encoding_for_ntc(pre_path_name=bit_stream_path)
                    tqdm.write(log_info)

                if success:
                    if os.stat(os.path.join(compr_results['out_path'], '_temporal_feat.mp4')).st_size > (1 * 1024):
                        decompressed_gaussians = GaussianModel(
                            dataset.feat_dim,
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
                            print_log=False
                        )
                        decompressed_gaussians.time_line = gaussians.time_line
                        decompressed_gaussians.interval = gaussians.interval
                        decompressed_gaussians.keyframe_num = gaussians.keyframe_num
                        tqdm.write('Start decoding ...')
                        run_single_decompression_hac(compr_results['out_path'], compr_results['png_quant'], decompressed_gaussians, gaussians_to_compress)

                        if stage != 'following':
                            log_info = gaussians_to_compress.conduct_decoding_new(pre_path_name=bit_stream_path, patched_infos=patched_infos, gaussians=decompressed_gaussians)
                            tqdm.write(log_info)
                        else:
                            # decompressed_gaussians.conduct_decoding_from_files(os.path.join(args.checkpoint_path, 'bitstreams'))
                            decompressed_gaussians._anchor = gaussians_to_compress._anchor.detach()
                            decompressed_gaussians._anchor_dynamic = gaussians_to_compress._anchor_dynamic.detach()
                            decompressed_gaussians._anchor_feat = gaussians_to_compress._anchor_feat.detach()
                            decompressed_gaussians._anchor_feat_dynamic = gaussians_to_compress._anchor_feat_dynamic.detach()
                            decompressed_gaussians._offset = gaussians_to_compress._offset.detach()
                            decompressed_gaussians._offset_dynamic = gaussians_to_compress._offset_dynamic.detach()
                            decompressed_gaussians._scaling = gaussians_to_compress._scaling.detach()
                            decompressed_gaussians._scaling_dynamic = gaussians_to_compress._scaling_dynamic.detach()

                            log_info = decompressed_gaussians.conduct_decoding_for_ntc(pre_path_name=bit_stream_path)
                            tqdm.write(log_info)

                            decompressed_gaussians.ntc_mlp.load_state_dict(gaussians_to_compress.ntc_mlp.state_dict())

                        decom_test_psnr = training_report(tb_writer, logger, iteration, loss_dict, loss, compression_iterations, scene, decompressed_gaussians,
                                                          hac_render, [pipe, background], stage, compress=True, include_feature=False)

                        if decom_test_psnr > best_decom_test_psnr:
                            best_decom_test_psnr = decom_test_psnr
                            compr_best_path = os.path.join(dataset.model_path, "compression", f"best")
                            if os.path.exists(compr_best_path):
                                os.remove(compr_best_path)
                            os.symlink(compr_path, compr_best_path)
                    else:
                        tqdm.write(f"[ERROR] temporal feature video compression failed, skip testing!")

            # Densification
            if stage != "following" and iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter - 100:
                    if batch_size == 1 and gaussians.mode == 'hybrid':
                        gaussians.training_statis_hybrid(viewspace_point_tensor, opacity_list[0], visibility_filter, offset_selection_mask,
                                                         voxel_visible_mask_list[0], time_batch[0])
                    elif batch_size == 1 and gaussians.mode == 'static':
                        gaussians.training_statis(viewspace_point_tensor, opacity_list[0], visibility_filter, offset_selection_mask, voxel_visible_mask_list[0], time_batch[0])
                    elif batch_size == 1 and gaussians.mode == 'dynamic':
                        gaussians.training_statis_dynamic(viewspace_point_tensor, opacity_list[0], visibility_filter, offset_selection_mask, voxel_visible_mask_list[0],
                                                          time_batch[0])
                    elif batch_size > 1:
                        gaussians.training_statis_batch(viewspace_point_tensor_grad, opacity_list, visibility_filter, voxel_visible_mask_list, time_batch)
                    else:
                        raise NotImplementedError

                if iteration > opt.densify_from_iter and (iteration % opt.densification_interval == 0) and iteration not in range(opt.step_flag2, opt.step_flag2+1000):
                    if iteration % opt.static_densification_interval == 0 and iteration < opt.static_densify_until_iter:
                        if gaussians.mode == 'hybrid' or gaussians.mode == 'static':
                            gaussians.adjust_anchor(check_interval=opt.static_densification_interval, success_threshold=opt.success_threshold,
                                                    grad_threshold=opt.densify_grad_threshold_static, min_opacity=opt.opacity_threshold_static)
                    if iteration % opt.dynamic_densification_interval == 0:
                        if gaussians.mode == 'hybrid' or gaussians.mode == 'dynamic':
                            gaussians.adjust_anchor_dynamic(check_interval=opt.dynamic_densification_interval, success_threshold=opt.success_threshold,
                                                            grad_threshold=opt.densify_grad_threshold_dynamic, min_opacity=opt.opacity_threshold_dynamic)
                    # if hyper.sorting_enabled:
                    #     if gaussians.get_dynamic_xyz.shape[0] > 10000:
                    #         gaussians.sort_dynamic_into_grid(hyper, False)
                elif iteration == opt.densify_until_iter:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration == 4000 and stage != 'following':
            gaussians.update_anchor_bound()

        if iteration > final_iter + 1:
            break


def scene_refine_language(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                          gaussians: GaussianModel, scene, stage, train_iter, gopid, logger, tb_writer=None):
    first_iter = 0
    compression_config = os.path.dirname(args.configs) + "/" + args.compre_config
    assert os.path.isfile(compression_config), f"Compression config file not exists in {compression_config}!"

    # gaussians.create_language_learnable_features()
    gaussians.training_language_setup(opt)
    opt.step_flag1 = 0
    opt.step_flag2 = -1000

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    final_iter = train_iter
    testing_iterations.append(final_iter)

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    batch_size = opt.batch_size
    if opt.dataloader:
        viewpoint_stack_loader = DataLoader(train_cams, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=list, pin_memory=True)
        loader = iter(viewpoint_stack_loader)

    ema_loss_for_log = 0.0
    best_test_psnr = 0.0
    best_decom_test_psnr = 0.0
    best_gaussian = None

    compression_iterations = []
    compr_results = None
    progress_bar = tqdm(range(first_iter, final_iter + 1), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, final_iter + 1):
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
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            render_pkg = hac_render(viewpoint_cam, gaussians, pipe, background, stage, visible_mask=voxel_visible_mask, training=False)

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

        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            total_point = gaussians._anchor.shape[0]
            dy_points = gaussians._anchor_dynamic.shape[0]
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
            test_psnr = training_report(tb_writer, logger, iteration, loss_dict, loss, testing_iterations, scene, gaussians,
                                        hac_render, [pipe, background], stage, lr_dict=None, training=False)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.lang_optimizer.step()
                gaussians.lang_optimizer.zero_grad(set_to_none=True)

    compr_path = os.path.join(dataset.model_path, "compression", "best", "png_quant", "mlp_language.pth")
    torch.save(gaussians.mlp_language.state_dict(), compr_path)


def training_coarse(dataset, hyper, opt, pipe, args, expname, postfix, full, coarse):
    tb_writer, logger = prepare_output_and_logger(os.path.join(expname, f'coarse' if postfix is None else f'coarse_{postfix}'))
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        enable_filter=dataset.enable_filter,
        language=args.language
    )
    dataset.model_path = args.model_path
    if coarse and full:
        scene = Scene(dataset, gaussians, duration=[0, int(dataset.num_times)], timedordered=False)
    else:
        scene = Scene(dataset, gaussians, duration=[0, 1], timedordered=False)

    args.test_iterations = [i for i in range(1000, opt.coarse_iterations+1, 1000)]
    gaussians.time_line = 1
    scene_reconstruction_with_dynamic(dataset, opt, hyper, pipe, args.test_iterations, args, gaussians, scene, 'coarse', tb_writer, opt.coarse_iterations, logger)
    # scene_reconstruction(dataset, opt, hyper, pipe, args.test_iterations, args, gaussians, scene, 'coarse', tb_writer, opt.coarse_iterations, logger)


def training_refine(dataset, hyper, opt, pipe, compres_thres, checkpoint_path, args, expname, postfix, idx, gop=60):
    args.model_path = None
    tb_writer, logger = prepare_output_and_logger(os.path.join(expname, f'gop{idx}' if postfix is None else f'gop{idx}_{postfix}'))
    ''' 
    set stage to 'coarse' and mode to 'static | dynamic | hybrid' to test whether branch function is well
    set stage to 'refine' and mode to static to disable static-dynamic decomposition, where all anchor have temporal features
    set stage to 'refine' and mode to hybrid to enable static-dynamic decomposition, where only dynamic anchor have temporal features
    '''
    gaussians = GaussianModel(
        dataset.feat_dim,
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
        language=args.language
    )
    dataset.model_path = args.model_path
    scene = Scene(dataset, gaussians, duration=[idx*gop, (idx+1)*gop], load_memory=dataset.load_memory, skip=args.skip, timedordered=False, skip_init=True)

    assert os.path.exists(os.path.join(checkpoint_path, "coarse_sd.pth")), f"checkpoint not found in {checkpoint_path}!"
    checkpoint = torch.load(os.path.join(checkpoint_path, "coarse_sd.pth"), map_location='cuda')
    gaussians.create_from_coarse(checkpoint, scene.cameras_extent, scene.maxtime, dy_threshold=dataset.dy_threshold)
    # gaussians.setup_interpolators(opt.position_erp, opt.rotation_erp)
    torch.cuda.empty_cache()

    testing_iterations = [i for i in range(1000, args.refine_iterations+1, 1000)]

    scene_reconstruction_refine(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                                gaussians, scene, 'refine', opt.refine_iterations, gopid=idx, logger=logger, tb_writer=tb_writer)


def training_refine_language(dataset, hyper, opt, pipe, compres_thres, checkpoint_path, args, expname, postfix, idx, gop=60):
    args.model_path = None
    tb_writer, logger = prepare_output_and_logger(os.path.join(expname, f'gop{idx}' if postfix is None else f'gop{idx}_{postfix}'))

    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2,
        log2_hashmap_size_2D=args.log2_2D,
        mode='hybrid',
        decoded_version=True,
        enable_filter=dataset.enable_filter,
        language=True
    )
    dataset.model_path = args.model_path

    scene = Scene(dataset, gaussians, duration=[idx*gop, (idx+1)*gop], load_memory=dataset.load_memory, timedordered=False, skip_init=True)

    if idx == 0:
        assert os.path.exists(os.path.join(checkpoint_path, "png_quant", "model.pth")), f"checkpoint not found in {checkpoint_path}!"
        checkpoint = torch.load(os.path.join(checkpoint_path, "png_quant", "model.pth"), map_location='cuda')
        gaussians.mlp_grid.load_state_dict(checkpoint['mlp_grid'])
        gaussians.conduct_decoding_from_files(os.path.join(checkpoint_path, 'bitstreams'))

        gaussians._rotation = nn.Parameter(checkpoint['_rotation'].requires_grad_(False))
        gaussians._opacity = nn.Parameter(checkpoint['_opacity'].requires_grad_(False))
        gaussians._rotation_dynamic = nn.Parameter(checkpoint['_rotation_dynamic'].requires_grad_(False))
        gaussians._opacity_dynamic = nn.Parameter(checkpoint['_opacity_dynamic'].requires_grad_(False))

        gaussians.mlp_opacity.load_state_dict(checkpoint['mlp_opacity'])
        gaussians.mlp_cov.load_state_dict(checkpoint['mlp_cov'])
        gaussians.mlp_color.load_state_dict(checkpoint['mlp_color'])
        gaussians.mlp_language.load_state_dict(checkpoint['mlp_language'])

        gaussians._temporal_feat = checkpoint['_temporal_feat']
        gaussians.mlp_deform_xyz.load_state_dict(checkpoint['mlp_deform_xyz'])
        gaussians.mlp_deform_cov.load_state_dict(checkpoint['mlp_deform_cov'])
        gaussians.mlp_deform_color.load_state_dict(checkpoint['mlp_deform_color'])
        gaussians.mlp_deform_opacity.load_state_dict(checkpoint['mlp_deform_opacity'])

        stage = 'eval'
    else:
        print(f"No need to refine language, exited!")
        sys.exit(0)

    torch.cuda.empty_cache()
    testing_iterations = [i for i in range(1000, args.iterations + 1, 1000)]

    scene_refine_language(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                          gaussians, scene, stage, 3000, gopid=idx, logger=logger, tb_writer=tb_writer)


def training_following(dataset, hyper, opt, pipe, compres_thres, checkpoint_path, args, expname, postfix, idx, gop=60):
    args.model_path = None
    tb_writer, logger = prepare_output_and_logger(os.path.join(expname, f'gop{idx}' if postfix is None else f'gop{idx}_{postfix}'))
    gaussians = GaussianModel(
        dataset.feat_dim,
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
        language=args.language,
        stage='following'
    )
    dataset.model_path = args.model_path

    scene = Scene(dataset, gaussians, duration=[idx*gop, (idx+1)*gop], load_memory=dataset.load_memory, skip=args.skip, timedordered=False, skip_init=True)

    if 'pt' in postfix:
        temp_postfix = "default1"
        if args.lanuage:
            temp_postfix = "default1_lang"
    else:
        temp_postfix = postfix
        # temp_postfix = "default1"
    assert os.path.exists(os.path.join(expname, f"gop0_{temp_postfix}", "compression/best/png_quant", "model.pth")), f"checkpoint not found in {os.path.join(expname, f'gop0_{temp_postfix}', 'compression/best/png_quant', 'model.pth')}!"
    checkpoint = torch.load(os.path.join(expname, f"gop0_{temp_postfix}", "compression/best/png_quant", "model.pth"), map_location='cuda')

    gaussians.create_temporal_feat(checkpoint, scene.cameras_extent, scene.maxtime)
    gaussians.conduct_decoding_from_files(os.path.join(expname, f"gop0_{temp_postfix}", "compression/best", 'bitstreams'))
    if gaussians.mlp_language is not None:
        if os.path.exists(os.path.join(expname, f"gop0_{temp_postfix}", "compression/best/png_quant", "mlp_language.pth")):
            mlp_language = torch.load(os.path.join(expname, f"gop0_{temp_postfix}", "compression/best/png_quant", "mlp_language.pth"))
            gaussians.mlp_language.load_state_dict(mlp_language)
        else:
            gaussians.mlp_language.load_state_dict(checkpoint['mlp_language'])
    if idx > 1 and 'pt' not in postfix:
        checkpoint = torch.load(os.path.join(expname, f"gop{idx-1}_{postfix}", "compression/best/png_quant", "model.pth"), map_location='cuda')
        gaussians._anchor_feat = checkpoint['feat']
        gaussians._anchor_feat_dynamic = checkpoint['feat_dynamic']
        gaussians._offset = checkpoint['offsets']
        gaussians._offset_dynamic = checkpoint['offsets_dynamic']
        if gaussians.mlp_language is not None:
            gaussians.mlp_language.load_state_dict(checkpoint['mlp_language'])
    print("recovery gaussian attributes from bit streams")

    torch.cuda.empty_cache()

    testing_iterations = [i for i in range(1000, args.iterations+1, 1000)]

    scene_reconstruction_refine(dataset, opt, hyper, pipe, testing_iterations, compres_thres, args,
                                gaussians, scene, 'following', opt.refine_iterations, gopid, logger=logger, tb_writer=tb_writer)



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
    parser.add_argument("--coarse_full", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--language", action="store_true")
    parser.add_argument("--gopids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--segment", type=int, default=300, help="fixed")
    parser.add_argument("--total", action="store_true")
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--lmbda", type=float, default = 0.001)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.coarse:
        training_coarse(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args, args.expname, args.postfix, args.coarse_full, args.coarse)
    else:
        gop_nums = args.segment // args.gop
        for gopid in args.gopids:
            segment_id = gopid // gop_nums
            if segment_id == 0:
                if gopid == 0:
                    if not os.path.exists(os.path.join(args.expname, f"gop0_{args.postfix}")):
                        print(f"segment_id={segment_id}, gopid={gopid}, training_refine")
                        training_refine(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, args.checkpoint_path,
                                            args, args.expname, args.postfix, gopid, args.gop)

                    # if args.language and not os.path.exists(os.path.join(args.expname, f"gop0_{args.postfix}", "compression/best/png_quant", "mlp_language.pth")):
                    #     print(f"segment_id={segment_id}, gopid={gopid}, training_refine_language")
                    #     checkpoint_path = os.path.join(args.expname, f"gop0_{args.postfix}/compression/best")
                    #     training_refine_language(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, checkpoint_path,
                    #                              args, args.expname, args.postfix, gopid, args.gop)
                else:
                    print(f"segment_id={segment_id}, gopid={gopid}, training_following")
                    if not os.path.exists(os.path.join(args.expname, f"gop{gopid}_{args.postfix}")):
                        training_following(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.compres_thres, args.checkpoint_path,
                                           args, args.expname, args.postfix, gopid, args.gop)
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
    release_logger()
    print("\nTraining complete.")
