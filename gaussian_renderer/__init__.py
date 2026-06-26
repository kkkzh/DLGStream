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
import sys
import time
import torch
import torch.nn.functional as F
import math


from scene.fdsd_gaussian_model import GaussianModel
from scene.scaffold_gaussian_model import GaussianModel as SFGaussianModel

from gaussian_swift_rasterization import GaussianRasterizationSettings as SwiftSettings
from gaussian_swift_rasterization import GaussianRasterizer as SwiftRasterizer

from diff_gaussian_rasterization import GaussianRasterizationSettings as OSettings
from diff_gaussian_rasterization import GaussianRasterizer as ORaster

from gaussian_langsplat_rasterization import GaussianRasterizationSettings as SOlangsplatSettings
from gaussian_langsplat_rasterization import GaussianRasterizer as SOlangsplatRaster

from gaussian_dualopa_langsplat_rasterization import GaussianRasterizationSettings as DOlangsplatSettings
from gaussian_dualopa_langsplat_rasterization import GaussianRasterizer as DOlangsplatRaster

from diff_hac_rasterization import GaussianRasterizationSettings as HACSettings
from diff_hac_rasterization import GaussianRasterizer as HACRaster


def fps_render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, stage="refine", former_feats=None, evaluation=False, following=False, noise=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    world_view_transform = viewpoint_camera.world_view_transform.cuda()
    full_proj_transform = viewpoint_camera.full_proj_transform.cuda()
    camera_center = viewpoint_camera.camera_center.cuda()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = OSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = ORaster(raster_settings=raster_settings)

    means2D = screenspace_points

    dynamic_xyz, dynamic_rot, lip_feat = pc.get_xyz_rot_keyframe

    means3D = torch.cat([pc.get_xyz, dynamic_xyz], dim=0).contiguous()

    # scales = torch.cat([pc.get_scaling, pc.get_scaling_dynamic], dim=0).contiguous()
    if stage == 'following':
        dynamic_scale = pc.get_scale_keyframe
        scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori + dynamic_scale], dim=0).contiguous()
    else:
        scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori], dim=0).contiguous()

    rotations = torch.cat([pc.get_rotation_ori, dynamic_rot], dim=0).contiguous()
    rotations = pc.rotation_activation(rotations)

    ob_view = pc._xyz_dynamic.detach() - viewpoint_camera.camera_center.cuda()
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    local_view_feat = torch.cat([lip_feat, ob_view, ob_dist], dim=1)
    dynamic_opacity = pc.get_opacity_dynamic_ori + pc.get_opacity_mlp(local_view_feat)
    opacity = torch.cat([pc.get_opacity_ori, dynamic_opacity], dim=0)
    opacity = pc.opacity_activation(opacity)

    drgb = pc.get_color_mlp(local_view_feat)
    drgb = drgb.reshape([pc._xyz_dynamic.shape[0], 4, 3])
    shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs + drgb], dim=0).contiguous()

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    if viewpoint_camera.fisheye_mapper is not None:
        newiamge = F.grid_sample(rendered_image.unsqueeze(0), viewpoint_camera.fisheye_mapper.cuda(), mode='bicubic', padding_mode='zeros', align_corners=True)
        rendered_image = F.interpolate(newiamge, size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
        rendered_image = rendered_image.squeeze(0)

    return rendered_image


def stream_render_eval_v1(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, stage="refine", former_feats=None, evaluation=False, following=False, noise=None, pixel_weights=None, near=0.2, far=100.0):
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    torch.cuda.synchronize()
    t0 = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = OSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = ORaster(raster_settings=raster_settings)

    means2D = screenspace_points
    timestamp = viewpoint_camera.time
    dynamic_xyz, dynamic_rot, lip_feat = pc.get_xyz_rot_keyframe(timestamp)

    means3D = torch.cat([pc.get_xyz, dynamic_xyz], dim=0).contiguous()

    if stage == 'following':
        dynamic_scale = pc.get_scale_keyframe(timestamp)
        scales = torch.cat([pc.get_scaling_ori, dynamic_scale], dim=0).contiguous()
    else:
        scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori], dim=0).contiguous()

    scales = pc.scaling_activation(scales)

    rotations = torch.cat([pc.get_rotation_ori, dynamic_rot], dim=0).contiguous()
    rotations = pc.rotation_activation(rotations)

    ob_view = pc.get_dynamic_xyz.detach() - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    local_view_feat = torch.cat([lip_feat, ob_view, ob_dist], dim=1)
    dynamic_opacity = pc.get_opacity_dynamic_ori + pc.get_opacity_mlp(local_view_feat)
    # dynamic_opacity = pc.get_opacity_dynamic_ori
    opacity = torch.cat([pc.get_opacity_ori, dynamic_opacity], dim=0)
    opacity = pc.opacity_activation(opacity)
    opacity = opacity[:, :1]

    drgb = pc.get_color_mlp(local_view_feat)
    drgb = drgb.reshape([pc.get_dynamic_xyz.shape[0], 4, 3])
    shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs + drgb], dim=0).contiguous()
    # shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs], dim=0).contiguous()

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    torch.cuda.synchronize()
    t1 = time.time()
    duration = t1 - t0

    return rendered_image, duration


def stream_render_eval(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, stage="refine", former_feats=None, evaluation=False, following=False, noise=None, pixel_weights=None, near=0.2, far=100.0):
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    torch.cuda.synchronize()
    t0 = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = OSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = ORaster(raster_settings=raster_settings)

    means2D = screenspace_points
    timestamp = viewpoint_camera.time
    dynamic_xyz, dynamic_rot, lip_feat = pc.get_xyz_rot_keyframe(timestamp)

    means3D = torch.cat([pc.get_xyz, dynamic_xyz], dim=0).contiguous()

    if stage == 'following':
        dynamic_scale = pc.get_scale_keyframe(timestamp)
        scales = torch.cat([pc.get_scaling_ori, dynamic_scale], dim=0).contiguous()
    else:
        scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori], dim=0).contiguous()

    scales = pc.scaling_activation(scales)

    rotations = torch.cat([pc.get_rotation_ori, dynamic_rot], dim=0).contiguous()
    rotations = pc.rotation_activation(rotations)

    ob_view = pc.get_dynamic_xyz.detach() - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    local_view_feat = torch.cat([lip_feat[:, 4:12], ob_view, ob_dist], dim=1)
    dynamic_opacity = pc.get_opacity_dynamic_ori + pc.get_opacity_mlp(local_view_feat)
    # dynamic_opacity = pc.get_opacity_dynamic_ori
    opacity = torch.cat([pc.get_opacity_ori, dynamic_opacity], dim=0)
    opacity = pc.opacity_activation(opacity)
    opacity = opacity[:, :1]

    drgb = pc.get_color_mlp(local_view_feat)
    drgb = drgb.reshape([pc.get_dynamic_xyz.shape[0], 4, 3])
    shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs + drgb], dim=0).contiguous()
    # shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs], dim=0).contiguous()

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    torch.cuda.synchronize()
    t1 = time.time()
    duration = t1 - t0

    return rendered_image, duration


def stream_lang_render_eval(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, stage="refine", former_feats=None, evaluation=False, following=False, noise=None, pixel_weights=None, near=0.2, far=100.0):
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    torch.cuda.synchronize()
    t0 = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = DOlangsplatSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=True,
    )

    rasterizer = DOlangsplatRaster(raster_settings=raster_settings)

    means2D = screenspace_points
    timestamp = viewpoint_camera.time
    dynamic_xyz, dynamic_rot, lip_feat = pc.get_xyz_rot_keyframe(timestamp)

    means3D = torch.cat([pc.get_xyz, dynamic_xyz], dim=0).contiguous()

    if stage == 'following':
        dynamic_scale = pc.get_scale_keyframe(timestamp)
        scales = torch.cat([pc.get_scaling_ori, dynamic_scale], dim=0).contiguous()
    else:
        scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori], dim=0).contiguous()

    scales = pc.scaling_activation(scales)

    rotations = torch.cat([pc.get_rotation_ori, dynamic_rot], dim=0).contiguous()
    rotations = pc.rotation_activation(rotations)

    ob_view = pc.get_dynamic_xyz.detach() - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    local_view_feat = torch.cat([lip_feat[:, 4:12], ob_view, ob_dist], dim=1)
    dynamic_opacity = pc.get_opacity_dynamic_ori + pc.get_opacity_mlp(local_view_feat)

    opacity = torch.cat([pc.get_opacity_ori, dynamic_opacity], dim=0)
    opacity = pc.opacity_activation(opacity)

    drgb = pc.get_color_mlp(local_view_feat)
    drgb = drgb.reshape([pc.get_dynamic_xyz.shape[0], 4, 3])
    shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs + drgb], dim=0).contiguous()

    dy_lang_feature = pc.get_lang_mlp(lip_feat[:, 8:])
    language_feature = torch.cat([pc.get_language_feature, pc.get_language_feature_dynamic + dy_lang_feature], dim=0).contiguous()

    rendered_image, rendered_language_feature, radii, max_weight_t = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        language_feature_precomp=language_feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    torch.cuda.synchronize()
    t1 = time.time()
    duration = t1 - t0

    return rendered_image, duration


def stream_render_fisd(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, stage="refine", former_feats=None, evaluation=False, following=False, noise=None, pixel_weights=None, include_feature=False, return_opacity=False):
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if pc._opacity.shape[1] == 1:
        LangsplatSettings = SOlangsplatSettings
        LangsplatRaster = SOlangsplatRaster
    elif pc._opacity.shape[1] == 2:
        LangsplatSettings = DOlangsplatSettings
        LangsplatRaster = DOlangsplatRaster
    else:
        raise NotImplementedError

    raster_settings = LangsplatSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=include_feature,
    )

    rasterizer = LangsplatRaster(raster_settings=raster_settings)

    means2D = screenspace_points
    means3D = pc.get_xyz

    if stage == "coarse":
        scales = pc.get_scaling
        rotations = pc.get_rotation
        opacity = pc.get_opacity
        shs = pc.get_static_shs
        feats = None
        delta_t = None

        if include_feature:
            language_feature = pc.get_language_feature
        else:
            language_feature = torch.zeros((means3D.shape[0], pc.lang_feat_dim), dtype=opacity.dtype, device=opacity.device)

    elif stage != "coarse" and pc.get_dynamic_xyz.shape[0] == 0:
        # means3D = pc.get_xyz
        scales = pc.get_scaling
        rotations = pc.get_rotation
        opacity = pc.get_opacity
        shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs], dim=0).contiguous()

        delta_t = None
        feats = None

        if include_feature:
            language_feature = pc.get_language_feature
        else:
            language_feature = torch.zeros((means3D.shape[0], pc.lang_feat_dim), dtype=opacity.dtype, device=opacity.device)

    elif stage != "coarse":
        timestamp = viewpoint_camera.time

        feats, t_idx, delta_t = pc.get_time_features(timestamp, noise=0)
        static_xyz = pc.get_xyz
        feat = feats[0]
        feat_next = feats[1]

        dy_off = pc.get_deform_mlp(feat[:, :12])
        dy_loc = pc.get_dynamic_xyz + dy_off[:, :3]
        dy_off_next = pc.get_deform_mlp(feat_next[:, :12])
        dy_loc_next = pc.get_dynamic_xyz + dy_off_next[:, :3]
        means3D = torch.cat([static_xyz, pc.interpolator(dy_loc, dy_off[:, 3:], dy_loc_next, dy_off_next[:, 3:], delta_t)], dim=0).contiguous()

        if stage == 'following':
            dcov = pc.get_cov_mlp(feat[:, :12])
            scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori + dcov[:, :3]], dim=0).contiguous()

            dy_rot = pc.get_rotation_dynamic_ori + dcov[:, 3:]
            dy_rot_next = pc.get_rotation_dynamic_ori + pc.get_cov_mlp(feat_next[:, :12])[:, 3:]
        else:
            scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori], dim=0).contiguous()

            dy_rot = pc.get_rotation_dynamic_ori + pc.get_cov_mlp(feat[:, :12])[:, 3:]
            dy_rot_next = pc.get_rotation_dynamic_ori + pc.get_cov_mlp(feat_next[:, :12])[:, 3:]

        scales = pc.scaling_activation(scales)

        rotations = torch.cat([pc.get_rotation_ori, pc.rot_interpolator(dy_rot, None, dy_rot_next, None, delta_t)], dim=0).contiguous()
        rotations = pc.rotation_activation(rotations)

        ob_view = pc.get_dynamic_xyz.detach() - viewpoint_camera.camera_center.cuda()
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist
        lip_feat = pc.linear_interpolator(feat[:, 4:12], None, feat_next[:, 4:12], None, delta_t)
        local_view_feat = torch.cat([lip_feat, ob_view, ob_dist], dim=1)

        dynamic_opacity = pc.get_opacity_dynamic_ori + pc.get_opacity_mlp(local_view_feat)
        # dynamic_opacity = pc.get_opacity_dynamic_ori
        opacity = torch.cat([pc.get_opacity_ori, dynamic_opacity], dim=0)
        opacity = pc.opacity_activation(opacity)

        drgb = pc.get_color_mlp(local_view_feat)
        drgb = drgb.reshape([pc.get_dynamic_xyz.shape[0], (pc.max_sh_degree + 1) ** 2, 3])

        shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs + drgb], dim=0).contiguous()
        # shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs], dim=0).contiguous()

        if include_feature:
            lip_feat2 = pc.linear_interpolator(feat[:, 8:], None, feat_next[:, 8:], None, delta_t)
            dy_lang_feature =  pc.get_lang_mlp(lip_feat2)
            language_feature = torch.cat([pc.get_language_feature, pc.get_language_feature_dynamic + dy_lang_feature], dim=0).contiguous()
            # language_feature / (language_feature.norm(dim=-1, keepdim=True) + 1e-9)
        else:
            language_feature = torch.zeros((means3D.shape[0], pc.lang_feat_dim), dtype=opacity.dtype, device=opacity.device)

    else:
        raise NotImplementedError

    if return_opacity:
        return torch.cat([pc.get_opacity_ori, dynamic_opacity], dim=0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_language_feature, radii, max_weight_t = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        language_feature_precomp=language_feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # if viewpoint_camera.fisheye_mapper is not None:
    #     newiamge = F.grid_sample(rendered_image.unsqueeze(0), viewpoint_camera.fisheye_mapper.cuda(), mode='bicubic', padding_mode='zeros', align_corners=True)
    #     rendered_image = F.interpolate(newiamge, size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    #     rendered_image = rendered_image.squeeze(0)

    output = {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "max_weight_t": max_weight_t,
        "opacity": opacity,
        "feats": feats,
        "delta_t": delta_t,
        "language_feature": rendered_language_feature
    }

    return output


def stream_render_lang(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, stage="refine", former_feats=None, evaluation=False, following=False, noise=None, pixel_weights=None, include_feature=False):
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if pc._opacity.shape[1] == 1:
        LangsplatSettings = SOlangsplatSettings
        LangsplatRaster = SOlangsplatRaster
    elif pc._opacity.shape[1] == 2:
        LangsplatSettings = DOlangsplatSettings
        LangsplatRaster = DOlangsplatRaster
    else:
        raise NotImplementedError
    # LangsplatSettings = DOlangsplatSettings
    # LangsplatRaster = DOlangsplatRaster

    raster_settings = LangsplatSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=include_feature,
    )

    rasterizer = LangsplatRaster(raster_settings=raster_settings)

    means2D = screenspace_points
    means3D = pc.get_xyz

    timestamp = viewpoint_camera.time

    feats, t_idx, delta_t = pc.get_time_features(timestamp, noise=0)
    static_xyz = pc.get_xyz
    feat = feats[0]
    feat_next = feats[1]

    dy_off = pc.get_deform_mlp(feat[:, :12])
    dy_loc = pc.get_dynamic_xyz + dy_off[:, :3]
    dy_off_next = pc.get_deform_mlp(feat_next[:, :12])
    dy_loc_next = pc.get_dynamic_xyz + dy_off_next[:, :3]
    means3D = torch.cat([static_xyz, pc.interpolator(dy_loc, dy_off[:, 3:], dy_loc_next, dy_off_next[:, 3:], delta_t)], dim=0).contiguous()

    if stage == 'following':
        dcov = pc.get_cov_mlp(feat[:, :12])
        scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori + dcov[:, :3]], dim=0).contiguous()

        dy_rot = pc.get_rotation_dynamic_ori + dcov[:, 3:]
        dy_rot_next = pc.get_rotation_dynamic_ori + pc.get_cov_mlp(feat_next[:, :12])[:, 3:]
    else:
        scales = torch.cat([pc.get_scaling_ori, pc.get_scaling_dynamic_ori], dim=0).contiguous()

        dy_rot = pc.get_rotation_dynamic_ori + pc.get_cov_mlp(feat[:, :12])[:, 3:]
        dy_rot_next = pc.get_rotation_dynamic_ori + pc.get_cov_mlp(feat_next[:, :12])[:, 3:]

    scales = pc.scaling_activation(scales)

    rotations = torch.cat([pc.get_rotation_ori, pc.rot_interpolator(dy_rot, None, dy_rot_next, None, delta_t)], dim=0).contiguous()
    rotations = pc.rotation_activation(rotations)

    ob_view = pc.get_dynamic_xyz.detach() - viewpoint_camera.camera_center.cuda()
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist
    lip_feat = pc.linear_interpolator(feat[:, 4:12], None, feat_next[:, 4:12], None, delta_t)
    local_view_feat = torch.cat([lip_feat, ob_view, ob_dist], dim=1)

    # dynamic_opacity = torch.cat([pc.get_opacity_dynamic_ori[:, 0:1], pc._lang_opacity_dynamic], dim=1)
    # dynamic_opacity = dynamic_opacity + pc.get_opacity_mlp(local_view_feat)
    # static_opacity  = torch.cat([pc.get_opacity_ori[:, 0:1], pc._lang_opacity], dim=1)
    # opacity = torch.cat([static_opacity, dynamic_opacity], dim=0)
    dynamic_opacity = pc.get_opacity_dynamic_ori + pc.get_opacity_mlp(local_view_feat)
    opacity = torch.cat([pc.get_opacity_ori, dynamic_opacity], dim=0)
    opacity = pc.opacity_activation(opacity)

    drgb = pc.get_color_mlp(local_view_feat)
    drgb = drgb.reshape([pc.get_dynamic_xyz.shape[0], (pc.max_sh_degree + 1) ** 2, 3])

    shs = torch.cat([pc.get_static_shs, pc.get_dynamic_shs + drgb], dim=0).contiguous()

    lang_temp_feat, lang_temp_feat_next = pc.get_time_lang_features(timestamp)
    lang_feat = torch.cat([feat[:, 8:12], lang_temp_feat], dim=1)
    lang_feat_next = torch.cat([feat_next[:, 8:12], lang_temp_feat_next], dim=1)
    lip_feat2 = pc.linear_interpolator(lang_feat, None, lang_feat_next, None, delta_t)
    dy_lang_feature =  pc.get_lang_mlp(lip_feat2)
    language_feature = torch.cat([pc.get_language_feature, pc.get_language_feature_dynamic + dy_lang_feature], dim=0).contiguous()

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_language_feature, radii, max_weight_t = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        language_feature_precomp=language_feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    output = {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "max_weight_t": max_weight_t,
        "opacity": opacity,
        "feats": [lang_temp_feat, lang_temp_feat_next],
        "delta_t": delta_t,
        "language_feature": rendered_language_feature
    }

    return output


def dynamic_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, stage="dynamic", bwd_dynamic=True):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    world_view_transform = viewpoint_camera.world_view_transform.cuda()
    full_proj_transform = viewpoint_camera.full_proj_transform.cuda()
    camera_center = viewpoint_camera.camera_center.cuda()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = SwiftSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        bwd_depth=False,
        bwd_dynamic=bwd_dynamic,
        debug=pipe.debug
    )

    rasterizer = SwiftRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    means3D = pc.get_xyz

    scales = pc.get_scaling
    rotations = pc.get_rotation
    opacity = pc.get_colored_opacity
    shs = pc.get_static_shs
    dynamics = pc.get_dynamic

    rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        dynamics=dynamics,
        cov3D_precomp=None)

    if viewpoint_camera.fisheye_mapper is not None:
        newiamge = F.grid_sample(dynamics_map.unsqueeze(0), viewpoint_camera.fisheye_mapper.cuda(), mode='bicubic', padding_mode='zeros', align_corners=True)
        dynamics_map = F.interpolate(newiamge, size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
        dynamics_map = dynamics_map.squeeze(0)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "dynamic_map": dynamics_map,
            "max_weight_t": max_weight_t
    }


from einops import repeat
from utils.encodings import STE_binary, STE_multistep
def generate_neural_gaussians(viewpoint_camera, pc: SFGaussianModel, stage, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    torch.cuda.synchronize()
    t0 = time.time()

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]

    feat = pc.get_anchor_features[visible_mask]
    # grid_offsets = pc._offset[visible_mask]
    grid_offsets = pc.get_offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    if stage == 'coarse' or stage == 'refine':
        bit_per_param = None
        bit_per_feat_param = None
        bit_per_scaling_param = None
        bit_per_offsets_param = None
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2
        if is_training:
            if step > pc.step_flag1 and step <= pc.step_flag2:
                # quantization
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

            if step == pc.step_flag2:
                pc.update_anchor_bound()

            if step > pc.step_flag2:
                feat_context = pc.calc_interp_feat(anchor)
                feat_context = pc.get_grid_mlp(feat_context)
                mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                    torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3 * pc.n_offsets, 3 * pc.n_offsets, 1, 1, 1], dim=-1)

                Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
                Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
                Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

                choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
                feat_chosen = feat[choose_idx]
                grid_scaling_chosen = grid_scaling[choose_idx]
                grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3 * pc.n_offsets)
                mean = mean[choose_idx]
                scale = scale[choose_idx]
                mean_scaling = mean_scaling[choose_idx]
                scale_scaling = scale_scaling[choose_idx]
                mean_offsets = mean_offsets[choose_idx]
                scale_offsets = scale_offsets[choose_idx]
                Q_feat = Q_feat[choose_idx]
                Q_scaling = Q_scaling[choose_idx]
                Q_offsets = Q_offsets[choose_idx]
                bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc.get_anchor_features.mean())
                bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
                bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc.get_offset.mean())
                bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
                bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
                bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
                bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                                (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

        elif not pc.decoded_version and (step > pc.step_flag2 or step == -1):  # training时test
            torch.cuda.synchronize()
            t1 = time.time()
            feat_context = pc.calc_interp_feat(anchor)
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(pc.get_grid_mlp(feat_context), split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3 * pc.n_offsets, 3 * pc.n_offsets, 1, 1, 1], dim=-1)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N_visible_anchor, 1]
            feat = (STE_multistep.apply(feat, Q_feat, pc.get_anchor_features.mean())).detach()
            grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
            grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc.get_offset.mean())).detach()
            torch.cuda.synchronize()
            time_sub = time.time() - t1

    elif stage == 'following':
        d_feat, d_offsets, d_anchor = pc.get_ntc(anchor)
        if d_anchor is not None:
            anchor = anchor.detach() + d_anchor
        feat = feat.detach() + d_feat
        grid_offsets = grid_offsets.detach() + d_offsets

    elif stage == 'eval':
        pass
    else:
        raise NotImplementedError

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    flag = True
    feats = None
    if stage != 'coarse' and flag:
        timestamp = viewpoint_camera.time
        visible_static_num = visible_mask[:pc.get_static_anchor_num].sum()
        visible_mask_dynamic = visible_mask[pc.get_static_anchor_num:]  # dynamic anchor mask

        feats, t_idx, delta_t = pc.get_time_features(timestamp)
        curr_feat, next_feat = feats[0][visible_mask_dynamic], feats[1][visible_mask_dynamic]
        lip_feat = pc.linear_interpolator(curr_feat, None, next_feat, None, delta_t)
        # curr_feat, next_feat, nnext_feat, nnnext_feat = feats[0][visible_mask_dynamic], feats[1][visible_mask_dynamic], feats[2][visible_mask_dynamic], feats[3][visible_mask_dynamic]
        # lip_feat = pc.linear_interpolator(curr_feat, next_feat, nnext_feat, nnnext_feat, delta_t)

        anchor_dynamic = anchor[visible_static_num:]
        anchor_dynamic_norm = (anchor_dynamic - pc.x_bound_min) / (pc.x_bound_max - pc.x_bound_min)
        anchor_dynamic_norm = torch.cat([lip_feat, anchor_dynamic_norm], dim=1)
        danchor = pc.mlp_deform_xyz(anchor_dynamic_norm)
        anchor = torch.cat([anchor[:visible_static_num], anchor_dynamic + danchor], dim=0).contiguous()

        dfeat = pc.mlp_deform_cov(lip_feat)
        feat = torch.cat([feat[:visible_static_num], feat[visible_static_num:] + dfeat], dim=0).contiguous()

        doffsets = pc.mlp_deform_opacity(lip_feat).reshape([anchor_dynamic.shape[0], pc.n_offsets, 3])
        grid_offsets = torch.cat([grid_offsets[:visible_static_num], grid_offsets[visible_static_num:] + doffsets], dim=0).contiguous()

        dscaling = pc.mlp_deform_color(lip_feat)
        grid_scaling = torch.cat([grid_scaling[:visible_static_num], grid_scaling[visible_static_num:] + dscaling], dim=0).contiguous()

    torch.cuda.synchronize()
    time_sub += time.time() - t0

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]
    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]

    if pc.mlp_language is not None:
        neural_opacity = neural_opacity.reshape([-1, 2])  # [N_visible_anchor*K, 2]
        mask = (neural_opacity[:, :1] > 0.0)
    else:
        neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
        mask = (neural_opacity > 0.0)

    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    # scale_rot = pc.get_cov_mlp(feat)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # language features
    lang_feats = None
    if pc.mlp_language is not None:
        lang_feats = pc.mlp_language(cat_local_view)
        lang_feats = lang_feats.reshape([anchor.shape[0] * pc.n_offsets, 9])  # [N_visible_anchor*K, 9]
        lang_feats = lang_feats[mask]

    grid_scaling = pc.scaling_activation(grid_scaling)

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    # scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    # offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    # rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]
    rot = scale_rot[:, 3:7]

    if stage != 'coarse' and not flag:
        timestamp = viewpoint_camera.time
        if pc.mode == 'hybrid':
            visible_static_num = visible_mask[:pc.get_static_anchor_num].sum()
            visible_mask_dynamic = visible_mask[pc.get_static_anchor_num:]  # dynamic anchor mask

            mask_dynamic = mask[visible_static_num * pc.n_offsets:]
            mask_static_num = mask[:visible_static_num * pc.n_offsets].sum()

            feats, t_idx, delta_t = pc.get_time_features(timestamp)
            curr_feat, next_feat = feats[0][visible_mask_dynamic], feats[1][visible_mask_dynamic]
            lip_feat = pc.linear_interpolator(curr_feat, None, next_feat, None, delta_t)

            anchor_dynamic = anchor[visible_static_num:]
            anchor_dynamic_norm = (anchor_dynamic - pc.x_bound_min) / (pc.x_bound_max - pc.x_bound_min)
            anchor_dynamic_norm = torch.cat([lip_feat, anchor_dynamic_norm], dim=1)
            dxyz = pc.mlp_deform_xyz(anchor_dynamic_norm).reshape([anchor_dynamic.shape[0] * pc.n_offsets, 3])[mask_dynamic]
            offsets = torch.cat([offsets[:mask_static_num], offsets[mask_static_num:] + dxyz], dim=0)
            offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]

            dcov = pc.mlp_deform_cov(lip_feat).reshape([anchor_dynamic.shape[0] * pc.n_offsets, 7])[mask_dynamic]
            scale_rot = torch.cat([scale_rot[:mask_static_num, :3], scale_rot[mask_static_num:, :3] + dcov[:, :3]], dim=0)
            scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
            rot = torch.cat([rot[:mask_static_num], rot[mask_static_num:] + dcov[:, 3:]], dim=0)

            local_view_feat = torch.cat([lip_feat, ob_view[visible_static_num:], ob_dist[visible_static_num:]], dim=1)
            dopacity= pc.mlp_deform_opacity(local_view_feat).reshape([-1, 1])[mask_dynamic]
            opacity = torch.cat([opacity[:mask_static_num], opacity[mask_static_num:] + dopacity], dim=0)
            dcolor = pc.mlp_deform_color(local_view_feat).reshape([anchor_dynamic.shape[0] * pc.n_offsets, 3])[mask_dynamic]
            color = torch.cat([color[:mask_static_num], color[mask_static_num:] + dcolor], dim=0)
        else:
            feats, t_idx, delta_t = pc.get_time_features(timestamp)
            curr_feat, next_feat = feats[0][visible_mask], feats[1][visible_mask]
            lip_feat = pc.linear_interpolator(curr_feat, None, next_feat, None, delta_t)
            anchor_norm = (anchor - pc.x_bound_min) / (pc.x_bound_max - pc.x_bound_min)
            anchor_norm = torch.cat([lip_feat, anchor_norm], dim=1)
            dxyz = pc.mlp_deform_xyz(anchor_norm).reshape([anchor.shape[0] * pc.n_offsets, 3])[mask]
            offsets = (offsets + dxyz) * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]

            cov = pc.mlp_deform_cov(lip_feat).reshape([anchor.shape[0] * pc.n_offsets, 7])[mask]
            scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3] + cov[:, :3])
            rot = rot + cov[:, 3:]

            local_view_feat = torch.cat([lip_feat, ob_view, ob_dist], dim=1)
            opacity = opacity + pc.mlp_deform_opacity(local_view_feat).reshape([-1, 1])[mask]
            color = color + pc.mlp_deform_color(local_view_feat).reshape([anchor.shape[0] * pc.n_offsets, 3])[mask]
    else:
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]

    rot = pc.rotation_activation(rot)
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

    if stage == 'coarse' and step == -1:
        dynamic = pc.get_dynamic[visible_mask]
        dynamic = dynamic.view([-1, 1])
        dynamic = dynamic[mask]

    if is_training and (stage == 'coarse' or stage == 'refine'):
        return xyz, color, opacity, scaling, rot, neural_opacity, lang_feats, mask, feats, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    elif is_training and stage == 'following':
        # d_xyz = 0
        # return xyz, color, opacity, scaling, rot, d_xyz, anchor_loss, feats
        return xyz, color, opacity, scaling, rot, lang_feats, feats
    elif stage == 'coarse' and step == -1:
        return xyz, color, opacity, scaling, rot, dynamic
    else:
        return xyz, color, opacity, scaling, rot, lang_feats, time_sub


def hac_render(viewpoint_camera, pc: SFGaussianModel, pipe, bg_color: torch.Tensor, stage, scaling_modifier=1.0, visible_mask=None, retain_grad=False, training=None, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if training is None:
        is_training = pc.get_color_mlp.training
    else:
        is_training = training

    if is_training and (stage == 'coarse' or stage == 'refine'):
        xyz, color, opacity, scaling, rot, neural_opacity, lang_feats, mask, feats, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians(
            viewpoint_camera, pc, stage, visible_mask, is_training=is_training, step=step)
    elif is_training and stage == 'following':
        xyz, color, opacity, scaling, rot, lang_feats, feats = generate_neural_gaussians(viewpoint_camera, pc, stage, visible_mask, is_training=is_training, step=step)
    else:
        xyz, color, opacity, scaling, rot, lang_feats, time_sub = generate_neural_gaussians(viewpoint_camera, pc, stage, visible_mask, is_training=is_training, step=step)

    torch.cuda.synchronize()
    t0 = time.time()
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if lang_feats is None:
        LangsplatRaster = HACRaster
        raster_settings = HACSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )
    else:
        LangsplatSettings = DOlangsplatSettings
        LangsplatRaster = DOlangsplatRaster
        # LangsplatSettings = SOlangsplatSettings
        # LangsplatRaster = SOlangsplatRaster
        raster_settings = LangsplatSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            include_feature=True,
        )

    rasterizer = LangsplatRaster(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_language_feature = None
    if lang_feats is None:
        rendered_image, radii = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            opacities=opacity,
            scales=scaling,
            rotations=rot,
            cov3D_precomp=None)
    else:
        rendered_image, rendered_language_feature, radii, max_weight_t = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        language_feature_precomp=lang_feats,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training and stage != 'following':
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity[:,:1],
                "scaling": scaling,
                "language_feature": rendered_language_feature,
                "feats": feats,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                }
    elif is_training and stage == "following":
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "scaling": scaling,
                "language_feature": rendered_language_feature,
                "feats": feats
                }
    else:
        torch.cuda.synchronize()
        time_sub += time.time() - t0
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "language_feature": rendered_language_feature,
                "time_sub": time_sub,
                }


def hac_dynamic_render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, stage="coarse", bwd_dynamic=True, visible_mask=None):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = SwiftSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        bwd_depth=False,
        bwd_dynamic=bwd_dynamic,
        debug=pipe.debug
    )
    rasterizer = SwiftRasterizer(raster_settings=raster_settings)

    xyz, color, opacity, scaling, rot, dynamics = generate_neural_gaussians(viewpoint_camera, pc, stage, visible_mask, is_training=False, step=-1)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points

    rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
        means3D=xyz,
        means2D=means2D,
        shs=None,
        colors_precomp=color,
        opacities=opacity[:, 0:1],
        scales=scaling,
        rotations=rot,
        dynamics=dynamics,
        cov3D_precomp=None)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "dynamic_map": dynamics_map,
            "max_weight_t": max_weight_t
    }


def prefilter_voxel(viewpoint_camera, pc: SFGaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    if not pc.enable_filter:
        visible_mask = torch.ones(pc.get_anchor_num, dtype=torch.bool, device=pc.get_anchor.device)

        return visible_mask

    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = HACSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = HACRaster(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc.get_scaling  # requires_grad = True
    rotations = pc.get_rotation  # requires_grad = True

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )

    return radii_pure > 0
