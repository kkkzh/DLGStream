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

import os
import time
import math
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric, build_rotation)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, Q_anchor, \
    encoder_anchor, decoder_anchor, \
    encoder, decoder, \
    encoder_gaussian, decoder_gaussian, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder_cuda, decoder_cuda, \
    encoder_gaussian_chunk, decoder_gaussian_chunk


bit2MB_scale = 8 * 1024 * 1024

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = False):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)

        return latent

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]

        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        
        return out_i
    

class NTC_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=1,
            resolutions_list=[514],
            log2_hashmap_size=15,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.output_dim = self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim 

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)

        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]

        out_i = torch.cat([out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        
        # return out_i, 0.0

        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        p = torch.sigmoid(out_xyz)

        return out_i * p, p

def nearest_interpolate(y1, y2, t):
    if t <= 0.5:
        return y1
    else:
        return y2

def linear_interp_uniiterval(y1, y2, t):
    return (y1 * (1 - t) + y2 * t)

def cubic_interpolate(y1, y2, y3, y4, t):
    # Catmull-Rom 三次样条插值
    t2 = t * t
    t3 = t2 * t

    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2

    return w0 * y1 + w1 * y2 + w2 * y3 + w3 * y4


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 mode='none',
                 enable_filter=True,
                 stage='none',
                 language=False,
                 print_log=True
                 ):
        super().__init__()
        if print_log:
            print('hash_params:', use_2D, n_features_per_level,
                  log2_hashmap_size, resolutions_list,
                  log2_hashmap_size_2D, resolutions_list_2D,
                  ste_binary, ste_multistep, add_noise)

        self.mode = mode
        self.enable_filter = enable_filter

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.time_line = 60
        self.interval = 10
        self.expand_time = 1
        self.keyframe_num = self.time_line // self.interval + self.expand_time
        self._temporal_feat = torch.empty(0)
        self.temporal_feat_dim = 16

        def linear_interpolation(y1, n1, y2, n2, delta_t):
            return linear_interp_uniiterval(y1, y2, delta_t).squeeze(1)
            # return cubic_interpolate(y1, n1, y2, n2, delta_t).squeeze(1)
            # return nearest_interpolate(y1, y2, delta_t).squeeze(1)
        self.linear_interpolator = linear_interpolation

        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        if self.mode == 'hybrid':
            self.offset_mode = 0

            self._anchor_dynamic = torch.empty(0)
            self._offset_dynamic = torch.empty(0)
            self._anchor_feat_dynamic = torch.empty(0)

            self._scaling_dynamic = torch.empty(0)
            self._rotation_dynamic = torch.empty(0)
            self._opacity_dynamic = torch.empty(0)

            self.opacity_dynamic_accum = torch.empty(0)
            self.anchor_dynamic_demon = torch.empty(0)
            self.offset_dynamic_gradient_accum = torch.empty(0)
            self.offset_dynamic_denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        if print_log:
            encoding_params_num = 0
            for n, p in self.encoding_xyz.named_parameters():
                encoding_params_num += p.numel()
            encoding_MB = encoding_params_num / 8 / 1024 / 1024
            if not ste_binary: encoding_MB *= 32
            print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB. keyframe_interval={self.interval}')

        mlp_input_feat_dim = feat_dim

        if language:
            self.mlp_opacity = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 2*n_offsets),
                nn.Tanh()
            ).cuda()
        else:
            self.mlp_opacity = nn.Sequential(
                    nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, n_offsets),
                    nn.Tanh()
                ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+1+1+1),
        ).cuda()

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()

        if language:
            self.mlp_language = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 9*self.n_offsets),
            ).cuda()
        else:
            self.mlp_language = None

        hidden = 64
        self.mlp_deform_xyz = nn.Sequential(
            nn.Linear(16+3, hidden, bias=False),
            nn.ReLU(True),
            nn.Linear(hidden, 3, bias=False),
            # nn.Linear(hidden, 3*self.n_offsets, bias=False),
        ).cuda()

        self.mlp_deform_cov = nn.Sequential(
            nn.Linear(16, int(hidden*2), bias=False),
            nn.ReLU(True),
            nn.Linear(int(hidden*2), feat_dim, bias=False),
            # nn.Linear(int(hidden * 2), 7*self.n_offsets, bias=False),
        ).cuda()

        self.mlp_deform_opacity = nn.Sequential(
            nn.Linear(16, hidden, bias=False),
            # nn.Linear(16+3+1, hidden, bias=False),
            nn.ReLU(True),
            nn.Linear(hidden, 3*self.n_offsets, bias=False),
            # nn.Linear(hidden, self.n_offsets, bias=False),
            # nn.Tanh()
        ).cuda()

        self.mlp_deform_color = nn.Sequential(
            nn.Linear(16, hidden, bias=False),
            # nn.Linear(3 + 1 + 16, hidden, bias=False),
            nn.ReLU(True),
            nn.Linear(hidden, 6, bias=False),
            # nn.Linear(hidden, 3*self.n_offsets, bias=False),
        ).cuda()

        if stage == 'following':
            self.ntc = NTC_encoding(
                n_features=4,
                resolutions_list_2D=[512, 1024, 2048, 4096],
                log2_hashmap_size_2D=15,
                ste_binary=self.ste_binary,
                ste_multistep=self.ste_multistep,
                add_noise=self.add_noise,
                Q=self.Q,
            ).cuda()

            output_dim = 3 * self.n_offsets + self.feat_dim   # d_offsets, d_feat, d_anchor
            self.ntc_mlp = nn.Sequential(
                nn.Linear(self.ntc.output_dim, self.feat_dim * 2),
                nn.ReLU(True),
                nn.Linear(self.feat_dim * 2, output_dim),
            ).cuda()
        else:
            self.ntc = None
            self.ntc_mlp = None
        self.step_flag1 = None
        self.step_flag2 = None

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_ntc_2D_params(self):
        params = []
        
        params.append(self.ntc.encoding_xy.params)
        params.append(self.ntc.encoding_xz.params)
        params.append(self.ntc.encoding_yz.params)
        
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params
    
    def get_ntc_3D_params(self):
        params = []
        
        params.append(self.ntc.encoding_xyz.params)
        
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params
    
    def get_ntc_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.ntc_mlp.named_parameters():
            mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()

        self.mlp_deform_xyz.eval()
        self.mlp_deform_cov.eval()
        self.mlp_deform_opacity.eval()
        self.mlp_deform_color.eval()

        if self.mlp_language is not None:
            self.mlp_language.eval()

        if self.ntc is not None:
            self.ntc.eval()
            self.ntc_mlp.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()

        self.mlp_deform_xyz.train()
        self.mlp_deform_cov.train()
        self.mlp_deform_opacity.train()
        self.mlp_deform_color.train()

        if self.mlp_language is not None:
            self.mlp_language.train()

        if self.ntc is not None:
            self.ntc.train()
            self.ntc_mlp.train()

    @property
    def get_scaling(self):
        # if self.decoded_version:
        #     if self.mode == 'static':
        #         return self._scaling
        #     elif self.mode == 'dynamic':
        #         return self._scaling_dynamic
        #     elif self.mode == 'hybrid':
        #         return torch.cat([self._scaling, self._scaling_dynamic], dim=0)
        #     return self._scaling
        # if self.mode == 'static':
        #     return self.scaling_activation(self._scaling)
        # elif self.mode == 'dynamic':
        #     return self.scaling_activation(self._scaling_dynamic)
        # elif self.mode == 'hybrid':
        #     return self.scaling_activation(torch.cat([self._scaling, self._scaling_dynamic], dim=0))
        # return self.scaling_activation(self._scaling)
        if self.mode == 'static':
            return self._scaling
        elif self.mode == 'dynamic':
            return self._scaling_dynamic
        elif self.mode == 'hybrid':
            return torch.cat([self._scaling, self._scaling_dynamic], dim=0)
        return self._scaling

    @property
    def get_scaling_activated(self):
        if self.mode == 'static':
            return self.scaling_activation(self._scaling)
        elif self.mode == 'dynamic':
            return self.scaling_activation(self._scaling_dynamic)
        elif self.mode == 'hybrid':
            return self.scaling_activation(torch.cat([self._scaling, self._scaling_dynamic], dim=0))
        return self.scaling_activation(self._scaling)

    @property
    def get_offset(self):
        if self.mode == 'static':
            return self._offset
        elif self.mode == 'dynamic':
            return self._offset_dynamic
        elif self.mode == 'hybrid':
            return torch.cat([self._offset, self._offset_dynamic], dim=0)
        return self._offset

    @property
    def get_anchor_features(self):
        if self.mode == 'static':
            return self._anchor_feat
        elif self.mode == 'dynamic':
            return self._anchor_feat_dynamic
        elif self.mode == 'hybrid':
            return torch.cat([self._anchor_feat, self._anchor_feat_dynamic], dim=0)
        return self._anchor_feat

    @property
    def get_rotation(self):
        if self.mode == 'static':
            return self.rotation_activation(self._rotation)
        elif self.mode == 'dynamic':
            return self.rotation_activation(self._rotation_dynamic)
        elif self.mode == 'hybrid':
            return self.rotation_activation(torch.cat([self._rotation, self._rotation_dynamic], dim=0))
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            if self.mode == 'static':
                return self._anchor
            elif self.mode == 'dynamic':
                return self._anchor_dynamic
            elif self.mode == 'hybrid':
                return torch.cat([self._anchor, self._anchor_dynamic], dim=0)
            return self._anchor
        if self.mode == 'static':
            anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
            return anchor
        elif self.mode == 'dynamic':
            anchor, quantized_v = Quantize_anchor.apply(self._anchor_dynamic, self.x_bound_min, self.x_bound_max)
            return anchor
        elif self.mode == 'hybrid':
            anchor = torch.cat([self._anchor, self._anchor_dynamic], dim=0)
            anchor, quantized_v = Quantize_anchor.apply(anchor, self.x_bound_min, self.x_bound_max)
            return anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor

    @property
    def get_quantized_v(self):
        if self.mode == 'static':
            anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
            return quantized_v
        elif self.mode == 'dynamic':
            anchor, quantized_v = Quantize_anchor.apply(self._anchor_dynamic, self.x_bound_min, self.x_bound_max)
            return quantized_v
        elif self.mode == 'hybrid':
            anchor = torch.cat([self._anchor, self._anchor_dynamic], dim=0)
            anchor, quantized_v = Quantize_anchor.apply(anchor, self.x_bound_min, self.x_bound_max)
            return quantized_v
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return quantized_v

    @property
    def get_anchor_static(self):
        if self.decoded_version:
            return self._anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor

    @property
    def get_anchor_dynamic(self):
        if self.decoded_version:
            return self._anchor_dynamic
        anchor, quantized_v = Quantize_anchor.apply(self._anchor_dynamic, self.x_bound_min, self.x_bound_max)
        return anchor

    @property
    def get_anchor_num(self):
        if self.mode == 'static':
            return self._anchor.shape[0]
        elif self.mode == 'dynamic':
            return self._anchor_dynamic.shape[0]
        elif self.mode == 'hybrid':
            return self._anchor.shape[0] + self._anchor_dynamic.shape[0]
        return self._anchor.shape[0]

    @property
    def get_static_anchor_num(self):
        return self._anchor.shape[0]

    def get_time_features(self, timestamp):
        t = int(timestamp * self.time_line)
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval

        t_idx = int(t_idx)
        feat = self._temporal_feat[:, t_idx, :]
        feat_next = self._temporal_feat[:, t_idx + 1, :]
        # feat_next2 = self._temporal_feat[:, t_idx + 2, :]
        # feat_next3 = self._temporal_feat[:, t_idx + 3, :]
        return [feat, feat_next], t_idx, delta_t

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @torch.no_grad()
    def update_anchor_bound(self, resize=False):
        # print("self.x_bound_min", self.x_bound_min)
        # print("self.x_bound_max", self.x_bound_max)
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        if resize:
            for c in range(x_bound_min.shape[-1]):
                x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
            for c in range(x_bound_max.shape[-1]):
                x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')
        print("self.x_bound_min", self.x_bound_min)
        print("self.x_bound_max", self.x_bound_max)

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features
    
    def get_ntc(self, x):
        # x: [N, 3]

        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]

        mask = (x >= 0) & (x <= 1)
        mask = mask.all(dim=1)

        features_input, p = self.ntc(x[mask])
        features = self.ntc_mlp(features_input)
        # d_anchor_mask, d_feat_mask, d_offsets_mask = torch.split(features, [3, self.feat_dim, 3*self.n_offsets], dim=1)
        d_feat_mask, d_offsets_mask = torch.split(features, [self.feat_dim, 3 * self.n_offsets], dim=1)

        d_feat = torch.full((x.shape[0], self.feat_dim), 0.0, dtype=torch.float32, device="cuda")
        d_offsets = torch.full((x.shape[0], 3*self.n_offsets), 0.0, dtype=torch.float32, device="cuda")
        # d_anchor = torch.full((x.shape[0], 3), 0.0, dtype=torch.float32, device="cuda")

        # d_anchor[mask] = d_anchor_mask
        d_anchor = None
        d_feat[mask] = d_feat_mask
        d_offsets[mask] = d_offsets_mask

        d_offsets = d_offsets.reshape(-1, self.n_offsets, 3)

        return d_feat, d_offsets, d_anchor

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        print("points : ", points.shape)

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        # self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

        # self.time_line = time_line
        # self._temporal_feat = nn.Parameter(torch.zeros((self._anchor.shape[0], self.keyframe_num, self.temporal_feat_dim), device="cuda").requires_grad_(False))

    def create_from_coarse(self, checkpoint, spatial_lr_scale: float, time_line: int, dy_threshold=5.0):
        self.spatial_lr_scale = spatial_lr_scale
        anchor = checkpoint['_anchor'].float().cuda()
        print("Number of anchors at initialisation : ", anchor.shape[0])
        offset = checkpoint['_offset'].float().cuda()
        anchor_feat = checkpoint['_anchor_feat'].float().cuda()
        scaling = checkpoint['_scaling'].float().cuda()
        rotation = checkpoint['_rotation'].float().cuda()
        opacity = checkpoint['_opacity'].float().cuda()

        self.time_line = time_line
        self.keyframe_num = self.time_line // self.interval + self.expand_time

        self.x_bound_max = checkpoint['x_bound_max']
        self.x_bound_min = checkpoint['x_bound_min']
        print("self.x_bound_min", self.x_bound_min)
        print("self.x_bound_max", self.x_bound_max)

        _dynamic = checkpoint['_dynamic'].float().cuda()
        _dynamic = _dynamic.sum(dim=1)
        dynamic_mask = (_dynamic > dy_threshold).squeeze(1)

        if self.mode == 'hybrid':
            self._anchor = nn.Parameter(anchor[~dynamic_mask].requires_grad_(True))
            self._offset = nn.Parameter(offset[~dynamic_mask].requires_grad_(True))
            self._anchor_feat = nn.Parameter(anchor_feat[~dynamic_mask].requires_grad_(True))
            self._scaling = nn.Parameter(scaling[~dynamic_mask].requires_grad_(True))
            self._rotation = nn.Parameter(rotation[~dynamic_mask].requires_grad_(False))
            self._opacity = nn.Parameter(opacity[~dynamic_mask].requires_grad_(False))

            self._anchor_dynamic = nn.Parameter(anchor[dynamic_mask].requires_grad_(True))
            self._offset_dynamic = nn.Parameter(offset[dynamic_mask].requires_grad_(True))
            self._anchor_feat_dynamic = nn.Parameter(anchor_feat[dynamic_mask].requires_grad_(True))
            self._scaling_dynamic = nn.Parameter(scaling[dynamic_mask].requires_grad_(True))
            self._rotation_dynamic = nn.Parameter(rotation[dynamic_mask].requires_grad_(False))
            self._opacity_dynamic = nn.Parameter(opacity[dynamic_mask].requires_grad_(False))
            self._temporal_feat = nn.Parameter(torch.zeros((self._anchor_dynamic.shape[0], self.keyframe_num, self.temporal_feat_dim), device="cuda").requires_grad_(True))
        elif self.mode == 'dynamic':
            self._anchor_dynamic = nn.Parameter(anchor.requires_grad_(True))
            self._offset_dynamic = nn.Parameter(offset.requires_grad_(True))
            self._anchor_feat_dynamic = nn.Parameter(anchor_feat.requires_grad_(True))
            self._scaling_dynamic = nn.Parameter(scaling.requires_grad_(True))
            self._rotation_dynamic = nn.Parameter(rotation.requires_grad_(False))
            self._opacity_dynamic = nn.Parameter(opacity.requires_grad_(False))
            self._temporal_feat = nn.Parameter(torch.zeros((self._anchor_dynamic.shape[0], self.keyframe_num, self.temporal_feat_dim), device="cuda").requires_grad_(True))
        else:
            self._anchor = nn.Parameter(anchor.requires_grad_(True))
            self._offset = nn.Parameter(offset.requires_grad_(True))
            self._anchor_feat = nn.Parameter(anchor_feat.requires_grad_(True))
            self._scaling = nn.Parameter(scaling.requires_grad_(True))
            self._rotation = nn.Parameter(rotation.requires_grad_(False))
            self._opacity = nn.Parameter(opacity.requires_grad_(False))
            self._temporal_feat = nn.Parameter(torch.zeros((self._anchor.shape[0], self.keyframe_num, self.temporal_feat_dim), device="cuda").requires_grad_(True))

        self.mlp_cov.load_state_dict(checkpoint['mlp_cov'])
        self.mlp_opacity.load_state_dict(checkpoint['mlp_opacity'])
        self.mlp_color.load_state_dict(checkpoint['mlp_color'])
        self.mlp_grid.load_state_dict(checkpoint['mlp_grid'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        if self.mlp_language is not None:
            self.mlp_language.load_state_dict(checkpoint['mlp_language'])

    def create_from_ckpt(self, checkpoint, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        anchor = checkpoint['_anchor'].float().cuda()
        offset = checkpoint['_offset'].float().cuda()
        anchor_feat = checkpoint['_anchor_feat'].float().cuda()
        scaling = checkpoint['_scaling'].float().cuda()
        rotation = checkpoint['_rotation'].float().cuda()
        opacity = checkpoint['_opacity'].float().cuda()

        self.time_line = time_line
        self.keyframe_num = self.time_line // self.interval + self.expand_time

        self.x_bound_max = checkpoint['x_bound_max']
        self.x_bound_min = checkpoint['x_bound_min']
        print("self.x_bound_min", self.x_bound_min)
        print("self.x_bound_max", self.x_bound_max)

        anchor_dynamic = checkpoint['_anchor_dynamic'].float().cuda()
        offset_dynamic = checkpoint['_offset_dynamic'].float().cuda()
        anchor_feat_dynamic = checkpoint['_anchor_feat_dynamic'].float().cuda()
        scaling_dynamic = checkpoint['_scaling_dynamic'].float().cuda()
        rotation_dynamic = checkpoint['_rotation_dynamic'].float().cuda()
        opacity_dynamic = checkpoint['_opacity_dynamic'].float().cuda()

        print(f"Number of static anchors={anchor.shape[0]}, dynamic anchors={anchor_dynamic.shape[0]}")

        self._anchor = nn.Parameter(anchor.requires_grad_(False))
        self._offset = nn.Parameter(offset.requires_grad_(False))
        self._anchor_feat = nn.Parameter(anchor_feat.requires_grad_(False))
        self._scaling = nn.Parameter(scaling.requires_grad_(False))
        self._rotation = nn.Parameter(rotation.requires_grad_(False))
        self._opacity = nn.Parameter(opacity.requires_grad_(False))

        self._anchor_dynamic = nn.Parameter(anchor_dynamic.requires_grad_(False))
        self._offset_dynamic = nn.Parameter(offset_dynamic.requires_grad_(False))
        self._anchor_feat_dynamic = nn.Parameter(anchor_feat_dynamic.requires_grad_(False))
        self._scaling_dynamic = nn.Parameter(scaling_dynamic.requires_grad_(False))
        self._rotation_dynamic = nn.Parameter(rotation_dynamic.requires_grad_(False))
        self._opacity_dynamic = nn.Parameter(opacity_dynamic.requires_grad_(False))
        self._temporal_feat = nn.Parameter(
            torch.zeros((self._anchor_dynamic.shape[0], self.keyframe_num, self.temporal_feat_dim), device="cuda").requires_grad_(True))

        self.mlp_cov.load_state_dict(checkpoint['mlp_cov'])
        self.mlp_opacity.load_state_dict(checkpoint['mlp_opacity'])
        self.mlp_color.load_state_dict(checkpoint['mlp_color'])
        self.mlp_grid.load_state_dict(checkpoint['mlp_grid'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])

    def create_temporal_feat(self, checkpoint, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        self.time_line = time_line
        self.keyframe_num = self.time_line // self.interval + self.expand_time
        self._temporal_feat = nn.Parameter(
            torch.zeros((checkpoint['_anchor_dynamic'].shape[0], self.keyframe_num, self.temporal_feat_dim), device="cuda").requires_grad_(True))

        self.mlp_cov.load_state_dict(checkpoint['mlp_cov'])
        self.mlp_opacity.load_state_dict(checkpoint['mlp_opacity'])
        self.mlp_color.load_state_dict(checkpoint['mlp_color'])
        self.mlp_grid.load_state_dict(checkpoint['mlp_grid'])

        self.x_bound_max = checkpoint['x_bound_max']
        self.x_bound_min = checkpoint['x_bound_min']
        print("self.x_bound_min", self.x_bound_min)
        print("self.x_bound_max", self.x_bound_max)

        self._rotation = nn.Parameter(checkpoint['_rotation'].requires_grad_(False))
        self._opacity = nn.Parameter(checkpoint['_opacity'].requires_grad_(False))
        self._rotation_dynamic = nn.Parameter(checkpoint['_rotation_dynamic'].requires_grad_(False))
        self._opacity_dynamic = nn.Parameter(checkpoint['_opacity_dynamic'].requires_grad_(False))

    def set_steps(self, flag_1, flag_2):
        self.step_flag1 = flag_1
        self.step_flag2 = flag_2

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self._anchor.shape[0], self.time_line, 1), device="cuda")
        self.anchor_demon = torch.zeros((self._anchor.shape[0], self.time_line, 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")

        if self.mode == 'hybrid' or self.mode == 'dynamic':
            self.opacity_dynamic_accum = torch.zeros((self._anchor_dynamic.shape[0], self.time_line, 1), device="cuda")
            self.anchor_dynamic_demon = torch.zeros((self._anchor_dynamic.shape[0], self.time_line, 1), device="cuda")

            self.offset_dynamic_gradient_accum = torch.zeros((self._anchor_dynamic.shape[0] * self.n_offsets, 1), device="cuda")
            self.offset_dynamic_denom = torch.zeros((self._anchor_dynamic.shape[0] * self.n_offsets, 1), device="cuda")

        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

            {'params': self.mlp_opacity.parameters(), 'lr': training_args.hac_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.hac_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.hac_color_lr_init, "name": "mlp_color"},

            {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
            {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},

            {'params': self.mlp_deform_xyz.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform_xyz"},
            {'params': self.mlp_deform_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_deform_cov"},
            {'params': self.mlp_deform_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_deform_color"},
            {'params': self.mlp_deform_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_deform_opacity"},
        ]

        if self.mode == 'static':
            l.extend([
                {'params': [self._temporal_feat], 'lr': training_args.temporal_feature_lr_init, "name": f"temporal_feat"},
            ])

        if self.mode == 'hybrid' or self.mode == 'dynamic':
            l.extend([
                {'params': [self._anchor_dynamic], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "dynamic_anchor"},
                {'params': [self._offset_dynamic], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "dynamic_offset"},
                {'params': [self._anchor_feat_dynamic], 'lr': training_args.feature_lr, "name": "dynamic_anchor_feat"},
                {'params': [self._opacity_dynamic], 'lr': training_args.opacity_lr, "name": "dynamic_opacity"},
                {'params': [self._scaling_dynamic], 'lr': training_args.scaling_lr, "name": "dynamic_scaling"},
                {'params': [self._rotation_dynamic], 'lr': training_args.rotation_lr, "name": "dynamic_rotation"},
                {'params': [self._temporal_feat], 'lr': training_args.temporal_feature_lr_init, "name": f"dynamic_feat"},
            ])

        if self.mlp_language is not None:
            l.extend([
                {'params': self.mlp_language.parameters(), 'lr': training_args.mlp_lang_lr_init, "name": f"mlp_lang"}
            ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        self.hac_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.hac_opacity_lr_init,
                                                    lr_final=training_args.hac_opacity_lr_final,
                                                    lr_delay_mult=training_args.hac_opacity_lr_delay_mult,
                                                    max_steps=training_args.hac_opacity_lr_max_steps)

        self.hac_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.hac_cov_lr_init,
                                                    lr_final=training_args.hac_cov_lr_final,
                                                    lr_delay_mult=training_args.hac_cov_lr_delay_mult,
                                                    max_steps=training_args.hac_cov_lr_max_steps)

        self.hac_color_scheduler_args = get_expon_lr_func(lr_init=training_args.hac_color_lr_init,
                                                    lr_final=training_args.hac_color_lr_final,
                                                    lr_delay_mult=training_args.hac_color_lr_delay_mult,
                                                    max_steps=training_args.hac_color_lr_max_steps)

        self.temporal_feature_scheduler_args = get_expon_lr_func(lr_init=training_args.temporal_feature_lr_init,
                                                           lr_final=training_args.temporal_feature_lr_final,
                                                           lr_delay_mult=training_args.temporal_feature_lr_delay_mult,
                                                           max_steps=training_args.temporal_feature_lr_steps)

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                           lr_final=training_args.mlp_deform_lr_final,
                                                           lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                           max_steps=training_args.mlp_deform_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["offset", "dynamic_offset"]:
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] in ["anchor", "dynamic_anchor"]:
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.hac_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.hac_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.hac_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr

            # temporal field
            if param_group["name"] == "temporal_feat":
                lr = self.temporal_feature_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_xyz":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr

    def training_following_setup(self, training_args):
        l = [
            {'params': self.mlp_deform_xyz.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform_xyz"},
            {'params': self.mlp_deform_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_deform_cov"},
            {'params': self.mlp_deform_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_deform_color"},
            {'params': self.mlp_deform_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_deform_opacity"},

            {'params': [self._temporal_feat], 'lr': training_args.temporal_feature_lr_init, "name": f"dynamic_feat"},

            {'params': self.ntc.parameters(), 'lr': 0.01, "name": "ntc"},
            {'params': self.ntc_mlp.parameters(), 'lr': 0.001, "name": "ntc_mlp"},
        ]

        # if self.mlp_language is not None:
        #     l.extend([
        #         {'params': self.mlp_language.parameters(), 'lr': training_args.mlp_lang_lr_init, "name": f"mlp_lang"}
        #     ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.hac_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.hac_opacity_lr_init,
                                                    lr_final=training_args.hac_opacity_lr_final,
                                                    lr_delay_mult=training_args.hac_opacity_lr_delay_mult,
                                                    max_steps=training_args.hac_opacity_lr_max_steps)

        self.hac_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.hac_cov_lr_init,
                                                    lr_final=training_args.hac_cov_lr_final,
                                                    lr_delay_mult=training_args.hac_cov_lr_delay_mult,
                                                    max_steps=training_args.hac_cov_lr_max_steps)

        self.hac_color_scheduler_args = get_expon_lr_func(lr_init=training_args.hac_color_lr_init,
                                                    lr_final=training_args.hac_color_lr_final,
                                                    lr_delay_mult=training_args.hac_color_lr_delay_mult,
                                                    max_steps=training_args.hac_color_lr_max_steps)

        self.temporal_feature_scheduler_args = get_expon_lr_func(lr_init=training_args.temporal_feature_lr_init,
                                                           lr_final=training_args.temporal_feature_lr_final,
                                                           lr_delay_mult=training_args.temporal_feature_lr_delay_mult,
                                                           max_steps=training_args.temporal_feature_lr_steps)

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                           lr_final=training_args.mlp_deform_lr_final,
                                                           lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                           max_steps=training_args.mlp_deform_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)

        self.ntc_scheduler_args = get_expon_lr_func(lr_init=0.01,
                                                    lr_final=0.0001, # 1e-5
                                                    lr_delay_mult=0.01,
                                                    max_steps=20_000)

    def update_learning_rate_following(self, iteration):
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "mlp_opacity":
            #     lr = self.hac_opacity_scheduler_args(iteration)
            #     param_group['lr'] = lr
            # if param_group["name"] == "mlp_cov":
            #     lr = self.hac_cov_scheduler_args(iteration)
            #     param_group['lr'] = lr
            # if param_group["name"] == "mlp_color":
            #     lr = self.hac_color_scheduler_args(iteration)
            #     param_group['lr'] = lr

            # if "ntc" in param_group["name"]:
            #     lr = self.ntc_scheduler_args(iteration)
            #     param_group['lr'] = lr

            # temporal field
            if param_group["name"] == "temporal_feat":
                lr = self.temporal_feature_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_xyz":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr

    # static-dynamic decouple
    def create_dynamic(self):
        self._dynamic = nn.Parameter(torch.zeros((self._anchor.shape[0], self.n_offsets, 1), device="cuda").requires_grad_(True))

    @property
    def get_dynamic(self):
        return self._dynamic

    def training_dynamic_setup(self, training_args):
        dynamics_l = [
            {'params': [self._dynamic], 'lr': 0.05, "name": "dynamic"},
        ]
        self.dy_optimizer = torch.optim.Adam(dynamics_l, lr=0.0, eps=1e-15)
        self.dy_scheduler_args = get_expon_lr_func(lr_init=0.05, lr_final=0.005,
                                                   lr_delay_mult=training_args.position_lr_delay_mult, max_steps=30000)

    def update_learning_dy_rate(self, iteration):
        for param_group in self.dy_optimizer.param_groups:
            if param_group["name"] == "dynamic":
                lr = self.dy_scheduler_args(iteration)
                param_group['lr'] = lr

    # static-dynamic decouple

    def training_language_setup(self, training_args):
        l = [
            # {'params': [self._lang_temp_feats], 'lr': training_args.temporal_feature_lr_init, "name": f"dynamic_feat"},
            {'params': self.mlp_language.parameters(), 'lr': training_args.mlp_lang_lr_init, "name": "mlp_language"},
        ]

        self.lang_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


    def save(self, path, canonical=False):
        state_dict = self.state_dict()
        if canonical:
            new_state_dict = {k: v for k, v in state_dict.items() if 'mlp' not in k and 'temporal' not in k}
        else:
            new_state_dict = {k: v for k, v in state_dict.items() if 'mlp' not in k}
        new_state_dict['mlp_opacity'] = self.mlp_opacity.state_dict()
        new_state_dict['mlp_cov'] = self.mlp_cov.state_dict()
        new_state_dict['mlp_color'] = self.mlp_color.state_dict()
        new_state_dict['encoding_xyz'] = self.encoding_xyz.state_dict()
        new_state_dict['mlp_grid'] = self.mlp_grid.state_dict()
        new_state_dict['x_bound_min'] = self.x_bound_min
        new_state_dict['x_bound_max'] = self.x_bound_max
        if not canonical:
            new_state_dict['mlp_deform_xyz'] = self.mlp_deform_xyz.state_dict()
            new_state_dict['mlp_deform_cov'] = self.mlp_deform_cov.state_dict()
            new_state_dict['mlp_deform_color'] = self.mlp_deform_color.state_dict()
            new_state_dict['mlp_deform_opacity'] = self.mlp_deform_opacity.state_dict()
        if self.mlp_language is not None:
            new_state_dict['mlp_language'] = self.mlp_language.state_dict()
        if self.ntc is not None:
            new_state_dict['ntc_mlp'] = self.ntc_mlp.state_dict()
            d_feat, d_offsets, _ = self.get_ntc(self.get_anchor)
            new_state_dict['d_feat'] = d_feat[:self.get_static_anchor_num]
            new_state_dict['d_offsets'] = d_offsets[:self.get_static_anchor_num]
            new_state_dict['d_feat_dynamic'] = d_feat[self.get_static_anchor_num:]
            new_state_dict['d_offsets_dynamic'] = d_offsets[self.get_static_anchor_num:]

            new_state_dict['feat'] = self._anchor_feat + d_feat[:self.get_static_anchor_num]
            new_state_dict['offsets'] = self._offset + d_offsets[:self.get_static_anchor_num]
            new_state_dict['feat_dynamic'] = self._anchor_feat_dynamic + d_feat[self.get_static_anchor_num:]
            new_state_dict['offsets_dynamic'] = self._offset_dynamic + d_offsets[self.get_static_anchor_num:]
        torch.save(new_state_dict, path)

    def save_mlps(self,path):
        state_dict = {
            # 'mlp_opacity': self.mlp_opacity.state_dict(),
            # 'mlp_cov': self.mlp_cov.state_dict(),
            # 'mlp_color': self.mlp_color.state_dict(),
            # 'mlp_grid': self.mlp_grid.state_dict(),
            'x_bound_min': self.x_bound_min,
            'x_bound_max': self.x_bound_max,
            'mlp_deform_xyz': self.mlp_deform_xyz.state_dict(),
            'mlp_deform_cov': self.mlp_deform_cov.state_dict(),
            'mlp_deform_color': self.mlp_deform_color.state_dict(),
            'mlp_deform_opacity': self.mlp_deform_opacity.state_dict(),
        }
        if self.mlp_language is not None:
            state_dict['mlp_language'] = self.mlp_language.state_dict()
        if self.ntc is not None:
            state_dict['ntc_mlp'] = self.ntc_mlp.state_dict()
        if self.ntc is None:
            state_dict['mlp_opacity'] = self.mlp_opacity.state_dict()
            state_dict['mlp_cov'] = self.mlp_cov.state_dict()
            state_dict['mlp_color'] = self.mlp_color.state_dict()
            state_dict['mlp_grid'] = self.mlp_grid.state_dict()
        torch.save(state_dict, path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            if not group["name"] in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis_batch(self, viewspace_point_tensor, opacity_list, update_filter, voxel_visible_mask_list, time_batch):
        for idx, n_opacity in enumerate(opacity_list):
            temp_opacity = n_opacity.clone().view(-1).detach()
            temp_opacity[temp_opacity < 0] = 0
            temp_opacity = temp_opacity.view([-1, self.n_offsets])
            temp_opacity = temp_opacity.sum(dim=1, keepdim=True)
            local_visible_mask = voxel_visible_mask_list[idx]

            t = int(time_batch[idx] * self.time_line)
            self.opacity_accum[local_visible_mask, t, :] = temp_opacity
            self.anchor_demon[local_visible_mask, t, :] += 1

        grad_norm = torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[update_filter] += grad_norm
        self.offset_denom[update_filter] += 1

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask, timestamp):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        temp_opacity = temp_opacity.sum(dim=1, keepdim=True)

        t = int(timestamp * self.time_line)
        self.opacity_accum[anchor_visible_mask, t, :] = temp_opacity
        self.anchor_demon[anchor_visible_mask, t, :] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def training_statis_dynamic(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask, timestamp):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        temp_opacity = temp_opacity.sum(dim=1, keepdim=True)

        t = int(timestamp * self.time_line)
        self.opacity_dynamic_accum[anchor_visible_mask, t, :] = temp_opacity
        self.anchor_dynamic_demon[anchor_visible_mask, t, :] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_dynamic_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_dynamic_gradient_accum[combined_mask] += grad_norm
        self.offset_dynamic_denom[combined_mask] += 1

    def training_statis_hybrid(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask, timestamp):
        static_num = self.get_static_anchor_num

        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        temp_opacity = temp_opacity.sum(dim=1, keepdim=True)

        assert anchor_visible_mask.sum() == temp_opacity.shape[0]

        anchor_visible_mask_static = anchor_visible_mask[:static_num]
        anchor_visible_mask_static_sum = anchor_visible_mask_static.sum()
        temp_opacity_static = temp_opacity[:anchor_visible_mask_static_sum]
        t = int(timestamp * self.time_line)
        # self.opacity_accum[anchor_visible_mask_static, :] = temp_opacity_static
        # self.anchor_demon[anchor_visible_mask_static, :] += 1
        self.opacity_accum[anchor_visible_mask_static, t, :] = temp_opacity_static
        self.anchor_demon[anchor_visible_mask_static, t, :] += 1

        anchor_visible_mask_dynamic = anchor_visible_mask[static_num:]
        temp_opacity_dynamic = temp_opacity[anchor_visible_mask_static_sum:]
        self.opacity_dynamic_accum[anchor_visible_mask_dynamic, t, :] = temp_opacity_dynamic
        self.anchor_dynamic_demon[anchor_visible_mask_dynamic, t, :] += 1

        visible_static_num = anchor_visible_mask[:self.get_static_anchor_num].sum()
        mask_static_num = offset_selection_mask[:visible_static_num * self.n_offsets].sum()
        # mask_dynamic_num = offset_selection_mask[visible_static_num * self.n_offsets:].sum()

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(torch.cat([self.offset_gradient_accum, self.offset_dynamic_gradient_accum], dim=0), dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        filter_static_num = update_filter[:mask_static_num].sum()
        grad_norm_static = grad_norm[:filter_static_num]
        combined_mask_static = combined_mask[:self.offset_gradient_accum.shape[0]]
        self.offset_gradient_accum[combined_mask_static] += grad_norm_static
        self.offset_denom[combined_mask_static] += 1

        grad_norm_dynamic = grad_norm[filter_static_num:]
        combined_mask_dynamic = combined_mask[self.offset_gradient_accum.shape[0]:]
        self.offset_dynamic_gradient_accum[combined_mask_dynamic] += grad_norm_dynamic
        self.offset_dynamic_denom[combined_mask_dynamic] += 1

    def training_statis_static(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask, 0, :] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask, 0, :] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            if not group["name"].startswith("dynamic"):
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    if group['name'] in ["scaling", "scaling_dynamic"]:
                        scales = group["params"][0]
                        temp = scales[:,3:]
                        temp[temp>0.05] = 0.05
                        group["params"][0][:,3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    if group['name'] in ["scaling", "scaling_dynamic"]:
                        scales = group["params"][0]
                        temp = scales[:,3:]
                        temp[temp>0.05] = 0.05
                        group["params"][0][:,3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_anchor_dynamic_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            if group["name"].startswith("dynamic"):
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    if group['name'] in ["scaling", "scaling_dynamic"]:
                        scales = group["params"][0]
                        temp = scales[:, 3:]
                        temp[temp > 0.05] = 0.05
                        group["params"][0][:, 3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    if group['name'] in ["scaling", "scaling_dynamic"]:
                        scales = group["params"][0]
                        temp = scales[:, 3:]
                        temp[temp > 0.05] = 0.05
                        group["params"][0][:, 3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def prune_anchor_dynamic(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_dynamic_optimizer(valid_points_mask)
        self._anchor_dynamic = optimizable_tensors["dynamic_anchor"]
        self._offset_dynamic = optimizable_tensors["dynamic_offset"]
        self._anchor_feat_dynamic = optimizable_tensors["dynamic_anchor_feat"]
        self._opacity_dynamic = optimizable_tensors["dynamic_opacity"]
        self._scaling_dynamic = optimizable_tensors["dynamic_scaling"]
        self._rotation_dynamic = optimizable_tensors["dynamic_rotation"]
        self._temporal_feat = optimizable_tensors["dynamic_feat"]

    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self._anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self._anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor_static.unsqueeze(dim=1) + self._offset * self.get_scaling_activated[:self.get_static_anchor_num, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor_static / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                if self.mode == 'static':
                    new_temporal_feat =  self._temporal_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1, 1]).view([-1, self.keyframe_num, self.temporal_feat_dim])[candidate_mask]
                    new_temporal_feat =  scatter_max(new_temporal_feat, inverse_indices.unsqueeze(1).unsqueeze(2).expand(-1, new_temporal_feat.size(1), new_temporal_feat.size(2)), dim=0)[0][remove_duplicates]

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities
                }

                if self.mode == 'static':
                    d["temporal_feat"] = new_temporal_feat

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], self.time_line, 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], self.time_line, 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                if self.mode == 'static':
                    self._temporal_feat = optimizable_tensors["temporal_feat"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self._anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self._anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        # prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        ratio = 0.3
        opacity_accum = ratio * self.opacity_accum.max(dim=1)[0] + (1 - ratio) * self.opacity_accum.mean(dim=1)
        anchor_demon = ratio * self.anchor_demon.max(dim=1)[0] + (1 - ratio) * self.anchor_demon.mean(dim=1)
        prune_mask = (opacity_accum < min_opacity * anchor_demon).squeeze(dim=1)
        anchors_mask = (anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), self.time_line, 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), self.time_line, 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

    def anchor_growing_dynamic(self, grads, threshold, offset_mask):
        init_length = self._anchor_dynamic.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self._anchor_dynamic.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor_dynamic.unsqueeze(dim=1) + self._offset_dynamic * self.get_scaling_activated[self.get_static_anchor_num:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor_dynamic / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat_dynamic.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                new_temporal_feat = self._temporal_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1, 1]).view([-1, self.keyframe_num, self.temporal_feat_dim])[candidate_mask]
                new_temporal_feat = scatter_max(new_temporal_feat, inverse_indices.unsqueeze(1).unsqueeze(2).expand(-1, new_temporal_feat.size(1), new_temporal_feat.size(2)), dim=0)[0][remove_duplicates]

                d = {
                    "dynamic_anchor": candidate_anchor,
                    "dynamic_scaling": new_scaling,
                    "dynamic_rotation": new_rotation,
                    "dynamic_anchor_feat": new_feat,
                    "dynamic_offset": new_offsets,
                    "dynamic_opacity": new_opacities,
                    "dynamic_feat": new_temporal_feat
                }

                temp_anchor_demon = torch.cat([self.anchor_dynamic_demon, torch.zeros([new_opacities.shape[0], self.time_line, 1], device='cuda').float()], dim=0)
                del self.anchor_dynamic_demon
                self.anchor_dynamic_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_dynamic_accum, torch.zeros([new_opacities.shape[0], self.time_line, 1], device='cuda').float()], dim=0)
                del self.opacity_dynamic_accum
                self.opacity_dynamic_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor_dynamic = optimizable_tensors["dynamic_anchor"]
                self._scaling_dynamic = optimizable_tensors["dynamic_scaling"]
                self._rotation_dynamic = optimizable_tensors["dynamic_rotation"]
                self._anchor_feat_dynamic = optimizable_tensors["dynamic_anchor_feat"]
                self._offset_dynamic = optimizable_tensors["dynamic_offset"]
                self._opacity_dynamic = optimizable_tensors["dynamic_opacity"]
                self._temporal_feat = optimizable_tensors["dynamic_feat"]

    def adjust_anchor_dynamic(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_dynamic_gradient_accum / self.offset_dynamic_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_dynamic_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing_dynamic(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_dynamic_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self._anchor_dynamic.shape[0]*self.n_offsets - self.offset_dynamic_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_dynamic_denom.device)
        self.offset_dynamic_denom = torch.cat([self.offset_dynamic_denom, padding_offset_demon], dim=0)

        self.offset_dynamic_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self._anchor_dynamic.shape[0]*self.n_offsets - self.offset_dynamic_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_dynamic_gradient_accum.device)
        self.offset_dynamic_gradient_accum = torch.cat([self.offset_dynamic_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        # prune_mask = (self.opacity_dynamic_accum < min_opacity*self.anchor_dynamic_demon).squeeze(dim=1)
        opacity_accum = self.opacity_dynamic_accum.max(dim=1)[0]
        anchor_demon = self.anchor_dynamic_demon.max(dim=1)[0]
        prune_mask = (opacity_accum < min_opacity * anchor_demon).squeeze(dim=1)
        anchors_mask = (anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_dynamic_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_dynamic_denom
        self.offset_dynamic_denom = offset_denom

        offset_gradient_accum = self.offset_dynamic_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_dynamic_gradient_accum
        self.offset_dynamic_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_dynamic_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), self.time_line, 1], device='cuda').float()
            self.anchor_dynamic_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), self.time_line, 1], device='cuda').float()

        temp_opacity_accum = self.opacity_dynamic_accum[~prune_mask]
        del self.opacity_dynamic_accum
        self.opacity_dynamic_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_dynamic_demon[~prune_mask]
        del self.anchor_dynamic_demon
        self.anchor_dynamic_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor_dynamic(prune_mask)

    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        _anchor = self.get_anchor
        # _feat = self._anchor_feat
        _feat = self.get_anchor_features
        _grid_offsets = self.get_offset
        _scaling = self.get_scaling
        hash_embeddings = self.get_encoding_params()

        feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets)

        bit_feat = self.entropy_gaussian.forward(_feat, mean, scale, Q_feat)
        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32

        print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash)

        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):

        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()

        _anchor = self.get_anchor
        # _feat = self._anchor_feat
        _feat = self.get_anchor_features
        _grid_offsets = self.get_offset
        _scaling = self.get_scaling

        N = _anchor.shape[0]
        MAX_batch_size = 1_000
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []
        anchor_infos_list = []
        indices_list = []
        min_feat_list = []
        max_feat_list = []
        min_scaling_list = []
        max_scaling_list = []
        min_offsets_list = []
        max_offsets_list = []

        feat_list = []
        scaling_list = []
        offsets_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        torch.save(_anchor, os.path.join(pre_path_name, 'anchor.pkl'))

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            anchor_infos = None
            anchor_infos_list.append(anchor_infos)
            indices_list.append(indices+N_start)

            anchor_sort = _anchor[N_start:N_end][indices]  # [N_num, 3]

            # encode feat
            feat_context = self.calc_interp_feat(anchor_sort)  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat = _feat[N_start:N_end][indices].view(-1)  # [N_num*32]
            feat = STE_multistep.apply(feat, Q_feat, _feat.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_feat, min_feat, max_feat = encoder_gaussian(feat, mean, scale, Q_feat, file_name=feat_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_feat_list.append(bit_feat)
            min_feat_list.append(min_feat)
            max_feat_list.append(max_feat)
            feat_list.append(feat)

            scaling = _scaling[N_start:N_end][indices].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, _scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling, min_scaling, max_scaling = encoder_gaussian(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)
            min_scaling_list.append(min_scaling)
            max_scaling_list.append(max_scaling)
            scaling_list.append(scaling)

            offsets = _grid_offsets[N_start:N_end][indices].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, _grid_offsets.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets, min_offsets, max_offsets = encoder_gaussian(offsets, mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)
            min_offsets_list.append(min_offsets)
            max_offsets_list.append(max_offsets)
            offsets_list.append(offsets)

            torch.cuda.empty_cache()

        bit_anchor = N * 3 * anchor_round_digits
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            p = torch.zeros_like(hash_embeddings).to(torch.float32)
            prob_hash = (((hash_embeddings + 1) / 2).sum() / hash_embeddings.numel()).item()
            p[...] = prob_hash
            bit_hash = encoder(hash_embeddings.view(-1), p.view(-1), file_name=hash_b_name)
        else:
            prob_hash = 0
            bit_hash = hash_embeddings.numel()*32

        indices = torch.cat(indices_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()
        # print('encoding time:', t2 - t1)
        # print('codec time:', t_codec)

        log_info = f"Encoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + self.get_mlp_size()[0])/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        patched_infos = [self.get_static_anchor_num, self.get_anchor_num, N, MAX_batch_size, anchor_infos_list, min_feat_list, max_feat_list, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_hash]
        return patched_infos, log_info

    @torch.no_grad()
    def conduct_encoding_new(self, pre_path_name):

        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()

        _anchor = self.get_anchor
        # _feat = self._anchor_feat
        _feat = self.get_anchor_features
        _grid_offsets = self.get_offset
        _scaling = self.get_scaling

        N = _anchor.shape[0]
        MAX_batch_size = 1000
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        # torch.save(_anchor, os.path.join(pre_path_name, 'anchor.pkl'))
        _quantized_v = self.get_quantized_v
        _quantized_v = _quantized_v.cpu().detach().numpy().astype(np.uint16)
        np.save(os.path.join(pre_path_name, 'anchor.npy'), _quantized_v)

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            anchor_slice = _anchor[N_start:N_end]

            # encode feat
            feat_context = self.calc_interp_feat(anchor_slice)  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat = _feat[N_start:N_end].view(-1)  # [N_num*32]
            feat = STE_multistep.apply(feat, Q_feat, _feat.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_feat = encoder_gaussian_chunk(feat, mean, scale, Q_feat, file_name=feat_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_feat_list.append(bit_feat)

            scaling = _scaling[N_start:N_end].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, _scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling = encoder_gaussian_chunk(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)

            offsets = _grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, _grid_offsets.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets = encoder_gaussian_chunk(offsets, mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)

            torch.cuda.empty_cache()

        bit_anchor = N * 3 * anchor_round_digits
        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            bit_hash = encoder_cuda(((hash_embeddings.view(-1) + 1) / 2), file_name=hash_b_name)
        else:
            bit_hash = hash_embeddings.numel()*32

        torch.cuda.synchronize(); t2 = time.time()
        # print('encoding time:', t2 - t1)
        # print('codec time:', t_codec)

        log_info = f"Encoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + self.get_mlp_size()[0])/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        patched_infos = [self.get_static_anchor_num, self.get_anchor_num, N, MAX_batch_size, self.x_bound_max.cpu().numpy(), self.x_bound_min.cpu().numpy()]
        np.save(os.path.join(pre_path_name, 'patched_infos.npy'), np.asarray(patched_infos, dtype=object))
        return patched_infos, log_info

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, patched_infos, gaussians):
        torch.cuda.synchronize(); t1 = time.time()

        [N_static, N_full, N, MAX_batch_size, anchor_infos_list, min_feat_list, max_feat_list, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_hash] = patched_infos
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        assert N_full == N
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        if self.ste_binary:
            p = torch.zeros_like(self.get_encoding_params()).to(torch.float32)
            p[...] = prob_hash
            hash_embeddings = decoder(p.view(-1), hash_b_name)  # {-1, 1}
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()

        for s in range(steps):
            min_feat = min_feat_list[s]
            max_feat = max_feat_list[s]
            min_scaling = min_scaling_list[s]
            max_scaling = max_scaling_list[s]
            min_offsets = min_offsets_list[s]
            max_offsets = max_offsets_list[s]

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # encode feat
            feat_context = self.calc_interp_feat(anchor_decoded[N_start:N_end])  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
            Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
            Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat_decoded = decoder_gaussian(mean, scale, Q_feat, file_name=feat_b_name, min_value=min_feat, max_value=max_feat)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, min_value=min_scaling, max_value=max_scaling)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            offsets_decoded = decoder_gaussian(mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name, min_value=min_offsets, max_value=max_offsets)
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()

        # fill back N_full
        if self.mode == 'static':
            _anchor = torch.zeros(size=[N_full, 3], device='cuda')
            _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
            _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
            _scaling = torch.zeros(size=[N_full, 6], device='cuda')

            _anchor[:N] = anchor_decoded
            _anchor_feat[:N] = feat_decoded
            _offset[:N] = offsets_decoded
            _scaling[:N] = scaling_decoded

            # replace attributes by decoded ones
            assert self._anchor_feat.shape == _anchor_feat.shape
            gaussians._anchor_feat = nn.Parameter(_anchor_feat)
            assert self._offset.shape == _offset.shape
            gaussians._offset = nn.Parameter(_offset)
            # If change the following attributes, decoded_version must be set True
            gaussians.decoded_version = True
            assert self.get_anchor.shape == _anchor.shape
            gaussians._anchor = nn.Parameter(_anchor)
            assert self._scaling.shape == _scaling.shape
            gaussians._scaling = nn.Parameter(_scaling)
        elif self.mode == 'hybrid':
            _anchor = torch.zeros(size=[N_static, 3], device='cuda')
            _anchor_feat = torch.zeros(size=[N_static, self.feat_dim], device='cuda')
            _offset = torch.zeros(size=[N_static, self.n_offsets, 3], device='cuda')
            _scaling = torch.zeros(size=[N_static, 6], device='cuda')

            _anchor[:N_static] = anchor_decoded[:N_static]
            _anchor_feat[:N_static] = feat_decoded[:N_static]
            _offset[:N_static] = offsets_decoded[:N_static]
            _scaling[:N_static] = scaling_decoded[:N_static]

            # replace attributes by decoded ones
            assert self._anchor_feat.shape == _anchor_feat.shape
            gaussians._anchor_feat = nn.Parameter(_anchor_feat)
            assert self._offset.shape == _offset.shape
            gaussians._offset = nn.Parameter(_offset)
            # If change the following attributes, decoded_version must be set True
            gaussians.decoded_version = True
            assert self._anchor.shape == _anchor.shape
            gaussians._anchor = nn.Parameter(_anchor)
            assert self._scaling.shape == _scaling.shape
            gaussians._scaling = nn.Parameter(_scaling)

            N_dynamic = N_full - N_static
            _anchor_dynamic = torch.zeros(size=[N_dynamic, 3], device='cuda')
            _anchor_feat_dynamic = torch.zeros(size=[N_dynamic, self.feat_dim], device='cuda')
            _offset_dynamic = torch.zeros(size=[N_dynamic, self.n_offsets, 3], device='cuda')
            _scaling_dynamic = torch.zeros(size=[N_dynamic, 6], device='cuda')

            _anchor_dynamic[:N_dynamic] = anchor_decoded[N_static:]
            _anchor_feat_dynamic[:N_dynamic] = feat_decoded[N_static:]
            _offset_dynamic[:N_dynamic] = offsets_decoded[N_static:]
            _scaling_dynamic[:N_dynamic] = scaling_decoded[N_static:]

            # replace attributes by decoded ones
            assert self._anchor_feat_dynamic.shape == _anchor_feat_dynamic.shape
            gaussians._anchor_feat_dynamic = nn.Parameter(_anchor_feat_dynamic)
            assert self._offset_dynamic.shape == _offset_dynamic.shape
            gaussians._offset_dynamic = nn.Parameter(_offset_dynamic)
            # If change the following attributes, decoded_version must be set True
            assert self._anchor_dynamic.shape == _anchor_dynamic.shape
            gaussians._anchor_dynamic = nn.Parameter(_anchor_dynamic)
            assert self._scaling_dynamic.shape == _scaling_dynamic.shape
            gaussians._scaling_dynamic = nn.Parameter(_scaling_dynamic)
        else:
            raise NotImplementedError

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                gaussians.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                gaussians.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                gaussians.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                gaussians.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                gaussians.encoding_xyz.params = nn.Parameter(hash_embeddings)

        log_info = f"DecTime {round(t2 - t1, 4)}"
        return log_info

    @torch.no_grad()
    def conduct_decoding_new(self, pre_path_name, patched_infos, gaussians):
        torch.cuda.synchronize(); t1 = time.time()

        [N_static, N_full, N, MAX_batch_size, x_bound_max, x_bound_min] = patched_infos
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        assert N_full == N
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')

        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = decoder_cuda(N_hash, hash_b_name)  # {0, 1}
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        # anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()
        _quantized_v_decoded = np.load(os.path.join(pre_path_name, 'anchor.npy')).astype(np.int32)
        _quantized_v_decoded = torch.from_numpy(_quantized_v_decoded).cuda().to(torch.int32)
        interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        anchor_decoded = _quantized_v_decoded * interval + self.x_bound_min

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # encode feat
            feat_context = self.calc_interp_feat(anchor_decoded[N_start:N_end])  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
            Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
            Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat_decoded = decoder_gaussian_chunk(mean, scale, Q_feat, file_name=feat_b_name)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            offsets_decoded = decoder_gaussian_chunk(mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name)
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()

        # fill back N_full
        if self.mode == 'static':
            _anchor = torch.zeros(size=[N_full, 3], device='cuda')
            _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
            _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
            _scaling = torch.zeros(size=[N_full, 6], device='cuda')

            _anchor[:N] = anchor_decoded
            _anchor_feat[:N] = feat_decoded
            _offset[:N] = offsets_decoded
            _scaling[:N] = scaling_decoded

            # replace attributes by decoded ones
            assert self._anchor_feat.shape == _anchor_feat.shape
            gaussians._anchor_feat = nn.Parameter(_anchor_feat)
            assert self._offset.shape == _offset.shape
            gaussians._offset = nn.Parameter(_offset)
            # If change the following attributes, decoded_version must be set True
            gaussians.decoded_version = True
            assert self.get_anchor.shape == _anchor.shape
            gaussians._anchor = nn.Parameter(_anchor)
            assert self._scaling.shape == _scaling.shape
            gaussians._scaling = nn.Parameter(_scaling)
        elif self.mode == 'hybrid':
            _anchor = torch.zeros(size=[N_static, 3], device='cuda')
            _anchor_feat = torch.zeros(size=[N_static, self.feat_dim], device='cuda')
            _offset = torch.zeros(size=[N_static, self.n_offsets, 3], device='cuda')
            _scaling = torch.zeros(size=[N_static, 6], device='cuda')

            _anchor[:N_static] = anchor_decoded[:N_static]
            _anchor_feat[:N_static] = feat_decoded[:N_static]
            _offset[:N_static] = offsets_decoded[:N_static]
            _scaling[:N_static] = scaling_decoded[:N_static]

            # replace attributes by decoded ones
            assert self._anchor_feat.shape == _anchor_feat.shape
            gaussians._anchor_feat = nn.Parameter(_anchor_feat)
            assert self._offset.shape == _offset.shape
            gaussians._offset = nn.Parameter(_offset)
            # If change the following attributes, decoded_version must be set True
            gaussians.decoded_version = True
            assert self._anchor.shape == _anchor.shape
            gaussians._anchor = nn.Parameter(_anchor)
            assert self._scaling.shape == _scaling.shape
            gaussians._scaling = nn.Parameter(_scaling)

            N_dynamic = N_full - N_static
            _anchor_dynamic = torch.zeros(size=[N_dynamic, 3], device='cuda')
            _anchor_feat_dynamic = torch.zeros(size=[N_dynamic, self.feat_dim], device='cuda')
            _offset_dynamic = torch.zeros(size=[N_dynamic, self.n_offsets, 3], device='cuda')
            _scaling_dynamic = torch.zeros(size=[N_dynamic, 6], device='cuda')

            _anchor_dynamic[:N_dynamic] = anchor_decoded[N_static:]
            _anchor_feat_dynamic[:N_dynamic] = feat_decoded[N_static:]
            _offset_dynamic[:N_dynamic] = offsets_decoded[N_static:]
            _scaling_dynamic[:N_dynamic] = scaling_decoded[N_static:]

            # replace attributes by decoded ones
            assert self._anchor_feat_dynamic.shape == _anchor_feat_dynamic.shape
            gaussians._anchor_feat_dynamic = nn.Parameter(_anchor_feat_dynamic)
            assert self._offset_dynamic.shape == _offset_dynamic.shape
            gaussians._offset_dynamic = nn.Parameter(_offset_dynamic)
            # If change the following attributes, decoded_version must be set True
            assert self._anchor_dynamic.shape == _anchor_dynamic.shape
            gaussians._anchor_dynamic = nn.Parameter(_anchor_dynamic)
            assert self._scaling_dynamic.shape == _scaling_dynamic.shape
            gaussians._scaling_dynamic = nn.Parameter(_scaling_dynamic)
        else:
            raise NotImplementedError

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                gaussians.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                gaussians.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                gaussians.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                gaussians.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                gaussians.encoding_xyz.params = nn.Parameter(hash_embeddings)

        log_info = f"DecTime {round(t2 - t1, 4)}"
        return log_info

    @torch.no_grad()
    def conduct_decoding_from_files(self, pre_path_name):
        torch.cuda.synchronize(); t1 = time.time()

        patched_infos = np.load(os.path.join(pre_path_name, 'patched_infos.npy'), allow_pickle=True)
        [N_static, N_full, N, MAX_batch_size, x_bound_max, x_bound_min] = patched_infos
        self.x_bound_max = torch.from_numpy(x_bound_max).cuda().to(torch.float)
        self.x_bound_min = torch.from_numpy(x_bound_min).cuda().to(torch.float)

        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        assert N_full == N
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = decoder_cuda(N_hash, hash_b_name)  # {0, 1}
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        # anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()
        _quantized_v_decoded = np.load(os.path.join(pre_path_name, 'anchor.npy')).astype(np.int32)
        _quantized_v_decoded = torch.from_numpy(_quantized_v_decoded).cuda().to(torch.int32)
        interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        anchor_decoded = _quantized_v_decoded * interval + self.x_bound_min

        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            # encode feat
            feat_context = self.calc_interp_feat(anchor_decoded[N_start:N_end])  # [N_num, ?]
            # many [N_num, ?]
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets, 3 * self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
            Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
            Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat_decoded = decoder_gaussian_chunk(mean, scale, Q_feat, file_name=feat_b_name)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            offsets_decoded = decoder_gaussian_chunk(mean_offsets, scale_offsets, Q_offsets, file_name=offsets_b_name)
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()

        # fill back N_full
        if self.mode == 'static':
            _anchor = torch.zeros(size=[N_full, 3], device='cuda')
            _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
            _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
            _scaling = torch.zeros(size=[N_full, 6], device='cuda')

            _anchor[:N] = anchor_decoded
            _anchor_feat[:N] = feat_decoded
            _offset[:N] = offsets_decoded
            _scaling[:N] = scaling_decoded

            # replace attributes by decoded ones
            self._anchor_feat = _anchor_feat
            self._offset = _offset
            # If change the following attributes, decoded_version must be set True
            self.decoded_version = True
            self._anchor = _anchor
            self._scaling = _scaling
        elif self.mode == 'hybrid':
            _anchor = torch.zeros(size=[N_static, 3], device='cuda')
            _anchor_feat = torch.zeros(size=[N_static, self.feat_dim], device='cuda')
            _offset = torch.zeros(size=[N_static, self.n_offsets, 3], device='cuda')
            _scaling = torch.zeros(size=[N_static, 6], device='cuda')

            _anchor[:N_static] = anchor_decoded[:N_static]
            _anchor_feat[:N_static] = feat_decoded[:N_static]
            _offset[:N_static] = offsets_decoded[:N_static]
            _scaling[:N_static] = scaling_decoded[:N_static]

            # replace attributes by decoded ones
            self._anchor_feat = _anchor_feat
            self._offset = _offset
            # If change the following attributes, decoded_version must be set True
            self.decoded_version = True
            self._anchor = _anchor
            self._scaling = _scaling

            N_dynamic = N_full - N_static
            _anchor_dynamic = torch.zeros(size=[N_dynamic, 3], device='cuda')
            _anchor_feat_dynamic = torch.zeros(size=[N_dynamic, self.feat_dim], device='cuda')
            _offset_dynamic = torch.zeros(size=[N_dynamic, self.n_offsets, 3], device='cuda')
            _scaling_dynamic = torch.zeros(size=[N_dynamic, 6], device='cuda')

            _anchor_dynamic[:N_dynamic] = anchor_decoded[N_static:]
            _anchor_feat_dynamic[:N_dynamic] = feat_decoded[N_static:]
            _offset_dynamic[:N_dynamic] = offsets_decoded[N_static:]
            _scaling_dynamic[:N_dynamic] = scaling_decoded[N_static:]

            # replace attributes by decoded ones
            self._anchor_feat_dynamic = _anchor_feat_dynamic
            self._offset_dynamic = _offset_dynamic
            self._anchor_dynamic = _anchor_dynamic
            self._scaling_dynamic = _scaling_dynamic
        else:
            raise NotImplementedError

        log_info = f"DecTime {round(t2 - t1, 4)}"
        return log_info

    @torch.no_grad()
    def frame_maker(self, img: torch.Tensor, img_width, img_height):
        frame = torch.zeros((img_width*4, img_height*4, 1), device=img.device)
        x, y = 0, 0
        for i in range(img.shape[2]):
            if y >= frame.shape[1]:
                y = 0
                x += img_width
            assert x + img_width <= frame.shape[0], f"Frame_maker: not enough space, needs {x + img_width}x{y} but only have {img_width*4}"

            frame[x:x+img_width, y:y+img_height, 0] = img[:, :, i]
            y += img_height
        return frame

    @torch.no_grad()
    def as_grid_img_dynamic(self, gs_attr):
        N_dynamic = self._anchor_dynamic.shape[0]
        self.dynamic_grid_sidelen = math.ceil(math.sqrt(N_dynamic))
        target_elements = self.dynamic_grid_sidelen * self.dynamic_grid_sidelen
        padded_tensor = torch.zeros(target_elements, self.temporal_feat_dim, dtype=self._anchor_dynamic.dtype, device=gs_attr.device)
        padded_tensor[:N_dynamic, :] = gs_attr
        img = padded_tensor.reshape((self.dynamic_grid_sidelen, self.dynamic_grid_sidelen, -1))
        return img

    @torch.no_grad()
    def attr_as_grid_img(self, attr_name):
        if 'temporal_feat' in attr_name:
            imgs = []
            for i in range(self.keyframe_num):
                gs_attr = self._temporal_feat[:, i, :]
                gs_attr = gs_attr.detach().cpu()

                img = self.as_grid_img_dynamic(gs_attr)  # width, length, 16
                img = self.frame_maker(img, self.dynamic_grid_sidelen, self.dynamic_grid_sidelen)  # width * 4, length * 4
                imgs.append(img)
            return imgs
        else:
            print(f'attr_name={attr_name}')
            raise ValueError

    @torch.no_grad()
    def set_point_feat_from_grid_img(self, feat_imgs, keyframe_num, N_dynamic):
        new_point_feats = None
        width = feat_imgs[0].shape[0] // 4
        height = feat_imgs[0].shape[1] // 4

        for idx, img in enumerate(feat_imgs):
            feats = np.zeros((width, height, self.temporal_feat_dim))
            x, y = 0, 0
            for i in range(self.temporal_feat_dim):
                if y >= img.shape[1]:
                    y = 0
                    x += width
                assert x + width <= img.shape[0], f"Tile_maker: not enough space, needs {x + width}x{y} but only have {width * 4}"
                feats[:, :, i] = img[x:x+width, y:y+height]
                y += height
            feats = feats.reshape(-1, self.temporal_feat_dim)
            tensor = torch.tensor(feats, dtype=torch.float, device="cuda")
            if new_point_feats is None:
                new_point_feats = torch.zeros((width*height, keyframe_num, self.temporal_feat_dim), device="cuda")
                new_point_feats[:, idx, :] = tensor
            else:
                new_point_feats[:, idx, :] = tensor
        new_point_feats = new_point_feats.contiguous()
        new_point_feats = new_point_feats[:N_dynamic]
        setattr(self, f'_temporal_feat', new_point_feats)

    @torch.no_grad()
    def conduct_encoding_for_ntc(self, pre_path_name):
        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()

        # triplanes
        ntc_2D_b_name = os.path.join(pre_path_name, 'ntc_2D.b')
        ntc_2D_embeddings = self.get_ntc_2D_params()  # {-1, 1}
        if self.ste_binary:
            # p = torch.zeros_like(ntc_2D_embeddings).to(torch.float32)
            # prob_ntc_2D = (((ntc_2D_embeddings + 1) / 2).sum() / ntc_2D_embeddings.numel()).item()
            # p[...] = prob_ntc_2D
            # bit_ntc_2D = encoder(ntc_2D_embeddings.view(-1), p.view(-1), file_name=ntc_2D_b_name)
            bit_ntc_2D = encoder_cuda(((ntc_2D_embeddings.view(-1) + 1) / 2), file_name=ntc_2D_b_name)
        else:
            # prob_ntc_2D = 0
            bit_ntc_2D = ntc_2D_embeddings.numel()*32
        
        # 3D grid
        ntc_3D_b_name = os.path.join(pre_path_name, 'ntc_3D.b')
        ntc_3D_embeddings = self.get_ntc_3D_params()  # {-1, 1}
        if self.ste_binary:
            # p = torch.zeros_like(ntc_3D_embeddings).to(torch.float32)
            # prob_ntc_3D = (((ntc_3D_embeddings + 1) / 2).sum() / ntc_3D_embeddings.numel()).item()
            # p[...] = prob_ntc_3D
            # bit_ntc_3D = encoder(ntc_3D_embeddings.view(-1), p.view(-1), file_name=ntc_3D_b_name)
            bit_ntc_3D = encoder_cuda(((ntc_3D_embeddings.view(-1) + 1) / 2), file_name=ntc_3D_b_name)
        else:
            # prob_ntc_3D = 0
            bit_ntc_3D = ntc_3D_embeddings.numel()*32

        torch.cuda.synchronize(); t2 = time.time()
        # print('encoding time:', t2 - t1)
        # print('codec time:', t_codec)

        ntc_2D = bit_ntc_2D/bit2MB_scale
        ntc_3D = bit_ntc_3D/bit2MB_scale
        ntc_mlp = self.get_ntc_mlp_size()[0]/bit2MB_scale
        Total = ntc_2D + ntc_3D + ntc_mlp
        Total_wo3D = ntc_2D + ntc_mlp

        log_info = f"ntc_2D {round(ntc_2D, 4)}, " \
                   f"ntc_3D {round(ntc_3D, 4)}, " \
                   f"ntc_mlp {round(ntc_mlp, 4)}, " \
                   f"Total {round( Total, 4)}, " \
                   f"Total wo3D {round(Total_wo3D, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_decoding_for_ntc(self, pre_path_name):
        torch.cuda.synchronize(); t1 = time.time()

        ntc_2D_b_name = os.path.join(pre_path_name, 'ntc_2D.b')
        ntc_3D_b_name = os.path.join(pre_path_name, 'ntc_3D.b')

        if self.ste_binary:
            # p = torch.zeros_like(self.get_ntc_2D_params()).to(torch.float32)
            # p[...] = prob_ntc_2D
            # ntc_2D_embeddings = decoder(p.view(-1), ntc_2D_b_name)  # {-1, 1}
            N_2D_hash = torch.zeros_like(self.get_ntc_2D_params()).numel()
            ntc_2D_embeddings = decoder_cuda(N_2D_hash, ntc_2D_b_name)
            ntc_2D_embeddings = (ntc_2D_embeddings * 2 - 1).to(torch.float32)
            ntc_2D_embeddings = ntc_2D_embeddings.view(-1, self.n_features_per_level)

            # p = torch.zeros_like(self.get_ntc_3D_params()).to(torch.float32)
            # p[...] = prob_ntc_3D
            # ntc_3D_embeddings = decoder(p.view(-1), ntc_3D_b_name)  # {-1, 1}
            N_3D_hash = torch.zeros_like(self.get_ntc_3D_params()).numel()
            ntc_3D_embeddings = decoder_cuda(N_3D_hash, ntc_3D_b_name)
            ntc_3D_embeddings = (ntc_3D_embeddings * 2 - 1).to(torch.float32)
            ntc_3D_embeddings = ntc_3D_embeddings.view(-1, 1)

        torch.cuda.synchronize(); t2 = time.time()

        if self.ste_binary:
            # 2D
            len_2D = self.ntc.encoding_xy.params.shape[0]
            self.ntc.encoding_xy.params = nn.Parameter(ntc_2D_embeddings[0:len_2D])
            self.ntc.encoding_xz.params = nn.Parameter(ntc_2D_embeddings[len_2D:len_2D*2])
            self.ntc.encoding_yz.params = nn.Parameter(ntc_2D_embeddings[len_2D*2:len_2D*3])

            # 3D
            self.ntc.encoding_xyz.params = nn.Parameter(ntc_3D_embeddings)

        log_info = f"DecTime {round(t2 - t1, 4)}"

        return log_info

