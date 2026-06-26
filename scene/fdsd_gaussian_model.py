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
import copy
import math
import os
from collections import OrderedDict

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from plas import sort_with_plas
from simple_knn._C import distCUDA2
from torch import nn

from compression.decoders import LatentDecoder
# from utils.encodings import GridEncoder
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH


# from utils.relocation_utils import compute_relocation_cuda


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=1, include_input=True):
    if i == -1:
        return torch.nn.Identity(), 3

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def linear_interp_uniiterval(y1, y2, t):
    return (y1 * (1 - t) + y2 * t)


def cube_interpolate(y_k1, y_k2, y_m1, y_m2, delta_t):
    # hermite basis
    h_00 = lambda x :  2*x**3 - 3*x**2 + 1
    h_10 = lambda x :    x**3 - 2*x**2 + x
    h_01 = lambda x : -2*x**3 + 3*x**2
    h_11 = lambda x :    x**3 -   x**2

    p_x = h_00(delta_t) * y_k1 + h_10(delta_t) * y_m1 + \
          h_01(delta_t) * y_k2 + h_11(delta_t) * y_m2

    return p_x


def cube_diff_interpolate(y_km1, y_k, y_k1, y_k2, delta_t):
    # hermite basis
    h_00 = lambda x :  2*x**3 - 3*x**2 + 1
    h_10 = lambda x :    x**3 - 2*x**2 + x
    h_01 = lambda x : -2*x**3 + 3*x**2
    h_11 = lambda x :    x**3 -   x**2

    m_k  = (y_k1 - y_km1) / 2
    m_k1 = (y_k2 - y_k) / 2

    p_x = h_00(delta_t) * y_k + h_10(delta_t) * m_k + h_01(delta_t) * y_k1 + h_11(delta_t) * m_k1

    return p_x


def slerp_interpolate(v1, v2, t):
    # https://en.wikipedia.org/wiki/Slerp
    # normalizse the input vectors
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    d = (v1 * v2).sum(-1, keepdim=True).clamp(-1 + 1e-4, 1 - 1e-4)
    omega = torch.acos(d).clamp_min(1e-4)
    s_omega = torch.sin(omega).clamp_min(1e-4)
    p_0 = torch.sin((1 - t) * omega) / s_omega
    p_1 = torch.sin(t * omega) / s_omega
    p_sum = (p_0 + p_1).clamp_min(1e-4)
    p_0 = p_0 / p_sum
    p_1 = p_1 / p_sum

    # prevent zero vector
    ret = (v1 * p_0 + v2 * p_1)
    ret = torch.where(ret.abs().sum(-1, keepdim=True) > 1e-4, ret, v1)

    return ret / ret.norm(dim=-1, keepdim=True)


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
        # self.opacity_activation = torch.tanh
        # self.inverse_opacity_activation = torch.atanh

        self.rotation_activation = torch.nn.functional.normalize

    def setup_compression_decoder(self):
        # self.quat_decoders = OrderedDict()
        self.quat_attrbutes = ['features_dc_dynamic_offset', 'features_rest_dynamic_offset', 'scaling_dynamic_offset', 'rotation_dynamic_offset', 'opacity_dynamic_offset', 'language_feature_dynamic_offset']
        self.quat_decoder_latent_dims = [8, 4, 8, 6, 4, 6]
        self.quat_decoder_feature_dims = [3, 3 * ((self.max_sh_degree + 1) ** 2 - 1), 3, 4, self.opacity_dim, 9]
        self.quat_decoders = nn.ModuleDict()
        self.quat_decoders['features_dc_dynamic_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[0],  # 8
            feature_dim=self.quat_decoder_feature_dims[0],  # 3
            ldecode_matrix="learnable",
        ).cuda()
        self.quat_decoders['features_rest_dynamic_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[1],  #4,
            feature_dim=self.quat_decoder_feature_dims[1],  #3 * ((self.max_sh_degree + 1) ** 2 - 1),
            ldecode_matrix="learnable",
        ).cuda()
        self.quat_decoders['scaling_dynamic_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[2],  #8,
            feature_dim=self.quat_decoder_feature_dims[2],  #3,
            ldecode_matrix="learnable",
        ).cuda()
        self.quat_decoders['rotation_dynamic_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[3],  #6,
            feature_dim=self.quat_decoder_feature_dims[3],  #4,
            ldecode_matrix="learnable",
        ).cuda()
        self.quat_decoders['opacity_dynamic_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[4],  #3,
            feature_dim=self.quat_decoder_feature_dims[4],  #1,
            ldecode_matrix="learnable",
        ).cuda()
        self.quat_decoders['language_feature_dynamic_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[5],  #3,
            feature_dim=self.quat_decoder_feature_dims[5],  #1,
            ldecode_matrix="learnable",
        ).cuda()

        self.quat_static_attrbutes = ['features_dc_offset', 'features_rest_offset', 'scaling_offset', 'rotation_offset', 'opacity_offset', 'language_feature_offset']
        self.quat_decoders['features_dc_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[0],  # 8
            feature_dim=self.quat_decoder_feature_dims[0],  # 3
            ldecode_matrix="learnable",
            # use_shift=False
        ).cuda()
        self.quat_decoders['features_rest_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[1],  #4,
            feature_dim=self.quat_decoder_feature_dims[1],  #3 * ((self.max_sh_degree + 1) ** 2 - 1),
            ldecode_matrix="learnable",
            # use_shift=False
        ).cuda()
        self.quat_decoders['scaling_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[2],  #8,
            feature_dim=self.quat_decoder_feature_dims[2],  #3,
            ldecode_matrix="learnable",
        ).cuda()
        self.quat_decoders['rotation_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[3],  #6,
            feature_dim=self.quat_decoder_feature_dims[3],  #4,
            ldecode_matrix="learnable",
            # ldec_std=0.01
        ).cuda()
        self.quat_decoders['opacity_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[4],  #3,
            feature_dim=self.quat_decoder_feature_dims[4],  #1,
            ldecode_matrix="learnable",
            # use_shift=False
        ).cuda()
        self.quat_decoders['language_feature_offset'] = LatentDecoder(
            latent_dim=self.quat_decoder_latent_dims[5],  #3,
            feature_dim=self.quat_decoder_feature_dims[5],  #1,
            ldecode_matrix="learnable",
        ).cuda()

    def setup_interpolators(self, pos='chip', rot='slerp'):
        if pos == 'chip':
            self.interpolator = self.chip
        elif pos == 'lerp':
            self.interpolator = self.lerp
        else:
            raise NotImplementedError

        if rot == 'slerp':
            self.rot_interpolator = self.slerp
        elif rot == 'lerp':
            self.rot_interpolator = self.lerp
        else:
            raise NotImplementedError

    def __init__(self, sh_degree : int, disable_xyz_log_activation, eval=False, language=True):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.disable_xyz_log_activation = disable_xyz_log_activation
        self.kernel_size = 0.1

        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')

        # static gaussian
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self._importance = torch.empty(0).cuda()
        self._language_feature = None
        self.lang_feat_dim = 9
        if language:
            self.opacity_dim = 2
        else:
            self.opacity_dim = 1

        self._xyz_dynamic_mask = None
        self._dynamic = torch.empty(0).cuda()

        # dynamic gaussian
        self._xyz_dynamic = torch.empty(0).cuda()
        self._features_dc_dynamic = torch.empty(0).cuda()
        self._features_rest_dynamic = torch.empty(0).cuda()
        self._scaling_dynamic = torch.empty(0).cuda()
        self._rotation_dynamic = torch.empty(0).cuda()
        self._opacity_dynamic = torch.empty(0).cuda()
        self._importance_dynamic = torch.empty(0).cuda()
        self._language_feature_dynamic = None

        # attribute offset
        self.offset_mode = 0
        self._xyz_offset = torch.empty(0).cuda()
        self._features_dc_offset = torch.empty(0).cuda()
        self._features_rest_offset = torch.empty(0).cuda()
        self._scaling_offset = torch.empty(0).cuda()
        self._rotation_offset = torch.empty(0).cuda()
        self._opacity_offset = torch.empty(0).cuda()
        self._language_feature_offset = torch.empty(0).cuda()

        self._xyz_dynamic_offset = torch.empty(0).cuda()
        self._features_dc_dynamic_offset = torch.empty(0).cuda()
        self._features_rest_dynamic_offset = torch.empty(0).cuda()
        self._scaling_dynamic_offset = torch.empty(0).cuda()
        self._rotation_dynamic_offset = torch.empty(0).cuda()
        self._opacity_dynamic_offset = torch.empty(0).cuda()
        self._language_feature_dynamic_offset = torch.empty(0).cuda()

        # entropy compression
        self.setup_compression_decoder()
        # self.shs_entropy_model = EntropyBottleneck(channels=(1+sh_degree)**2-1, entropy_coder='rangecoder').to('cuda')
        # self.lang_entropy_model = EntropyBottleneck(channels=self.lang_feat_dim, entropy_coder='rangecoder').to('cuda')

        # interpolate params
        self.time_line = 60
        self.interval = 10
        self.keyframe_num = self.time_line // self.interval + 1
        self.time_pad = 0
        self.time_shift = self.time_pad

        def cubic_interpolation(y1, y1d, y2, y2d, delta_t):
            return cube_interpolate(y1, y2, y1d, y2d, delta_t).squeeze(1)

        def linear_interpolation(y1, n1, y2, n2, delta_t):
            return linear_interp_uniiterval(y1, y2, delta_t).squeeze(1)

        def slerp_interpolation(y1, n1, y2, n2, delta_t):
            return slerp_interpolate(y1, y2, delta_t).squeeze(1)

        self.chip = cubic_interpolation
        self.slerp = slerp_interpolation
        self.lerp = linear_interpolation

        self.interpolator = cubic_interpolation
        self.rot_interpolator = slerp_interpolation
        self.linear_interpolator = linear_interpolation

        if eval:
            self._xyz_keyframe = torch.empty(0).cuda()  # [N, keyframe, 6]
            self._rotation_keyframe = torch.empty(0).cuda()  # [N, keyframe, 4]
            self._scaling_keyframe = torch.empty(0).cuda()  # [N, keyframe, 3]

        self.feat_dim = 16
        self.time_batch = 1
        # self.time_per_feat = self.feat_dim // self.time_batch
        self.total_feat_dim = self.feat_dim * self.time_batch
        self._point_feats = torch.empty(0).cuda()
        self.time_embedding_num = self.keyframe_num

        self.time_pe_fn, time_pe_ch = get_embedder(7, 1, include_input=False)
        self.xyz_pe_fn, xyz_pe_ch = get_embedder(6, 3, include_input=False)
        # print(f"time pe dim = {time_pe_ch}, position pe dim = {xyz_pe_ch}")

        input_dim = self.feat_dim
        hidden = 64
        bias = False

        # self.mlp_time = nn.Sequential(
        #     nn.Linear(self.total_feat_dim+time_pe_ch, hidden, bias=bias),
        #     nn.ReLU(True),
        #     nn.Linear(hidden, hidden, bias=bias),
        #     nn.Tanh()
        # ).cuda()

        self.mlp_deform = nn.Sequential(
            nn.Linear(12, hidden, bias=bias),  # 12  16
            nn.ReLU(True),
            nn.Linear(hidden, 6, bias=bias),
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(12, hidden, bias=bias),  # 12  16
            nn.ReLU(True),
            nn.Linear(hidden, 7, bias=bias),
        ).cuda()

        self.mlp_opacity = nn.Sequential(
            nn.Linear(3+1+8, hidden, bias=bias),  # 8  16
            nn.ReLU(True),
            nn.Linear(hidden, self.opacity_dim, bias=bias),
            nn.Tanh()
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(3+1+8, hidden, bias=bias),  # 8  16
            nn.ReLU(True),
            nn.Linear(hidden, ((self.max_sh_degree + 1) ** 2) * 3, bias=bias),
        ).cuda()

        self.mlp_lang = nn.Sequential(
            nn.Linear(8, hidden, bias=bias),
            nn.ReLU(True),
            nn.Linear(hidden, self.lang_feat_dim, bias=bias),
        ).cuda()

        self.max_radii2D = torch.empty(0).cuda()
        self.min_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()

        self.opacity_accum = torch.empty(0).cuda()
        self.dynamic_max_radii2D = torch.empty(0).cuda()
        self.dynamic_min_radii2D = torch.empty(0).cuda()
        self.dynamic_xyz_gradient_accum = torch.empty(0).cuda()
        self.dynamic_denom = torch.empty(0).cuda()

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def eval(self):
        # self.mlp_time.eval()
        self.mlp_deform.eval()
        self.mlp_cov.eval()
        self.mlp_opacity.eval()
        self.mlp_color.eval()

    def train(self):
        # self.mlp_time.train()
        self.mlp_deform.train()
        self.mlp_cov.train()
        self.mlp_opacity.train()
        self.mlp_color.train()

    @property
    def get_scaling(self):
        if self.offset_mode == 11:
            return self.scaling_activation(self._scaling + self._scaling_offset)
        elif self.offset_mode == 22:
            return self.scaling_activation(self._scaling + self.quat_decoders['scaling_offset'](self._scaling_offset))
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_ori(self):
        if self.offset_mode == 11:
            return self._scaling + self._scaling_offset
        elif self.offset_mode == 22:
            return self._scaling + self.quat_decoders['scaling_offset'](self._scaling_offset)
        return self._scaling

    @property
    def get_scaling_dynamic(self):
        if self.offset_mode % 10 == 1:
            return self.scaling_activation(self._scaling_dynamic + self._scaling_dynamic_offset)
        elif self.offset_mode % 10 == 2:
            return self.scaling_activation(self._scaling_dynamic + self.quat_decoders['scaling_dynamic_offset'](self._scaling_dynamic_offset))
        return self.scaling_activation(self._scaling_dynamic)

    @property
    def get_scaling_dynamic_ori(self):
        if self.offset_mode % 10 == 1:
            return self._scaling_dynamic + self._scaling_dynamic_offset
        elif self.offset_mode % 10 == 2:
            return self._scaling_dynamic + self.quat_decoders['scaling_dynamic_offset'](self._scaling_dynamic_offset)
        return self._scaling_dynamic
    
    @property
    def get_rotation(self):
        if self.offset_mode == 11:
            return self.rotation_activation(self._rotation + self._rotation_offset)
        elif self.offset_mode == 22:
            return self.rotation_activation(self._rotation + self.quat_decoders['rotation_offset'](self._rotation_offset))
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_ori(self):
        if self.offset_mode == 11:
            return self._rotation + self._rotation_offset
        elif self.offset_mode == 22:
            return self._rotation + self.quat_decoders['rotation_offset'](self._rotation_offset)
        return self._rotation

    @property
    def get_rotation_dynamic_ori(self):
        if self.offset_mode % 10 == 1:
            return self._rotation_dynamic + self._rotation_dynamic_offset
        elif self.offset_mode % 10 == 2:
            return self._rotation_dynamic + self.quat_decoders['rotation_dynamic_offset'](self._rotation_dynamic_offset)

        return self._rotation_dynamic

    @property
    def get_xyz(self):
        if self.offset_mode >= 10:
            return self._xyz_offset
        return self._xyz

    @property
    def get_xyz_all(self):
        return torch.cat([self.get_xyz, self.get_dynamic_xyz], dim=0)

    @property
    def get_dynamic_xyz(self):
        if 1 <= self.offset_mode <= 22:
            return self._xyz_dynamic_offset
        else:
            return self._xyz_dynamic

    @property
    def get_dynamic(self):
        return self._dynamic

    def get_time_features(self, timestamp, noise=0.):
        t = int(timestamp * self.time_line)
        t += self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval

        t_idx = int(t_idx)
        feat = self._point_feats[:, t_idx, :]
        feat_next = self._point_feats[:, t_idx + 1, :]

        return [feat, feat_next], t_idx, delta_t

    def get_xyz_rot_keyframe(self, timestamp):
        t = int(timestamp * self.time_line)
        t += self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval

        t_idx = int(t_idx)

        feat = self._point_feats[:, t_idx, :]
        feat_next = self._point_feats[:, t_idx + 1, :]
        lip_feat = self.linear_interpolator(feat, None, feat_next, None, delta_t)

        xyz_off = self._xyz_keyframe[:, t_idx, :]
        xyz_off_next = self._xyz_keyframe[:, t_idx + 1, :]
        dy_loc = self.get_dynamic_xyz + xyz_off[:, :3]
        dy_loc_next = self.get_dynamic_xyz + xyz_off_next[:, :3]
        xyz_dynamic = self.interpolator(dy_loc, xyz_off[:, 3:], dy_loc_next, xyz_off_next[:, 3:], delta_t)

        dy_rot = self._rotation_keyframe[:, t_idx, :]
        dy_rot_next = self._rotation_keyframe[:, t_idx + 1, :]
        rotation_dynamic = self.rot_interpolator(dy_rot, None, dy_rot_next, None, delta_t)
        return (xyz_dynamic, rotation_dynamic, lip_feat)

    def get_scale_keyframe(self, timestamp):
        t = int(timestamp * self.time_line)
        t += self.time_shift
        t_idx = t // self.interval
        return self._scaling_keyframe[:, t_idx, :]

    @property
    def get_opacity(self):
        if self.offset_mode == 11:
            return self.opacity_activation(self._opacity + self._opacity_offset)
        elif self.offset_mode == 22:
            return self.opacity_activation(self._opacity + self.quat_decoders['opacity_offset'](self._opacity_offset))
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_ori(self):
        if self.offset_mode == 11:
            return self._opacity + self._opacity_offset
        elif self.offset_mode == 22:
            return self._opacity + self.quat_decoders['opacity_offset'](self._opacity_offset)
        return self._opacity

    @property
    def get_colored_opacity(self):
        if self.offset_mode == 11:
            return self.opacity_activation(self._opacity[:, 0:1] + self._opacity_offset[:, 0:1])
        elif self.offset_mode == 22:
            return self.opacity_activation(self._opacity[:, 0:1] + self.quat_decoders['opacity_offset'](self._opacity_offset)[:, 0:1])
        return self.opacity_activation(self._opacity[:, 0:1])

    @property
    def get_colored_opacity_ori(self):
        if self.offset_mode == 11:
            return self._opacity[:, 0:1] + self._opacity_offset[:, 0:1]
        elif self.offset_mode == 22:
            return self._opacity[:, 0:1] + self.quat_decoders['opacity_offset'](self._opacity_offset)[:, 0:1]
        return self._opacity[:, 0:1]

    @property
    def get_opacity_dynamic_ori(self):
        if self.offset_mode % 10 == 1:
            return self._opacity_dynamic + self._opacity_dynamic_offset
        elif self.offset_mode % 10 == 2:
            return self._opacity_dynamic + self.quat_decoders['opacity_dynamic_offset'](self._opacity_dynamic_offset)
        return self._opacity_dynamic

    @property
    def get_static_shs(self):
        return torch.cat((self.get_feature_dc_ori, self.get_feature_rest_ori), dim=1)

    @property
    def get_feature_dc_ori(self):
        if self.offset_mode == 11:
            return self._features_dc + self._features_dc_offset
        elif self.offset_mode == 22:
            features_dc_offset = self.quat_decoders['features_dc_offset'](self._features_dc_offset)
            return self._features_dc + features_dc_offset.reshape(self._features_dc.shape[0], 1, 3)
        return self._features_dc

    @property
    def get_feature_rest_ori(self):
        if self.offset_mode == 11:
            return self._features_rest + self._features_rest_offset
        elif self.offset_mode == 22:
            features_rest_offset = self.quat_decoders['features_rest_offset'](self._features_rest_offset)
            return self._features_rest + features_rest_offset.reshape(self._features_rest.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)
        return self._features_rest

    @property
    def get_dynamic_shs(self):
        return torch.cat((self.get_feature_dc_dynamic_ori, self.get_feature_rest_dynamic_ori), dim=1)

    @property
    def get_feature_dc_dynamic_ori(self):
        if self.offset_mode % 10 == 1:
            return self._features_dc_dynamic + self._features_dc_dynamic_offset
        elif self.offset_mode % 10 == 2:
            features_dc_dynamic_offset = self.quat_decoders['features_dc_dynamic_offset'](self._features_dc_dynamic_offset)
            return self._features_dc_dynamic + features_dc_dynamic_offset.reshape(self._features_dc_dynamic.shape[0], 1, 3)
        return self._features_dc_dynamic

    @property
    def get_feature_rest_dynamic_ori(self):
        if self.offset_mode % 10 == 1:
            return self._features_rest_dynamic + self._features_rest_dynamic_offset
        elif self.offset_mode % 10 == 2:
            features_rest_dynamic_offset = self.quat_decoders['features_rest_dynamic_offset'](self._features_rest_dynamic_offset)
            return self._features_rest_dynamic + features_rest_dynamic_offset.reshape(self._features_dc_dynamic.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)
        return self._features_rest_dynamic

    @property
    def get_language_feature(self):
        if self.offset_mode == 11:
            return self._language_feature + self._language_feature_offset
        elif self.offset_mode == 22:
            return self._language_feature + self.quat_decoders['language_feature_offset'](self._language_feature_offset)
        return self._language_feature

    @property
    def get_language_feature_dynamic(self):
        if self.offset_mode == 11:
            return self._language_feature_dynamic + self._language_feature_dynamic_offset
        elif self.offset_mode == 22:
            return self._language_feature_dynamic + self.quat_decoders['language_feature_dynamic_offset'](self._language_feature_dynamic_offset)
        return self._language_feature_dynamic

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_lang_mlp(self):
        return self.mlp_lang

    @property
    def get_time_pe(self):
        return self.time_pe_fn

    @property
    def get_xyz_pe(self):
        return self.xyz_pe_fn

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], self.opacity_dim), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda") * 1000

        self.time_line = time_line
        self._importance = torch.zeros((self._xyz.shape[0]), device="cuda")

        language_feature = torch.zeros((self._xyz.shape[0], self.lang_feat_dim), device="cuda")
        self._language_feature = nn.Parameter(language_feature.requires_grad_(True))

    def create_from_coarse(self, checkpoint, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = checkpoint['_xyz'].float().cuda()
        features_dc = checkpoint['_features_dc'].float().cuda()
        features_rest = checkpoint['_features_rest'].float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        scales = checkpoint['_scaling'].float().cuda()
        rots = checkpoint['_rotation'].float().cuda()
        opacities = checkpoint['_opacity'].float().cuda()
        self._dynamic = checkpoint['_dynamic'].float().cuda()
        dynamic_mask = (self._dynamic > 5.0).squeeze(1)

        if '_language_feature' in checkpoint.keys():
            language_feature = checkpoint['_language_feature'].float().cuda()
            self._language_feature = nn.Parameter(language_feature[~dynamic_mask].requires_grad_(True))
            self._language_feature_dynamic = nn.Parameter(language_feature[dynamic_mask].requires_grad_(True))

        self._xyz = nn.Parameter(fused_point_cloud[~dynamic_mask].requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc[~dynamic_mask].requires_grad_(True))

        # o_features_rest = features_rest[~dynamic_mask]
        # if o_features_rest.shape[1] < ((self.max_sh_degree + 1) ** 2) - 1:
        #     padding_tensor = torch.zeros(o_features_rest.shape[0], ((self.max_sh_degree + 1) ** 2) - 1 - o_features_rest.shape[1], o_features_rest.shape[2])
        #     o_features_rest = torch.cat((o_features_rest, padding_tensor.cuda()), dim=1)
        # self._features_rest = nn.Parameter(o_features_rest.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest[~dynamic_mask].requires_grad_(True))

        self._scaling = nn.Parameter(scales[~dynamic_mask].requires_grad_(True))
        self._rotation = nn.Parameter(rots[~dynamic_mask].requires_grad_(True))
        self._opacity = nn.Parameter(opacities[~dynamic_mask].requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda") * 1000
        self._importance = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.time_line = time_line

        self._xyz_dynamic = nn.Parameter(fused_point_cloud[dynamic_mask].requires_grad_(True))
        self._features_dc_dynamic = nn.Parameter(features_dc[dynamic_mask].requires_grad_(True))

        # o_features_rest_dynamic = features_rest[dynamic_mask]
        # if o_features_rest_dynamic.shape[1] < ((self.max_sh_degree + 1) ** 2) - 1:
        #     padding_tensor = torch.zeros(o_features_rest_dynamic.shape[0], ((self.max_sh_degree + 1) ** 2) - 1 - o_features_rest_dynamic.shape[1], o_features_rest_dynamic.shape[2])
        #     o_features_rest_dynamic = torch.cat((o_features_rest_dynamic, padding_tensor.cuda()), dim=1)
        # self._features_rest_dynamic = nn.Parameter(o_features_rest_dynamic.requires_grad_(True))
        self._features_rest_dynamic = nn.Parameter(features_rest[dynamic_mask].requires_grad_(True))

        self._opacity_dynamic = nn.Parameter(opacities[dynamic_mask].requires_grad_(True))
        self._scaling_dynamic = nn.Parameter(scales[dynamic_mask].requires_grad_(True))
        self._rotation_dynamic = nn.Parameter(rots[dynamic_mask].requires_grad_(True))
        self._point_feats = nn.Parameter(torch.zeros((self._xyz_dynamic.shape[0], self.keyframe_num, self.feat_dim), device="cuda").requires_grad_(True))

        self.opacity_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_xyz_gradient_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_denom = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_max_radii2D = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")
        self.dynamic_min_radii2D = torch.ones((self._xyz_dynamic.shape[0]), device="cuda") * 1000
        self._importance_dynamic = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")

    def create_from_ckpt(self, checkpoint, spatial_lr_scale: float, time_line: int, preserve=0.2):
        self.spatial_lr_scale = spatial_lr_scale
        self.time_line = time_line

        if '_xyz_offset' not in checkpoint.keys():
            fused_point_cloud = checkpoint['_xyz'].float().cuda()
            print("Number of static gaussians from previous ckpt : ", fused_point_cloud.shape[0])
            features_dc = checkpoint['_features_dc'].float().cuda()
            features_rest = checkpoint['_features_rest'].float().cuda()
            scales = checkpoint['_scaling'].float().cuda()
            rots = checkpoint['_rotation'].float().cuda()
            opacities = checkpoint['_opacity'].float().cuda()
            if '_language_feature' in checkpoint.keys():
                language_feature = checkpoint['_language_feature'].float().cuda()
        else:
            fused_point_cloud = checkpoint['_xyz_offset'].float().cuda()
            print("Number of static gaussians from previous ckpt : ", fused_point_cloud.shape[0])
            features_dc = checkpoint['_features_dc_offset'].float().cuda()
            features_rest = checkpoint['_features_rest_offset'].float().cuda()
            scales = checkpoint['_scaling_offset'].float().cuda()
            rots = checkpoint['_rotation_offset'].float().cuda()
            opacities = checkpoint['_opacity_offset'].float().cuda()
            if '_language_feature' in checkpoint.keys():
                language_feature = checkpoint['_language_feature_offset'].float().cuda()

        self._xyz = nn.Parameter(copy.deepcopy(fused_point_cloud)).requires_grad_(False)
        self._features_dc = nn.Parameter(features_dc).requires_grad_(False)
        self._features_rest = nn.Parameter(features_rest).requires_grad_(False)
        self._scaling = nn.Parameter(scales).requires_grad_(False)
        self._rotation = nn.Parameter(rots).requires_grad_(False)
        self._opacity = nn.Parameter(opacities).requires_grad_(False)
        if '_language_feature' in checkpoint.keys():
            self._language_feature = nn.Parameter(language_feature).requires_grad_(False)

        if preserve == 2:
            self._xyz_offset = nn.Parameter(fused_point_cloud.requires_grad_(True))

            self._features_dc_offset = nn.Parameter(torch.zeros_like(self._features_dc).requires_grad_(True))
            self._features_rest_offset = nn.Parameter(torch.zeros_like(self._features_rest).requires_grad_(True))
            self._scaling_offset = nn.Parameter(torch.zeros_like(self._scaling)).requires_grad_(True)
            self._rotation_offset = nn.Parameter(torch.zeros_like(self._rotation)).requires_grad_(True)
            self._opacity_offset = nn.Parameter(torch.zeros_like(self._opacity).requires_grad_(True))
            if '_language_feature' in checkpoint.keys():
                self._language_feature_offset = nn.Parameter(torch.zeros_like(self._language_feature_offset).requires_grad_(True))

            self.offset_mode = 10
        elif preserve == 2.1:
            self._xyz_offset = nn.Parameter(fused_point_cloud.requires_grad_(True))

            # zero = self.quat_decoders["features_dc_offset"].invert(torch.zeros_like(self._features_dc).flatten(start_dim=1).contiguous().cuda())
            zero = torch.ones((self._xyz.shape[0], self.quat_decoders["features_dc_offset"].latent_dim)).to('cuda')
            self._features_dc_offset = nn.Parameter(zero.requires_grad_(True))
            # zero = self.quat_decoders["scaling_offset"].invert(torch.zeros_like(self._scaling).cuda())
            zero = torch.ones((self._xyz.shape[0], self.quat_decoders["scaling_offset"].latent_dim)).to('cuda')
            self._scaling_offset = nn.Parameter(zero.requires_grad_(True))

            # zero = self.quat_decoders["rotation_offset"].invert(torch.zeros_like(self._rotation).cuda())
            zero = torch.ones((self._xyz.shape[0], self.quat_decoders["rotation_offset"].latent_dim)).to('cuda')
            self._rotation_offset = nn.Parameter(zero.requires_grad_(True))
            # self._rotation_offset = nn.Parameter(torch.zeros_like(self._rotation).requires_grad_(True))
            # zero = self.quat_decoders["opacity_offset"].invert(torch.zeros_like(self._opacity).cuda())
            zero = torch.ones((self._xyz.shape[0], self.quat_decoders["opacity_offset"].latent_dim)).to('cuda') * 1.5
            self._opacity_offset = nn.Parameter(zero.requires_grad_(True))
            # zero = torch.zeros((self._xyz.shape[0], self.quat_decoders["features_rest_offset"].latent_dim)).to('cuda').contiguous()
            zero = torch.ones((self._xyz.shape[0], self.quat_decoders["features_rest_offset"].latent_dim)).to('cuda').contiguous()
            self._features_rest_offset = nn.Parameter(zero.requires_grad_(True))
            if '_language_feature' in checkpoint.keys():
                zero = torch.ones((self._xyz.shape[0], self.quat_decoders["language_feature_offset"].latent_dim)).to('cuda').contiguous()
                self._language_feature_offset = nn.Parameter(zero.requires_grad_(True))
            self.offset_mode = 20

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda") * 1000
        self._importance = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.grid_sidelen = int(np.sqrt(fused_point_cloud.shape[0]))

        if '_xyz_dynamic_offset' not in checkpoint.keys():
            dynamic_point_cloud = checkpoint['_xyz_dynamic'].float().cuda()
            print("Number of dynamic gaussians from previous ckpt : ", dynamic_point_cloud.shape[0])
            dynamic_features_dc = checkpoint['_features_dc_dynamic'].float().cuda()
            dynamic_features_rest = checkpoint['_features_rest_dynamic'].float().cuda()
            dynamic_scales = checkpoint['_scaling_dynamic'].float().cuda()
            dynamic_rots = checkpoint['_rotation_dynamic'].float().cuda()
            dynamic_opacities = checkpoint['_opacity_dynamic'].float().cuda()
            if '_language_feature' in checkpoint.keys():
                dynamic_language_feature = checkpoint['_language_feature_dynamic'].float().cuda()
        else:
            dynamic_point_cloud = checkpoint['_xyz_dynamic_offset'].float().cuda()
            print("Number of dynamic gaussians from previous ckpt : ", dynamic_point_cloud.shape[0])
            dynamic_features_dc = checkpoint['_features_dc_dynamic_offset'].float().cuda()
            dynamic_features_rest = checkpoint['_features_rest_dynamic_offset'].float().cuda()
            dynamic_scales = checkpoint['_scaling_dynamic_offset'].float().cuda()
            dynamic_rots = checkpoint['_rotation_dynamic_offset'].float().cuda()
            dynamic_opacities = checkpoint['_opacity_dynamic_offset'].float().cuda()
            if '_language_feature' in checkpoint.keys():
                dynamic_language_feature = checkpoint['_language_feature_dynamic_offset'].float().cuda()

        if preserve < 1:
            dynamic_opacity = torch.sigmoid(dynamic_opacities)
            threshold = torch.quantile(dynamic_opacity, preserve)
            mask = (dynamic_opacity > threshold).squeeze(1)
            self._xyz_dynamic = nn.Parameter(dynamic_point_cloud[mask].requires_grad_(True))
            self._features_dc_dynamic = nn.Parameter(dynamic_features_dc[mask].requires_grad_(True))
            self._features_rest_dynamic = nn.Parameter(dynamic_features_rest[mask].requires_grad_(True))
            self._opacity_dynamic = nn.Parameter(dynamic_opacities[mask].requires_grad_(True))
            self._scaling_dynamic = nn.Parameter(dynamic_scales[mask].requires_grad_(True))
            self._rotation_dynamic = nn.Parameter(dynamic_rots[mask].requires_grad_(True))
            if '_language_feature' in checkpoint.keys():
                self._language_feature_dynamic = nn.Parameter(dynamic_language_feature[mask].requires_grad_(True))

            self._previous_last_feat = checkpoint['_point_feats'][:, -1, :][mask].cuda()
            self.dynamic_grid_sidelen = 0
        elif preserve == 1 or preserve == 2:
            self._xyz_dynamic = nn.Parameter(copy.deepcopy(dynamic_point_cloud)).requires_grad_(False)
            self._features_dc_dynamic = nn.Parameter(dynamic_features_dc).requires_grad_(False)
            self._opacity_dynamic = nn.Parameter(dynamic_opacities).requires_grad_(False)
            self._scaling_dynamic = nn.Parameter(dynamic_scales).requires_grad_(False)
            self._rotation_dynamic = nn.Parameter(dynamic_rots).requires_grad_(False)

            self._features_rest_dynamic = nn.Parameter(dynamic_features_rest).requires_grad_(False)
            if '_language_feature' in checkpoint.keys():
                self._language_feature_dynamic = nn.Parameter(dynamic_language_feature.requires_grad_(True))
            self._previous_last_feat = checkpoint['_point_feats'][:, -1, :].cuda()
            self.dynamic_grid_sidelen = int(np.sqrt(self._xyz_dynamic.shape[0]))

            self.offset_mode += 1
            self._xyz_dynamic_offset = nn.Parameter(dynamic_point_cloud.requires_grad_(True))
            self._features_dc_dynamic_offset = nn.Parameter(torch.zeros_like(self._features_dc_dynamic).requires_grad_(True))
            self._features_rest_dynamic_offset = nn.Parameter(torch.zeros_like(self._features_rest_dynamic).requires_grad_(True))
            self._opacity_dynamic_offset = nn.Parameter(torch.zeros_like(self._opacity_dynamic).requires_grad_(True))
            self._scaling_dynamic_offset = nn.Parameter(torch.zeros_like(self._scaling_dynamic).requires_grad_(True))
            self._rotation_dynamic_offset = nn.Parameter(torch.zeros_like(self._rotation_dynamic).requires_grad_(True))
            if '_language_feature' in checkpoint.keys():
                self._language_feature_dynamic_offset = nn.Parameter(torch.zeros_like(self._language_feature_dynamic).requires_grad_(True))
        elif preserve == 1.1 or preserve == 2.1:
            self._xyz_dynamic = nn.Parameter(copy.deepcopy(dynamic_point_cloud)).requires_grad_(False)
            self._features_dc_dynamic = nn.Parameter(dynamic_features_dc).requires_grad_(False)
            self._opacity_dynamic = nn.Parameter(dynamic_opacities).requires_grad_(False)
            self._scaling_dynamic = nn.Parameter(dynamic_scales).requires_grad_(False)
            self._rotation_dynamic = nn.Parameter(dynamic_rots).requires_grad_(False)

            self._features_rest_dynamic = nn.Parameter(dynamic_features_rest).requires_grad_(False)
            if '_language_feature' in checkpoint.keys():
                self._language_feature_dynamic = nn.Parameter(dynamic_language_feature.requires_grad_(False))
            self._previous_last_feat = checkpoint['_point_feats'][:, -1, :].cuda()
            self.dynamic_grid_sidelen = int(np.sqrt(self._xyz_dynamic.shape[0]))

            self.offset_mode += 2
            self._xyz_dynamic_offset = nn.Parameter(dynamic_point_cloud.requires_grad_(True))

            # zero = self.quat_decoders["features_dc_dynamic_offset"].invert(torch.zeros_like(self._features_dc_dynamic).flatten(start_dim=1).contiguous().cuda())
            zero = torch.ones((self._xyz_dynamic.shape[0], self.quat_decoders["features_dc_dynamic_offset"].latent_dim)).to('cuda')
            self._features_dc_dynamic_offset = nn.Parameter(zero.requires_grad_(True))
            # zero = self.quat_decoders["opacity_dynamic_offset"].invert(torch.zeros_like(self._opacity_dynamic).cuda())
            zero = torch.ones((self._xyz_dynamic.shape[0], self.quat_decoders["opacity_dynamic_offset"].latent_dim)).to('cuda')
            self._opacity_dynamic_offset = nn.Parameter(zero.requires_grad_(True))
            # zero = self.quat_decoders["scaling_dynamic_offset"].invert(torch.zeros_like(self._scaling_dynamic).cuda())
            zero = torch.ones((self._xyz_dynamic.shape[0], self.quat_decoders["scaling_dynamic_offset"].latent_dim)).to('cuda')
            self._scaling_dynamic_offset = nn.Parameter(zero.requires_grad_(True))
            # zero = self.quat_decoders["rotation_dynamic_offset"].invert(torch.zeros_like(self._rotation_dynamic).cuda())
            zero = torch.ones((self._xyz_dynamic.shape[0], self.quat_decoders["rotation_dynamic_offset"].latent_dim)).to('cuda')
            self._rotation_dynamic_offset = nn.Parameter(zero.requires_grad_(True))
            # self._rotation_dynamic_offset = nn.Parameter(torch.zeros_like(self._rotation_dynamic).requires_grad_(True))

            # zero = torch.zeros((self._xyz_dynamic.shape[0], self.quat_decoders["features_rest_dynamic_offset"].latent_dim)).to('cuda').contiguous()
            zero = torch.ones((self._xyz_dynamic.shape[0], self.quat_decoders["features_rest_dynamic_offset"].latent_dim)).to('cuda').contiguous()
            self._features_rest_dynamic_offset = nn.Parameter(zero.requires_grad_(True))
            if '_language_feature' in checkpoint.keys():
                zero = torch.ones((self._xyz_dynamic.shape[0], self.quat_decoders["language_feature_dynamic_offset"].latent_dim)).to('cuda').contiguous()
                self._language_feature_dynamic_offset = nn.Parameter(zero.requires_grad_(True))

        self._point_feats = nn.Parameter(torch.zeros((self._xyz_dynamic.shape[0], self.keyframe_num, self.feat_dim), device="cuda").requires_grad_(True))

        self.opacity_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_xyz_gradient_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_denom = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_max_radii2D = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")
        self.dynamic_min_radii2D = torch.ones((self._xyz_dynamic.shape[0]), device="cuda") * 1000
        self._importance_dynamic = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")

    def create_from_ckpt_lang(self, checkpoint, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        self.time_line = time_line

        # static gaussian
        fused_point_cloud = checkpoint['_xyz'].float().cuda()
        print("Number of static gaussians from previous ckpt : ", fused_point_cloud.shape[0])
        features_dc = checkpoint['_features_dc'].float().cuda()
        features_rest = checkpoint['_features_rest'].float().cuda()
        scales = checkpoint['_scaling'].float().cuda()
        rots = checkpoint['_rotation'].float().cuda()
        opacities = checkpoint['_opacity'].float().cuda()

        self._xyz = nn.Parameter(copy.deepcopy(fused_point_cloud)).requires_grad_(False)
        self._features_dc = nn.Parameter(features_dc).requires_grad_(False)
        self._features_rest = nn.Parameter(features_rest).requires_grad_(False)
        self._scaling = nn.Parameter(scales).requires_grad_(False)
        self._rotation = nn.Parameter(rots).requires_grad_(False)
        self._opacity = nn.Parameter(opacities).requires_grad_(False)
        if self.offset_mode == 22:
            xyz_offset = checkpoint['_xyz_offset'].float().cuda()
            self._xyz_offset = nn.Parameter(xyz_offset.requires_grad_(False))
            zero = torch.ones((self._xyz.shape[0], self.quat_decoders["language_feature_offset"].latent_dim)).to('cuda').contiguous()
            self._language_feature_offset = nn.Parameter(zero.requires_grad_(True))
        elif self.offset_mode == 0:
            language_feature = checkpoint['_language_feature'].float().cuda()
            self._language_feature = nn.Parameter(language_feature).requires_grad_(True)
        else:
            raise NotImplementedError

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda") * 1000
        self._importance = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.grid_sidelen = int(np.sqrt(fused_point_cloud.shape[0]))

        # dynamic gaussian
        dynamic_point_cloud = checkpoint['_xyz_dynamic'].float().cuda()
        print("Number of dynamic gaussians from previous ckpt : ", dynamic_point_cloud.shape[0])
        dynamic_features_dc = checkpoint['_features_dc_dynamic'].float().cuda()
        dynamic_features_rest = checkpoint['_features_rest_dynamic'].float().cuda()
        dynamic_scales = checkpoint['_scaling_dynamic'].float().cuda()
        dynamic_rots = checkpoint['_rotation_dynamic'].float().cuda()
        dynamic_opacities = checkpoint['_opacity_dynamic'].float().cuda()

        self._xyz_dynamic = nn.Parameter(copy.deepcopy(dynamic_point_cloud)).requires_grad_(False)
        self._features_dc_dynamic = nn.Parameter(dynamic_features_dc).requires_grad_(False)
        self._features_rest_dynamic = nn.Parameter(dynamic_features_rest).requires_grad_(False)
        self._opacity_dynamic = nn.Parameter(dynamic_opacities).requires_grad_(False)
        self._scaling_dynamic = nn.Parameter(dynamic_scales).requires_grad_(False)
        self._rotation_dynamic = nn.Parameter(dynamic_rots).requires_grad_(False)

        if self.offset_mode == 22:
            xyz_dynamic_offset = checkpoint['_xyz_dynamic_offset'].float().cuda()
            self._xyz_dynamic_offset = nn.Parameter(xyz_dynamic_offset.requires_grad_(False))
            zero = torch.ones((self._xyz_dynamic.shape[0], self.quat_decoders["language_feature_dynamic_offset"].latent_dim)).to('cuda').contiguous()
            self._language_feature_dynamic_offset = nn.Parameter(zero.requires_grad_(True))
        elif self.offset_mode == 0:
            dynamic_language_feature = checkpoint['_language_feature_dynamic'].float().cuda()
            self._language_feature_dynamic = nn.Parameter(dynamic_language_feature.requires_grad_(True))
        else:
            raise NotImplementedError

        self._previous_last_feat = checkpoint['_point_feats'][:, -1, :].cuda()
        self.dynamic_grid_sidelen = int(np.sqrt(self._xyz_dynamic.shape[0]))

        # self._point_feats = nn.Parameter(checkpoint['_point_feats'].requires_grad_(False)).cuda()
        self._point_feats = checkpoint['_point_feats'].cuda()

        self.opacity_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_xyz_gradient_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_denom = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_max_radii2D = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")
        self.dynamic_min_radii2D = torch.ones((self._xyz_dynamic.shape[0]), device="cuda") * 1000
        self._importance_dynamic = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        self.dynamic_xyz_gradient_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_denom = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")

        l = [
            {'params': [self._point_feats], 'lr': training_args.temporal_feature_lr_init, "name": f"dynamic_feat"},

            {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_lang.parameters(), 'lr': training_args.mlp_lang_lr_init, "name": "mlp_lang"}
        ]

        if self.offset_mode == 0:
            new = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            ]
            l.extend(new)

            if self._language_feature is not None:
                l.extend([{'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"}])

            # for param in self.lang_entropy_model.parameters():
            #     l.append({'params': [param], 'lr': 1e-3, "name": "mlp_lang_entropy_model"})

        if self.offset_mode <= 0:
            new = [
                {'params': [self._xyz_dynamic], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "dynamic_xyz"},
                {'params': [self._features_dc_dynamic], 'lr': training_args.feature_lr, "name": "dynamic_f_dc"},
                {'params': [self._features_rest_dynamic], 'lr': training_args.feature_lr / 20.0, "name": "dynamic_f_rest"},
                {'params': [self._opacity_dynamic], 'lr': training_args.opacity_lr, "name": "dynamic_opacity"},
                {'params': [self._scaling_dynamic], 'lr': training_args.scaling_lr, "name": "dynamic_scaling"},
                {'params': [self._rotation_dynamic], 'lr': training_args.rotation_lr, "name": "dynamic_rotation"}
            ]
            l.extend(new)

            if self._language_feature_dynamic is not None:
                l.extend([{'params': [self._language_feature_dynamic], 'lr': training_args.language_feature_lr, "name": "dynamic_language_feature"}])

        if self.offset_mode >= 1:
            new = [
                {'params': [self._xyz_dynamic_offset], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "dynamic_xyz_offset"},
                {'params': [self._features_dc_dynamic_offset], 'lr': 0.01, "name": "dynamic_f_dc_offset"},  # 0.005
                {'params': [self._features_rest_dynamic_offset], 'lr': 0.001, "name": "dynamic_f_rest_offset"},
                {'params': [self._opacity_dynamic_offset], 'lr': 0.005, "name": "dynamic_opacity_offset"},
                {'params': [self._scaling_dynamic_offset], 'lr': 0.025, "name": "dynamic_scaling_offset"},
                {'params': [self._rotation_dynamic_offset], 'lr': 0.025, "name": "dynamic_rotation_offset"},
                {'params': [self._language_feature_dynamic_offset], 'lr': 0.002, "name": "dynamic_language_feature_offset"},

                {'params': self.quat_decoders["features_dc_dynamic_offset"].parameters(), 'lr': 0.001, "name": "dynamic_f_dc_offset_probmlp"},
                {'params': self.quat_decoders["features_rest_dynamic_offset"].parameters(), 'lr': 0.001, "name": "dynamic_f_rest_offset_probmlp"},
                {'params': self.quat_decoders["opacity_dynamic_offset"].parameters(), 'lr': 0.002, "name": "dynamic_opacity_offset_probmlp"},
                {'params': self.quat_decoders["scaling_dynamic_offset"].parameters(), 'lr': 0.002, "name": "dynamic_scaling_offset_probmlp"},
                {'params': self.quat_decoders["rotation_dynamic_offset"].parameters(), 'lr': 0.002, "name": "dynamic_rotation_offset_probmlp"},
                {'params': self.quat_decoders["language_feature_dynamic_offset"].parameters(), 'lr': 0.002, "name": "dynamic_language_feature_offset_probmlp"},
            ]
            l.extend(new)

        if self.offset_mode >= 10:
            new = [
                {'params': [self._xyz_offset], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz_offset"},
                {'params': [self._features_dc_offset], 'lr': training_args.feature_lr, "name": "f_dc_offset"},
                {'params': [self._features_rest_offset], 'lr': training_args.feature_lr / 10, "name": "f_rest_offset"},
                {'params': [self._opacity_offset], 'lr': 0.005, "name": "opa_offset"},
                {'params': [self._scaling_offset], 'lr': 0.001, "name": "sca_offset"},
                {'params': [self._rotation_offset], 'lr': 0.001, "name": "rot_offset"},
                {'params': [self._language_feature_offset], 'lr': 0.001, "name": "language_feature_offset"},

                {'params': self.quat_decoders["features_dc_offset"].parameters(), 'lr': 0.001, "name": "f_dc_offset_probmlp"},
                {'params': self.quat_decoders["features_rest_offset"].parameters(), 'lr': 0.001, "name": "f_rest_offset_probmlp"},
                {'params': self.quat_decoders["opacity_offset"].parameters(), 'lr': 0.001, "name": "opacity_offset_probmlp"},
                {'params': self.quat_decoders["scaling_offset"].parameters(), 'lr': 0.001, "name": "scaling_offset_probmlp"},
                {'params': self.quat_decoders["rotation_offset"].parameters(), 'lr': 0.001, "name": "rotation_offset_probmlp"},
                {'params': self.quat_decoders["language_feature_offset"].parameters(), 'lr': 0.001, "name": "language_feature_offset_probmlp"},
            ]
            l.extend(new)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.lr_scheduler(training_args)

    def lr_scheduler(self, training_args):
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.f_dc_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr,
                                                     lr_final=training_args.feature_lr_final,
                                                     lr_delay_mult=training_args.position_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)

        self.f_rest_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr / 20,
                                                     lr_final=training_args.feature_lr_final / 20 * 0.1,
                                                     lr_delay_mult=training_args.position_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)

        self.opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.opacity_lr,
                                                        lr_final=training_args.opacity_lr_final,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
                                                        lr_final=training_args.scaling_lr_final,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
                                                         lr_final=training_args.rotation_lr_final,
                                                         lr_delay_mult=training_args.position_lr_delay_mult,
                                                         max_steps=training_args.position_lr_max_steps)

        self.temporal_feature_scheduler_args = get_expon_lr_func(lr_init=training_args.temporal_feature_lr_init,
                                                           lr_final=training_args.temporal_feature_lr_final,
                                                           lr_delay_mult=training_args.temporal_feature_lr_delay_mult,
                                                           max_steps=training_args.temporal_feature_lr_steps)

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                           lr_final=training_args.mlp_deform_lr_final,
                                                           lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                           max_steps=training_args.mlp_deform_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr_dict = {}
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "dynamic_feat":
                lr = self.temporal_feature_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr

            elif "xyz" in param_group["name"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_dc" or param_group["name"] == "f_dc_offset":
                lr = self.f_dc_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_rest" or param_group["name"] == "f_rest_offset":
                lr = self.f_rest_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "opacity" or param_group["name"] == "opa_offset_":
                lr = self.opacity_scheduler_args(iteration)
                param_group['lr'] = lr

        return lr_dict

    def create_dynamic(self, points):
        self._dynamic = nn.Parameter(torch.zeros((points.shape[0], 1), device="cuda").requires_grad_(True))

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

    def create_language_learnable_features(self):
        self._lang_temp_feats = nn.Parameter(torch.zeros((self._xyz_dynamic.shape[0], self.keyframe_num, 4), device="cuda").requires_grad_(True))
        # if self.offset_mode == 22:
            # self._language_feature_offset = nn.Parameter(self._language_feature_offset.requires_grad_(True))
            # self._language_feature_dynamic_offset = nn.Parameter(self._language_feature_dynamic_offset.requires_grad_(True))

            # self._features_dc_offset = nn.Parameter(self._features_dc_offset.requires_grad_(False))
            # self._features_rest_offset = nn.Parameter(self._features_rest_offset.requires_grad_(False))
            # self._opacity_offset = nn.Parameter(self._opacity_offset.requires_grad_(False))
            # self._scaling_offset = nn.Parameter(self._scaling_offset.requires_grad_(False))
            # self._rotation_offset = nn.Parameter(self._rotation_offset.requires_grad_(False))
            #
            # self._features_dc_dynamic_offset = nn.Parameter(self._features_dc_dynamic_offset.requires_grad_(False))
            # self._features_rest_dynamic_offset = nn.Parameter(self._features_rest_dynamic_offset.requires_grad_(False))
            # self._opacity_dynamic_offset = nn.Parameter(self._opacity_dynamic_offset.requires_grad_(False))
            # self._scaling_dynamic_offset = nn.Parameter(self._scaling_dynamic_offset.requires_grad_(False))
            # self._rotation_dynamic_offset = nn.Parameter(self._rotation_dynamic_offset.requires_grad_(False))
        # if self.opacity_dim == 2 and self.offset_mode == 0:
        #     # if self.offset_mode == 0:
        #     lang_opacity = self._opacity[:, 0:1].detach().clone()
        #     lang_opacity_dynamic = self._opacity_dynamic[:, 0:1].detach().clone()
        #     self._lang_opacity = nn.Parameter(lang_opacity.requires_grad_(True))
        #     self._lang_opacity_dynamic = nn.Parameter(lang_opacity_dynamic.requires_grad_(True))
        #     # else:
        #     #     lang_opacity = self._opacity_offset[:, 0:1].detach().clone()
        #     #     lang_opacity_dynamic = self._opacity_dynamic_offset[:, 0:1].detach().clone()
        #     #     self._lang_opacity = nn.Parameter(lang_opacity.requires_grad_(True))
        #     #     self._lang_opacity_dynamic = nn.Parameter(lang_opacity_dynamic.requires_grad_(True))

    def get_time_lang_features(self, timestamp, noise=0.):
        t = int(timestamp * self.time_line)
        t += self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval

        t_idx = int(t_idx)
        feat = self._lang_temp_feats[:, t_idx, :]
        feat_next = self._lang_temp_feats[:, t_idx + 1, :]

        return feat, feat_next

    def training_language_setup(self, training_args):
        l = [
            {'params': [self._lang_temp_feats], 'lr': training_args.temporal_feature_lr_init, "name": f"dynamic_feat"},
            {'params': self.mlp_lang.parameters(), 'lr': training_args.mlp_lang_lr_init, "name": "mlp_lang"},
        ]
        if self.offset_mode == 0:
            l.extend([
                {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"},
                {'params': [self._language_feature_dynamic], 'lr': training_args.language_feature_lr, "name": "dynamic_language_feature"}
            ])
        # if self.opacity_dim == 2:
        #     l.extend([
        #         {'params': [self._lang_opacity], 'lr': training_args.opacity_lr if self.offset_mode == 0 else 0.001, "name": "opacity"},
        #         {'params': [self._lang_opacity_dynamic], 'lr': training_args.opacity_lr if self.offset_mode == 0 else 0.001, "name": "dynamic_opacity"},
        #     ])
        if self.offset_mode >= 1:
            l.extend([
                {'params': [self._language_feature_dynamic_offset], 'lr': training_args.language_feature_lr, "name": "dynamic_language_feature_offset"},
                {'params': self.quat_decoders["language_feature_dynamic_offset"].parameters(), 'lr': training_args.language_feature_lr, "name": "dynamic_language_feature_offset_probmlp"},
            ])
        if self.offset_mode >= 10:
            l.extend([
                {'params': [self._language_feature_offset], 'lr': training_args.language_feature_lr, "name": "language_feature_offset"},
                {'params': self.quat_decoders["language_feature_offset"].parameters(), 'lr': training_args.language_feature_lr, "name": "language_feature_offset_probmlp"},
            ])

        self.lang_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def save(self, path, name='model.pth', save_all=False, save_geometry_only=False):
        if save_all:
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            new_state_dict = self.state_dict()

            torch.save(new_state_dict, os.path.join(path, name))
            if self.offset_mode >= 1:
                new_state_dict['_xyz_dynamic_offset'] = self._xyz_dynamic_offset.detach()
                new_state_dict['_features_dc_dynamic_offset'] = self.get_feature_dc_dynamic_ori.detach()
                new_state_dict['_features_rest_dynamic_offset'] = self.get_feature_rest_dynamic_ori.detach()
                new_state_dict['_opacity_dynamic_offset'] = self.get_opacity_dynamic_ori.detach()
                new_state_dict['_scaling_dynamic_offset'] = self.get_scaling_dynamic_ori.detach()
                new_state_dict['_rotation_dynamic_offset'] = self.get_rotation_dynamic_ori.detach()
                new_state_dict['_language_feature_dynamic_offset'] = self.get_language_feature_dynamic.detach()
                if self.offset_mode >= 10:
                    new_state_dict['_xyz_offset'] = self._xyz_offset.detach()
                    new_state_dict['_features_dc_offset'] = self.get_feature_dc_ori.detach()
                    new_state_dict['_features_rest_offset'] = self.get_feature_rest_ori.detach()
                    new_state_dict['_opacity_offset'] = self.get_opacity_ori.detach()
                    new_state_dict['_scaling_offset'] = self.get_scaling_ori.detach()
                    new_state_dict['_rotation_offset'] = self.get_rotation_ori.detach()
                    new_state_dict['_language_feature_offset'] = self.get_language_feature.detach()
                new_state_dict['offset_mode'] = self.offset_mode
                torch.save(new_state_dict, os.path.join(path, 'model_offset.pth'))

    def save_mlps(self, path, name=''):
        torch.save(self.mlp_lang.state_dict(), os.path.join(path, name + "mlp_lang.pth"))
        torch.save(self.mlp_deform.state_dict(), os.path.join(path, name + "mlp_deform.pth"))
        torch.save(self.mlp_cov.state_dict(), os.path.join(path, name + "mlp_cov.pth"))
        torch.save(self.mlp_opacity.state_dict(), os.path.join(path, name + "mlp_opacity.pth"))
        torch.save(self.mlp_color.state_dict(), os.path.join(path, name + "mlp_color.pth"))

    # # # densify
    def replace_tensor_to_optimizer(self, tensor_dict):
        optimizable_tensors = {}
        for name, tensor in tensor_dict.items():
            for group in self.optimizer.param_groups:
                if group["name"] == name:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                        del self.optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                        self.optimizer.state[group['params'][0]] = stored_state

                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(group["params"][0].requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, static_mask, dynamic_mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'encoding' in group['name']:
                continue
            if len(group["params"]) > 1:
                continue

            if group["name"].startswith("dynamic"):
                mask = dynamic_mask
            else:
                mask = static_mask
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_static_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'encoding' in group['name']:
                continue
            if len(group["params"]) > 1:
                continue

            if not group["name"].startswith("dynamic"):
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_dynamic_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'encoding' in group['name']:
                continue
            if len(group["params"]) > 1:
                continue

            if group["name"].startswith("dynamic"):

                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'encoding' in group['name']:
                continue

            if len(group["params"]) > 1:
                continue
            assert len(group["params"]) == 1

            if not group["name"] in tensors_dict.keys():
                # print(f"attr {group['name']} not in {tensors_dict.keys()}")
                continue

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            try:
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
            except RuntimeError:
                print(f"attr name = {group['name']}")
                break

        return optimizable_tensors

    def prune_points(self, static_mask, dynamic_mask):
        valid_points_mask = ~static_mask

        if dynamic_mask.shape[0] == 0:
            dynamic_mask = torch.empty(0, dtype=torch.bool, device=static_mask.device)
        valid_dynamic_mask = ~dynamic_mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask, valid_dynamic_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self._language_feature is not None:
            self._language_feature = optimizable_tensors["language_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.min_radii2D = self.min_radii2D[valid_points_mask]
        self._importance = self._importance[valid_points_mask]

        if valid_dynamic_mask.shape[0] == 0:
            return

        self._xyz_dynamic = optimizable_tensors["dynamic_xyz"]
        self._features_dc_dynamic = optimizable_tensors["dynamic_f_dc"]
        self._features_rest_dynamic = optimizable_tensors["dynamic_f_rest"]
        self._opacity_dynamic = optimizable_tensors["dynamic_opacity"]
        self._scaling_dynamic = optimizable_tensors["dynamic_scaling"]
        self._rotation_dynamic = optimizable_tensors["dynamic_rotation"]
        self._point_feats = optimizable_tensors[f"dynamic_feat"]

        if self._language_feature_dynamic is not None:
            self._language_feature_dynamic = optimizable_tensors["dynamic_language_feature"]

        self.opacity_accum = self.opacity_accum[valid_dynamic_mask]
        self.dynamic_xyz_gradient_accum = self.dynamic_xyz_gradient_accum[valid_dynamic_mask]
        self.dynamic_denom = self.dynamic_denom[valid_dynamic_mask]
        self.dynamic_max_radii2D = self.dynamic_max_radii2D[valid_dynamic_mask]
        self.dynamic_min_radii2D = self.dynamic_min_radii2D[valid_dynamic_mask]

    def prune_static_points(self, static_mask):
        valid_points_mask = ~static_mask

        optimizable_tensors = self._prune_static_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self._language_feature is not None:
            self._language_feature = optimizable_tensors["language_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.min_radii2D = self.min_radii2D[valid_points_mask]

        self._importance = self._importance[valid_points_mask]

    def prune_dynamic_points(self, dynamic_mask):
        if dynamic_mask.shape[0] == 0:
            dynamic_mask = torch.empty(0, dtype=torch.bool, device="cuda")
        valid_dynamic_mask = ~dynamic_mask

        optimizable_tensors = self._prune_dynamic_optimizer(valid_dynamic_mask)

        self._xyz_dynamic = optimizable_tensors["dynamic_xyz"]
        self._features_dc_dynamic = optimizable_tensors["dynamic_f_dc"]
        self._features_rest_dynamic = optimizable_tensors["dynamic_f_rest"]
        self._opacity_dynamic = optimizable_tensors["dynamic_opacity"]
        self._scaling_dynamic = optimizable_tensors["dynamic_scaling"]
        self._rotation_dynamic = optimizable_tensors["dynamic_rotation"]
        self._point_feats = optimizable_tensors[f"dynamic_feat"]

        if self._language_feature_dynamic is not None:
            self._language_feature_dynamic = optimizable_tensors["dynamic_language_feature"]

        self.opacity_accum = self.opacity_accum[valid_dynamic_mask]
        self.dynamic_xyz_gradient_accum = self.dynamic_xyz_gradient_accum[valid_dynamic_mask]
        self.dynamic_denom = self.dynamic_denom[valid_dynamic_mask]
        self.dynamic_max_radii2D = self.dynamic_max_radii2D[valid_dynamic_mask]
        self.dynamic_min_radii2D = self.dynamic_min_radii2D[valid_dynamic_mask]

        self._importance_dynamic = self._importance_dynamic[valid_dynamic_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_language_feature,
                              new_xyz_dynamic, new_features_dc_dynamic, new_features_rest_dynamic, new_opacities_dynamic, new_scaling_dynamic, new_rotation_dynamic, new_dynamic_feats, new_language_feature_dynamic):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,

             "dynamic_xyz": new_xyz_dynamic,
             "dynamic_f_dc": new_features_dc_dynamic,
             "dynamic_f_rest": new_features_rest_dynamic,
             "dynamic_opacity": new_opacities_dynamic,
             "dynamic_scaling": new_scaling_dynamic,
             "dynamic_rotation": new_rotation_dynamic,
             "dynamic_feat": new_dynamic_feats
        }

        if self._language_feature is not None:
            d["language_feature"] = new_language_feature

        if self._language_feature_dynamic is not None:
            d["dynamic_language_feature"] = new_language_feature_dynamic

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self._language_feature is not None:
            self._language_feature = optimizable_tensors["language_feature"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000

        self._xyz_dynamic = optimizable_tensors["dynamic_xyz"]
        self._features_dc_dynamic = optimizable_tensors["dynamic_f_dc"]
        self._features_rest_dynamic = optimizable_tensors["dynamic_f_rest"]
        self._opacity_dynamic = optimizable_tensors["dynamic_opacity"]
        self._scaling_dynamic = optimizable_tensors["dynamic_scaling"]
        self._rotation_dynamic = optimizable_tensors["dynamic_rotation"]
        self._point_feats = optimizable_tensors[f"dynamic_feat"]

        if self._language_feature_dynamic is not None:
            self._language_feature_dynamic = optimizable_tensors["dynamic_language_feature"]

        self.opacity_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_xyz_gradient_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_denom = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_max_radii2D = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")
        self.dynamic_min_radii2D = torch.ones((self._xyz_dynamic.shape[0]), device="cuda") * 1000

    def densification_postfix_static_only(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_language_feature, new_importance):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             }

        if self._language_feature is not None:
            d["language_feature"] = new_language_feature

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self._language_feature is not None:
            self._language_feature = optimizable_tensors["language_feature"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000

        self._importance = torch.cat([self._importance, new_importance], dim=0)

    def densification_postfix_dynamic_only(self, new_xyz_dynamic, new_features_dc_dynamic, new_features_rest_dynamic, new_opacities_dynamic,
                                           new_scaling_dynamic, new_rotation_dynamic, new_dynamic_feats, new_importance_dynamic, new_language_feature_dynamic):
        d = {
             "dynamic_xyz": new_xyz_dynamic,
             "dynamic_f_dc": new_features_dc_dynamic,
             "dynamic_f_rest": new_features_rest_dynamic,
             "dynamic_opacity": new_opacities_dynamic,
             "dynamic_scaling": new_scaling_dynamic,
             "dynamic_rotation": new_rotation_dynamic,
             "dynamic_feat": new_dynamic_feats
             }

        if self._language_feature_dynamic is not None:
            d["dynamic_language_feature"] = new_language_feature_dynamic

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz_dynamic = optimizable_tensors["dynamic_xyz"]
        self._features_dc_dynamic = optimizable_tensors["dynamic_f_dc"]
        self._features_rest_dynamic = optimizable_tensors["dynamic_f_rest"]
        self._opacity_dynamic = optimizable_tensors["dynamic_opacity"]
        self._scaling_dynamic = optimizable_tensors["dynamic_scaling"]
        self._rotation_dynamic = optimizable_tensors["dynamic_rotation"]
        self._point_feats = optimizable_tensors[f"dynamic_feat"]

        if self._language_feature_dynamic is not None:
            self._language_feature_dynamic = optimizable_tensors["dynamic_language_feature"]

        self.opacity_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_xyz_gradient_accum = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_denom = torch.zeros((self._xyz_dynamic.shape[0], 1), device="cuda")
        self.dynamic_max_radii2D = torch.zeros((self._xyz_dynamic.shape[0]), device="cuda")
        self.dynamic_min_radii2D = torch.ones((self._xyz_dynamic.shape[0]), device="cuda") * 1000
        self._importance_dynamic = torch.cat([self._importance_dynamic, new_importance_dynamic], dim=0)

    def densify_and_clone(self, grads, dynamic_grads, grad_threshold, grad_dynamic_threshold, scene_extent):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_importance = self._importance[selected_pts_mask]

        new_language_feature = None
        if self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask]

        if self._xyz_dynamic.shape[0] == 0:
            self.densification_postfix_static_only(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_language_feature, new_importance)
            return

        selected_dynamic_pts_mask = torch.where(torch.norm(dynamic_grads, dim=-1) >= grad_dynamic_threshold, True, False)
        selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                                      torch.max(self.get_scaling_dynamic, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz_dynamic = self._xyz_dynamic[selected_dynamic_pts_mask]
        new_features_dc_dynamic = self._features_dc_dynamic[selected_dynamic_pts_mask]
        new_features_rest_dynamic = self._features_rest_dynamic[selected_dynamic_pts_mask]
        new_opacity_dynamic = self._opacity_dynamic[selected_dynamic_pts_mask]
        new_scaling_dynamic = self._scaling_dynamic[selected_dynamic_pts_mask]
        new_rotation_dynamic = self._rotation_dynamic[selected_dynamic_pts_mask]
        new_point_feats = self._point_feats[selected_dynamic_pts_mask]

        new_language_feature_dynamic = None
        if self._language_feature_dynamic is not None:
            new_language_feature_dynamic = self._language_feature_dynamic[selected_dynamic_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_language_feature,
                                   new_xyz_dynamic, new_features_dc_dynamic, new_features_rest_dynamic, new_opacity_dynamic, new_scaling_dynamic, new_rotation_dynamic,new_point_feats, new_language_feature_dynamic)

    def densify_and_clone_static(self, grads, dynamic_grads, grad_threshold, grad_dynamic_threshold, scene_extent):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_language_feature = None
        if self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask]

        new_importance = self._importance[selected_pts_mask]

        self.densification_postfix_static_only(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                                   new_language_feature, new_importance)

    def densify_and_clone_dynamic(self, grads, dynamic_grads, grad_threshold, grad_dynamic_threshold, scene_extent):
        selected_dynamic_pts_mask = torch.where(torch.norm(dynamic_grads, dim=-1) >= grad_dynamic_threshold, True, False)
        selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                                      torch.max(self.get_scaling_dynamic, dim=1).values <= self.percent_dense * scene_extent)
        new_xyz_dynamic = self._xyz_dynamic[selected_dynamic_pts_mask]
        new_features_dc_dynamic = self._features_dc_dynamic[selected_dynamic_pts_mask]
        new_features_rest_dynamic = self._features_rest_dynamic[selected_dynamic_pts_mask]
        new_opacity_dynamic = self._opacity_dynamic[selected_dynamic_pts_mask]
        new_scaling_dynamic = self._scaling_dynamic[selected_dynamic_pts_mask]
        new_rotation_dynamic = self._rotation_dynamic[selected_dynamic_pts_mask]
        new_point_feats = self._point_feats[selected_dynamic_pts_mask]

        new_language_feature_dynamic = None
        if self._language_feature_dynamic is not None:
            new_language_feature_dynamic = self._language_feature_dynamic[selected_dynamic_pts_mask]

        new_importance_dynamic = self._importance_dynamic[selected_dynamic_pts_mask]

        self.densification_postfix_dynamic_only(new_xyz_dynamic, new_features_dc_dynamic, new_features_rest_dynamic, new_opacity_dynamic,
                                                new_scaling_dynamic, new_rotation_dynamic, new_point_feats, new_importance_dynamic, new_language_feature_dynamic)

    def densify_and_split(self, grads, dynamic_grads, grad_threshold, grad_dynamic_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        candidate_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, candidate_mask)

        num_gaussians = n_init_points + torch.sum(selected_pts_mask).detach().cpu()
        self.grid_sidelen = math.ceil(np.sqrt(num_gaussians))

        candidate_global_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]
        candidate_grads = padded_grad[candidate_global_indices]

        sorted_indices = torch.argsort(candidate_grads, descending=True)
        selected_indices = sorted_indices[:self.grid_sidelen*self.grid_sidelen]
        selected_global_indices = candidate_global_indices[selected_indices]
        selected_pts_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool, device="cuda")
        selected_pts_mask[selected_global_indices] = True

        # num_preserved = -1
        # num_removed = -1
        # tmp = 0
        # while num_preserved < 0 or num_removed < 0:
        #     selected_pts_mask = torch.where(padded_grad >= (grad_threshold - tmp * 0.000005), True, False)
        #     selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                           torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        #
        #     num_gaussians = n_init_points + torch.sum(selected_pts_mask).detach().cpu()
        #     if tmp == 0:
        #         self.grid_sidelen = math.ceil(np.sqrt(num_gaussians))
        #     num_removed = num_gaussians - self.grid_sidelen * self.grid_sidelen
        #     num_preserved = torch.sum(selected_pts_mask) - num_removed
        #     tmp += 1
        # assert num_preserved >= 0, "The newly added Gaussian is not enough to be removed"
        #
        # selected_padded_grad = padded_grad[selected_pts_mask]
        # sorted_indices = torch.argsort(selected_padded_grad, descending=True)
        # top_indices = sorted_indices[:num_preserved]
        # selected_global_indices = torch.nonzero(selected_pts_mask, as_tuple=True)[0][top_indices]
        # selected_pts_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool, device="cuda")
        # selected_pts_mask[selected_global_indices] = True

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)  # (N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_importance = self._importance[selected_pts_mask].repeat(N)
        new_language_feature = None
        if self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask].repeat(N, 1)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        if self._xyz_dynamic.shape[0] > 0:
            n_init_points = self._xyz_dynamic.shape[0]
            # Extract points that satisfy the gradient condition
            padded_dynamic_grad = torch.zeros((n_init_points), device="cuda")
            padded_dynamic_grad[:dynamic_grads.shape[0]] = dynamic_grads.squeeze()

            if self._xyz_dynamic.shape[0] > 8000:
                # num_preserved = -1
                # num_removed = -1
                # tmp = 0
                # while num_preserved < 0 or num_removed < 0:
                #     selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= (grad_dynamic_threshold - tmp * 0.000005), True, False)
                #     selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                #                                           torch.max(self.get_scaling_dynamic, dim=1).values > self.percent_dense * scene_extent)
                #
                #     num_gaussians = n_init_points + torch.sum(selected_dynamic_pts_mask).detach().cpu()
                #     if tmp == 0:
                #         self.dynamic_grid_sidelen = math.ceil(np.sqrt(num_gaussians))
                #     num_removed = num_gaussians - self.dynamic_grid_sidelen * self.dynamic_grid_sidelen
                #     num_preserved = torch.sum(selected_dynamic_pts_mask) - num_removed
                #     tmp += 1
                # assert num_preserved >= 0, "The newly added Gaussian is not enough to be removed"
                #
                # selected_padded_dynamic_grad = padded_dynamic_grad[selected_dynamic_pts_mask]
                # sorted_indices = torch.argsort(selected_padded_dynamic_grad, descending=True)
                # top_indices = sorted_indices[:num_preserved]
                # selected_global_indices = torch.nonzero(selected_dynamic_pts_mask, as_tuple=True)[0][top_indices]
                # selected_dynamic_pts_mask = torch.zeros_like(selected_dynamic_pts_mask, dtype=torch.bool, device="cuda")
                # selected_dynamic_pts_mask[selected_global_indices] = True

                candidate_mask = torch.max(self.get_scaling_dynamic, dim=1).values > self.percent_dense * scene_extent
                selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= grad_dynamic_threshold, True, False)
                selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask, candidate_mask)

                num_gaussians = n_init_points + torch.sum(selected_dynamic_pts_mask).detach().cpu()
                self.dynamic_grid_sidelen = math.ceil(np.sqrt(num_gaussians))

                candidate_global_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]
                candidate_grads = padded_dynamic_grad[candidate_global_indices]

                sorted_indices = torch.argsort(candidate_grads, descending=True)
                selected_indices = sorted_indices[:self.dynamic_grid_sidelen * self.dynamic_grid_sidelen]
                selected_global_indices = candidate_global_indices[selected_indices]
                selected_dynamic_pts_mask = torch.zeros_like(selected_dynamic_pts_mask, dtype=torch.bool, device="cuda")
                selected_dynamic_pts_mask[selected_global_indices] = True
            else:
                selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= grad_dynamic_threshold, True, False)
                selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                                              torch.max(self.get_scaling_dynamic, dim=1).values > self.percent_dense * scene_extent)

            stds = self.get_scaling_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation_dynamic[selected_dynamic_pts_mask]).repeat(N, 1, 1)
            new_xyz_dynamic = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
            new_scaling_dynamic = self.scaling_inverse_activation(self.get_scaling_dynamic[selected_dynamic_pts_mask].repeat(N, 1) / (0.8 * N))
            new_rotation_dynamic = self._rotation_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
            new_opacity_dynamic = self._opacity_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
            new_features_dc_dynamic = self._features_dc_dynamic[selected_dynamic_pts_mask].repeat(N, 1, 1)  # (N, 1, 1)
            new_features_rest_dynamic = self._features_rest_dynamic[selected_dynamic_pts_mask].repeat(N, 1, 1)
            new_point_feats = self._point_feats[selected_dynamic_pts_mask].repeat(N, 1, 1)

            new_language_feature_dynamic = None
            if self._language_feature_dynamic is not None:
                new_language_feature_dynamic = self._language_feature_dynamic[selected_dynamic_pts_mask].repeat(N, 1)

            dynamic_prune_filter = torch.cat(
                (selected_dynamic_pts_mask, torch.zeros(N * selected_dynamic_pts_mask.sum(), device="cuda", dtype=bool)))

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_language_feature,
                                       new_xyz_dynamic, new_features_dc_dynamic, new_features_rest_dynamic, new_opacity_dynamic, new_scaling_dynamic, new_rotation_dynamic, new_point_feats, new_language_feature_dynamic)

        else:
            self.densification_postfix_static_only(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_language_feature, new_importance)
            dynamic_prune_filter = torch.empty(0).cuda()

        self.prune_points(prune_filter, dynamic_prune_filter)

    def densify_and_split_static(self, grads, dynamic_grads, grad_threshold, grad_dynamic_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # num_preserved = -1
        # num_removed = -1
        # tmp = 0
        # while num_preserved < 0 or num_removed < 0:
        #     selected_pts_mask = torch.where(padded_grad >= (grad_threshold - tmp * 0.000005), True, False)
        #     selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                           torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        #
        #     num_gaussians = n_init_points + torch.sum(selected_pts_mask).detach().cpu()
        #     if tmp == 0:
        #         self.grid_sidelen = math.ceil(np.sqrt(num_gaussians))
        #     num_removed = num_gaussians - self.grid_sidelen * self.grid_sidelen
        #     num_preserved = torch.sum(selected_pts_mask) - num_removed
        #     tmp += 1
        # assert num_preserved >= 0, "The newly added Gaussian is not enough to be removed"

        # selected_padded_grad = padded_grad[selected_pts_mask]
        # sorted_indices = torch.argsort(selected_padded_grad, descending=True)
        # top_indices = sorted_indices[:num_preserved]
        # selected_global_indices = torch.nonzero(selected_pts_mask, as_tuple=True)[0][top_indices]
        # selected_pts_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool, device="cuda")
        # selected_pts_mask[selected_global_indices] = True

        candidate_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, candidate_mask)

        num_gaussians = n_init_points + torch.sum(selected_pts_mask).detach().cpu()
        self.grid_sidelen = math.ceil(np.sqrt(num_gaussians))

        candidate_global_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]
        candidate_grads = padded_grad[candidate_global_indices]

        sorted_indices = torch.argsort(candidate_grads, descending=True)
        num_split = self.grid_sidelen * self.grid_sidelen - n_init_points
        selected_indices = sorted_indices[:num_split]
        selected_global_indices = candidate_global_indices[selected_indices]
        selected_pts_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool, device="cuda")
        selected_pts_mask[selected_global_indices] = True

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)  # (N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)

        new_language_feature = None
        if self._language_feature is not None:
            new_language_feature = self._language_feature[selected_pts_mask].repeat(N, 1)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        new_importance = self._importance[selected_pts_mask].repeat(N)

        self.densification_postfix_static_only(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_language_feature, new_importance)

        self.prune_static_points(prune_filter)

    def densify_and_split_dynamic(self, grads, dynamic_grads, grad_threshold, grad_dynamic_threshold, scene_extent, N=2):
        n_init_points = self._xyz_dynamic.shape[0]
        # Extract points that satisfy the gradient condition
        padded_dynamic_grad = torch.zeros((n_init_points), device="cuda")
        padded_dynamic_grad[:dynamic_grads.shape[0]] = dynamic_grads.squeeze()

        if n_init_points > 8000:
            # num_preserved = -1
            # num_removed = -1
            # tmp = 0
            # while num_preserved < 0 or num_removed < 0:
            #     selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= (grad_dynamic_threshold - tmp * 0.000005), True, False)
            #     selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
            #                                                   torch.max(self.get_scaling_dynamic, dim=1).values > self.percent_dense * scene_extent)
            #
            #     num_gaussians = n_init_points + torch.sum(selected_dynamic_pts_mask).detach().cpu()
            #     if tmp == 0:
            #         self.dynamic_grid_sidelen = math.ceil(np.sqrt(num_gaussians))
            #     num_removed = num_gaussians - self.dynamic_grid_sidelen * self.dynamic_grid_sidelen
            #     num_preserved = torch.sum(selected_dynamic_pts_mask) - num_removed
            #     tmp += 1
            # assert num_preserved >= 0, "The newly added Gaussian is not enough to be removed"
            #
            # selected_padded_dynamic_grad = padded_dynamic_grad[selected_dynamic_pts_mask]
            # sorted_indices = torch.argsort(selected_padded_dynamic_grad, descending=True)
            # top_indices = sorted_indices[:num_preserved]
            # selected_global_indices = torch.nonzero(selected_dynamic_pts_mask, as_tuple=True)[0][top_indices]
            # selected_dynamic_pts_mask = torch.zeros_like(selected_dynamic_pts_mask, dtype=torch.bool, device="cuda")
            # selected_dynamic_pts_mask[selected_global_indices] = True
            candidate_mask = torch.max(self.get_scaling_dynamic, dim=1).values > self.percent_dense * scene_extent
            selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= grad_dynamic_threshold, True, False)
            selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask, candidate_mask)

            num_gaussians = n_init_points + torch.sum(selected_dynamic_pts_mask).detach().cpu()
            self.dynamic_grid_sidelen = math.ceil(np.sqrt(num_gaussians))

            candidate_global_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]
            candidate_grads = padded_dynamic_grad[candidate_global_indices]

            sorted_indices = torch.argsort(candidate_grads, descending=True)
            # selected_indices = sorted_indices[:self.dynamic_grid_sidelen * self.dynamic_grid_sidelen]
            num_split = self.dynamic_grid_sidelen * self.dynamic_grid_sidelen - n_init_points
            selected_indices = sorted_indices[:num_split]
            selected_global_indices = candidate_global_indices[selected_indices]
            selected_dynamic_pts_mask = torch.zeros_like(selected_dynamic_pts_mask, dtype=torch.bool, device="cuda")
            selected_dynamic_pts_mask[selected_global_indices] = True
        else:
            selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= grad_dynamic_threshold, True, False)
            selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                                          torch.max(self.get_scaling_dynamic, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation_dynamic[selected_dynamic_pts_mask]).repeat(N, 1, 1)
        new_xyz_dynamic = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
        new_scaling_dynamic = self.scaling_inverse_activation(self.get_scaling_dynamic[selected_dynamic_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation_dynamic = self._rotation_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
        new_opacity_dynamic = self._opacity_dynamic[selected_dynamic_pts_mask].repeat(N, 1)
        new_features_dc_dynamic = self._features_dc_dynamic[selected_dynamic_pts_mask].repeat(N, 1, 1)  # (N, 1, 1)
        new_features_rest_dynamic = self._features_rest_dynamic[selected_dynamic_pts_mask].repeat(N, 1, 1)
        new_point_feats = self._point_feats[selected_dynamic_pts_mask].repeat(N, 1, 1)

        new_language_feature_dynamic = None
        if self._language_feature_dynamic is not None:
            new_language_feature_dynamic = self._language_feature_dynamic[selected_dynamic_pts_mask].repeat(N, 1)

        dynamic_prune_filter = torch.cat(
            (selected_dynamic_pts_mask, torch.zeros(N * selected_dynamic_pts_mask.sum(), device="cuda", dtype=bool)))

        new_importance_dynamic = self._importance_dynamic[selected_dynamic_pts_mask].repeat(N)

        self.densification_postfix_dynamic_only(new_xyz_dynamic, new_features_dc_dynamic, new_features_rest_dynamic, new_opacity_dynamic,
                                                new_scaling_dynamic, new_rotation_dynamic, new_point_feats, new_importance_dynamic, new_language_feature_dynamic)

        self.prune_dynamic_points(dynamic_prune_filter)

    def densify_static_only(self, max_grad, max_dgrad, extent, min_opacity=0.01, min_dynamic_opacity=0.01):
        prune_mask = (self.get_colored_opacity < min_opacity).squeeze()
        dynamic_prune_mask = torch.empty(0).cuda()
        self.prune_points(prune_mask, dynamic_prune_mask)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # print("after opacity prune:", self._xyz.shape[0])
        self.densify_and_clone_static(grads, None, max_grad, max_dgrad, extent)
        # print("after clone:", self._xyz.shape[0])
        self.densify_and_split_static(grads, None, max_grad, max_dgrad, extent)
        # print("after split:", self._xyz.shape[0])
        torch.cuda.empty_cache()

    def densify(self, max_grad, max_dgrad, sn, dn, extent, min_opacity=0.01, min_dynamic_opacity=0.01):
        prune_mask = (self.get_colored_opacity < min_opacity).squeeze()

        if self._point_feats.shape[0] == 0:
            dynamic_prune_mask = torch.empty(0).cuda()
        else:
            dynamic_prune_mask = torch.zeros(self._xyz_dynamic.size(0), device=self._xyz_dynamic.device, dtype=torch.bool)

        self.prune_points(prune_mask, dynamic_prune_mask)

        if self._xyz.shape[0] < sn:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            self.densify_and_clone_static(grads, None, max_grad, max_dgrad, extent)
            self.densify_and_split_static(grads, None, max_grad, max_dgrad, extent)

        if self._xyz_dynamic.shape[0] < dn:
            dynamic_grads = self.dynamic_xyz_gradient_accum / self.dynamic_denom
            dynamic_grads[dynamic_grads.isnan()] = 0.0
            self.densify_and_clone_dynamic(None, dynamic_grads, max_grad, max_dgrad, extent)
            self.densify_and_split_dynamic(None, dynamic_grads, max_grad, max_dgrad, extent)

        # self.densify_and_clone(grads, dynamic_grads, max_grad, max_dgrad, extent)
        # self.densify_and_split(grads, dynamic_grads, max_grad, max_dgrad, extent)

        torch.cuda.empty_cache()

    def densify_static(self, max_grad, max_dgrad, sn, dn, extent, min_opacity=0.01, min_dynamic_opacity=0.01):
        if self._xyz.shape[0] < sn:
            prune_mask = (self.get_colored_opacity < min_opacity).squeeze()
            dynamic_prune_mask = torch.zeros(self._xyz_dynamic.size(0), device=self._xyz_dynamic.device, dtype=torch.bool)
            self.prune_points(prune_mask, dynamic_prune_mask)

            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            self.densify_and_clone_static(grads, None, max_grad, max_dgrad, extent)
            self.densify_and_split_static(grads, None, max_grad, max_dgrad, extent)
        else:
            return

        torch.cuda.empty_cache()

    def densify_dynamic(self, max_grad, max_dgrad, sn, dn, extent, min_opacity=0.01, min_dynamic_opacity=0.01):
        if self._xyz_dynamic.shape[0] < dn:
            dynamic_grads = self.dynamic_xyz_gradient_accum / self.dynamic_denom
            dynamic_grads[dynamic_grads.isnan()] = 0.0
            self.densify_and_clone_dynamic(None, dynamic_grads, max_grad, max_dgrad, extent)
            self.densify_and_split_dynamic(None, dynamic_grads, max_grad, max_dgrad, extent)

        torch.cuda.empty_cache()

    def prune_low_importance(self, importance, sn):
        if self._xyz.shape[0] < sn:
            prune_static_mask = self._importance < importance
            self.prune_static_points(prune_static_mask)
            self._importance = self._importance * 0.0
        prune_dynamic_mask = self._importance_dynamic < importance
        self.prune_dynamic_points(prune_dynamic_mask)
        self._importance_dynamic = self._importance_dynamic * 0.0

    def prune_dynamic_low_importance(self, importance):
        prune_dynamic_mask = self._importance_dynamic < importance
        self.prune_dynamic_points(prune_dynamic_mask)
        self._importance_dynamic = self._importance_dynamic * 0.0

    def prune_small(self):
        static_prune_mask = (self.min_radii2D < 1).squeeze()
        dynamic_prune_mask = (self.dynamic_min_radii2D < 1).squeeze()
        self.prune_points(static_prune_mask, dynamic_prune_mask)

        torch.cuda.empty_cache()

    def prune_nan_points(self):
        if self._xyz.shape[0] > 0 and self._xyz.isnan().any():
            static_prune_mask = self._xyz.isnan().any(dim=-1)
        else:
            static_prune_mask = torch.zeros(self._xyz.size(0), device=self._xyz.device, dtype=torch.bool)
        if self._xyz_dynamic.shape[0] > 0 and self._xyz_dynamic.isnan().any():
            dynamic_prune_mask = self._xyz_dynamic.isnan().flatten(start_dim=1).any(dim=-1)
        else:
            dynamic_prune_mask = torch.zeros(self._xyz_dynamic.size(0), device=self._xyz_dynamic.device, dtype=torch.bool)

        if static_prune_mask.sum() + dynamic_prune_mask.sum() > 0:
            print("Prune {} static and {} dynamic points".format(static_prune_mask.sum(), dynamic_prune_mask.sum()))
            self.prune_points(static_prune_mask, dynamic_prune_mask)
            torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, dynamic_update_filter, static_num):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[:static_num][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

        if self._point_feats.shape[0] == 0:
            return

        self.dynamic_xyz_gradient_accum[dynamic_update_filter] += torch.norm(viewspace_point_tensor[static_num:][dynamic_update_filter,:2], dim=-1, keepdim=True)
        self.dynamic_denom[dynamic_update_filter] += 1

    def add_dynamic_densification_stats(self, viewspace_point_tensor, update_filter, dynamic_update_filter, static_num):
        self.dynamic_xyz_gradient_accum[dynamic_update_filter] += torch.norm(viewspace_point_tensor[static_num:][dynamic_update_filter,:2], dim=-1, keepdim=True)
        self.dynamic_denom[dynamic_update_filter] += 1

    def update_opacity_stats(self, opacity, dynamic_update_filter, min_opacity):
        temp_opacity = opacity.detach().clone()
        # temp_opacity[temp_opacity < min_opacity] = 0
        self.opacity_accum[dynamic_update_filter] += temp_opacity[dynamic_update_filter]

    def mark_prune_stats(self, radii, viewspace_point_error_tensor):
        static_num = self._xyz.shape[0]

        static_radii = radii[:static_num]
        static_vis_filter = viewspace_point_error_tensor[:static_num, 0] > 0
        self.min_radii2D[static_vis_filter] = torch.min(self.min_radii2D[static_vis_filter], static_radii[static_vis_filter])

        if self._point_feats.shape[0] == 0:
            return

        dynamic_radii = radii[static_num:]
        dynamic_vis_filter = viewspace_point_error_tensor[static_num:, 0] > 0
        self.dynamic_min_radii2D[dynamic_vis_filter] = torch.min(self.dynamic_min_radii2D[dynamic_vis_filter],
                                                                 dynamic_radii[dynamic_vis_filter])


    # # #  self organizing gaussian
    def prune_all_but_these_indices(self, indices, dynamic_indices):
        if self.optimizer is not None:
            optimizable_tensors = self._prune_optimizer(indices, dynamic_indices)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            if self._language_feature is not None:
                self._language_feature = optimizable_tensors["language_feature"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[indices]
            self.denom = self.denom[indices]
            self.max_radii2D = self.max_radii2D[indices]
            self.min_radii2D = self.min_radii2D[indices]

            self._xyz_dynamic = optimizable_tensors["dynamic_xyz"]
            self._features_dc_dynamic = optimizable_tensors["dynamic_f_dc"]
            self._features_rest_dynamic = optimizable_tensors["dynamic_f_rest"]
            self._opacity_dynamic = optimizable_tensors["dynamic_opacity"]
            self._scaling_dynamic = optimizable_tensors["dynamic_scaling"]
            self._rotation_dynamic = optimizable_tensors["dynamic_rotation"]
            self._point_feats = optimizable_tensors[f"dynamic_feat"]

            if self._language_feature_dynamic is not None:
                self._language_feature_dynamic = optimizable_tensors["dynamic_language_feature"]

            if self._point_feats.shape[0] > 0:
                self.opacity_accum = self.opacity_accum[dynamic_indices]
                self.dynamic_xyz_gradient_accum = self.dynamic_xyz_gradient_accum[dynamic_indices]
                self.dynamic_denom = self.dynamic_denom[dynamic_indices]
                self.dynamic_max_radii2D = self.dynamic_max_radii2D[dynamic_indices]
                self.dynamic_min_radii2D = self.dynamic_min_radii2D[dynamic_indices]

        else:
            self._xyz = self._xyz[indices]
            self._features_dc = self._features_dc[indices]
            self._features_rest = self._features_rest[indices]
            self._opacity = self._opacity[indices]
            self._scaling = self._scaling[indices]
            self._rotation = self._rotation[indices]

            if self._language_feature is not None:
                self._language_feature = self._language_feature[indices]

            if dynamic_indices.any():
                self._xyz_dynamic = self._xyz_dynamic[dynamic_indices]
                self._features_dc_dynamic = self._features_dc_dynamic[dynamic_indices]
                self._features_rest_dynamic = self._features_rest_dynamic[dynamic_indices]
                self._opacity_dynamic = self._opacity_dynamic[dynamic_indices]
                self._scaling_dynamic = self._scaling_dynamic[dynamic_indices]
                self._rotation_dynamic = self._rotation_dynamic[dynamic_indices]
                self._point_feats = self._point_feats[dynamic_indices]
                if self._language_feature_dynamic is not None:
                    self._language_feature_dynamic = self._language_feature_dynamic[dynamic_indices]

    def prune_to_square_shape(self, sort_by_opacity=True, verbose=None):
        num_gaussians = self._xyz.shape[0]

        self.grid_sidelen = int(np.sqrt(num_gaussians))
        num_removed = num_gaussians - self.grid_sidelen * self.grid_sidelen

        if verbose is not None:
            verbose.write(f"Removing {num_removed}/{num_gaussians} gaussians to fit the grid. ({100 * num_removed / num_gaussians:.4f}%)")
        if self.grid_sidelen * self.grid_sidelen < num_gaussians:
            if sort_by_opacity:
                alpha = self.get_opacity[:, 0]
                _, keep_indices = torch.topk(alpha, k=self.grid_sidelen * self.grid_sidelen)
            else:
                shuffled_indices = torch.randperm(num_gaussians)
                keep_indices = shuffled_indices[:self.grid_sidelen * self.grid_sidelen]
            sorted_keep_indices = torch.sort(keep_indices)[0]
            dynamic_mask = torch.ones(self._xyz_dynamic.shape[0], dtype=torch.bool, device="cuda")
            self.prune_all_but_these_indices(sorted_keep_indices, dynamic_mask)

    def as_grid_img(self, gs_attr):
        if not hasattr(self, "grid_sidelen"):
            raise "Gaussians not pruned yet!"

        if self.grid_sidelen * self.grid_sidelen != gs_attr.shape[0]:
            raise "Tensor shape does not match img sidelen, needs pruning?"

        img = gs_attr.reshape((self.grid_sidelen, self.grid_sidelen, -1))
        return img

    def as_grid_img_dynamic(self, gs_attr):
        if self.dynamic_grid_sidelen * self.dynamic_grid_sidelen != gs_attr.shape[0]:
            raise "Tensor shape does not match img sidelen, needs pruning?"

        img = gs_attr.reshape((self.dynamic_grid_sidelen, self.dynamic_grid_sidelen, -1))
        return img

    def reshape_as_grid_img(self, gs_attr, grid_sidelen):
        # self.grid_sidelen = grid_sidelen
        return gs_attr.reshape((grid_sidelen, grid_sidelen, -1))

    def sort_into_grid(self, sorting_cfg, verbose):
        normalization_fn = self.normalize if sorting_cfg.sorting_normalize else lambda x: x

        params_to_sort = []
        for attr_name, attr_weight in sorting_cfg.sorting_weights.items():
            if attr_weight > 0:
                gs_attr = getattr(self, f"_{attr_name}")
                gs_attr = gs_attr.flatten(start_dim=1)
                params_to_sort.append(normalization_fn(gs_attr) * attr_weight)
        params_to_sort = torch.cat(params_to_sort, dim=1)

        if sorting_cfg.sorting_shuffle:
            shuffled_indices = torch.randperm(params_to_sort.shape[0], device=params_to_sort.device)
            params_to_sort = params_to_sort[shuffled_indices]

        grid_to_sort = self.as_grid_img(params_to_sort).permute(2, 0, 1)
        _, sorted_indices = sort_with_plas(grid_to_sort, improvement_break=sorting_cfg.improvement_break, verbose=verbose)
        sorted_indices = sorted_indices.squeeze().flatten()

        sorted_dynamic_indices = torch.ones(self._xyz_dynamic.shape[0], dtype=torch.bool, device="cuda")

        if sorting_cfg.sorting_shuffle:
            sorted_indices = shuffled_indices[sorted_indices.long()]

        self.prune_all_but_these_indices(sorted_indices, sorted_dynamic_indices)

    def sort_all_into_grid(self, sorting_cfg, verbose):
        normalization_fn = self.normalize if sorting_cfg.sorting_normalize else lambda x: x

        params_to_sort = []
        dynamic_params_to_sort = []
        for attr_name, attr_weight in sorting_cfg.sorting_weights.items():
            if attr_weight > 0:
                if attr_name == '_point_feats':
                    point_feat = getattr(self, f"_{attr_name}")
                    gs_attr = point_feat[:, 0, :]
                    dynamic_params_to_sort.append(normalization_fn(gs_attr) * attr_weight)
                elif attr_name.endswith('dynamic'):
                    gs_attr = getattr(self, f"_{attr_name}")
                    gs_attr = gs_attr.flatten(start_dim=1)
                    dynamic_params_to_sort.append(normalization_fn(gs_attr) * attr_weight)
                else:
                    gs_attr = getattr(self, f"_{attr_name}")
                    gs_attr = gs_attr.flatten(start_dim=1)
                    params_to_sort.append(normalization_fn(gs_attr) * attr_weight)
        params_to_sort = torch.cat(params_to_sort, dim=1)
        dynamic_params_to_sort = torch.cat(dynamic_params_to_sort, dim=1)

        if sorting_cfg.sorting_shuffle:
            shuffled_indices = torch.randperm(params_to_sort.shape[0], device=params_to_sort.device)
            params_to_sort = params_to_sort[shuffled_indices]

            shuffled_dynamic_indices = torch.randperm(dynamic_params_to_sort.shape[0], device=dynamic_params_to_sort.device)
            dynamic_params_to_sort = dynamic_params_to_sort[shuffled_dynamic_indices]

        grid_to_sort = self.as_grid_img(params_to_sort).permute(2, 0, 1)
        _, sorted_indices = sort_with_plas(grid_to_sort, improvement_break=sorting_cfg.improvement_break, verbose=verbose)
        sorted_indices = sorted_indices.squeeze().flatten()

        if self._point_feats.shape[0] > 10000:
            grid_to_sort = self.as_grid_img_dynamic(dynamic_params_to_sort).permute(2, 0, 1)
            _, sorted_dynamic_indices = sort_with_plas(grid_to_sort, improvement_break=sorting_cfg.improvement_break, verbose=verbose)
            sorted_dynamic_indices = sorted_dynamic_indices.squeeze().flatten()
        else:
            sorted_dynamic_indices = torch.ones(self._xyz_dynamic.shape[0], dtype=torch.bool, device="cuda")
            # sorted_dynamic_indices = torch.empty(0).cuda()

        if sorting_cfg.sorting_shuffle:
            sorted_indices = shuffled_indices[sorted_indices.long()]
            sorted_dynamic_indices = shuffled_dynamic_indices[sorted_dynamic_indices.long()]

        self.prune_all_but_these_indices(sorted_indices, sorted_dynamic_indices)

    def sort_dynamic_into_grid(self, sorting_cfg, verbose):
        normalization_fn = self.normalize if sorting_cfg.sorting_normalize else lambda x: x

        params_to_sort = []
        dynamic_params_to_sort = []
        for attr_name, attr_weight in sorting_cfg.sorting_dynamic_weights.items():
            if attr_weight > 0:
                if attr_name == '_point_feats':
                    point_feat = getattr(self, f"_{attr_name}")
                    gs_attr = point_feat[:, 0, :]
                    dynamic_params_to_sort.append(normalization_fn(gs_attr) * attr_weight)
                else:
                    gs_attr = getattr(self, f"_{attr_name}")
                    gs_attr = gs_attr.flatten(start_dim=1)
                    dynamic_params_to_sort.append(normalization_fn(gs_attr) * attr_weight)
        dynamic_params_to_sort = torch.cat(dynamic_params_to_sort, dim=1)

        if sorting_cfg.sorting_shuffle:
            shuffled_dynamic_indices = torch.randperm(dynamic_params_to_sort.shape[0], device=dynamic_params_to_sort.device)
            dynamic_params_to_sort = dynamic_params_to_sort[shuffled_dynamic_indices]

        grid_to_sort = self.as_grid_img_dynamic(dynamic_params_to_sort).permute(2, 0, 1)
        _, sorted_dynamic_indices = sort_with_plas(grid_to_sort, improvement_break=sorting_cfg.improvement_break, verbose=verbose)
        sorted_dynamic_indices = sorted_dynamic_indices.squeeze().flatten()

        sorted_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")

        if sorting_cfg.sorting_shuffle:
            sorted_dynamic_indices = shuffled_dynamic_indices[sorted_dynamic_indices.long()]

        self.prune_all_but_these_indices(sorted_indices, sorted_dynamic_indices)

    @staticmethod
    def normalize(tensor):
        tensor = tensor - tensor.mean()
        if tensor.std() > 0:
            tensor = tensor / tensor.std()
        return tensor

    def neighborloss_2d(self, attr_name, neighbor_cfg):
        if attr_name == '_point_feats':
            point_feat = getattr(self, f"_{attr_name}")
            gs_attr = point_feat[:, 0, :]
        else:
            if self.offset_mode >= 1:
                gs_attr = getattr(self, f"get_{attr_name}_ori")
            else:
                gs_attr = getattr(self, f"_{attr_name}")

        if neighbor_cfg.normalize:
            gs_attr = self.normalize(gs_attr)

        if attr_name.endswith('dynamic'):
            if self._xyz_dynamic.shape[0] > 10000:
                attr_img = self.as_grid_img_dynamic(gs_attr)
            else:
                return 0
        else:
            attr_img = self.as_grid_img(gs_attr)
        attr_img = attr_img.permute(2, 0, 1).unsqueeze(0)

        blurred_x = kornia.filters.gaussian_blur2d(
            attr_img.detach(),
            kernel_size=(1, neighbor_cfg.kernel_size),
            sigma=(neighbor_cfg.sigma, neighbor_cfg.sigma),
            border_type="circular",
        )

        blurred_xy = kornia.filters.gaussian_blur2d(
            blurred_x,
            kernel_size=(neighbor_cfg.kernel_size, 1),
            sigma=(neighbor_cfg.sigma, neighbor_cfg.sigma),
            border_type="reflect",
        )

        return F.huber_loss(blurred_xy, attr_img)

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

    def attr_as_grid_img(self, attr_name):
        if '_' not in attr_name:
            attr_name = '_' + attr_name
        if 'offset' in attr_name:
            gs_attr = getattr(self, f"{attr_name}")
            gs_attr = gs_attr.detach().cpu()
            if 'dynamic' in attr_name:
                return [self.as_grid_img_dynamic(gs_attr)]
            return [self.as_grid_img(gs_attr)]
        elif 'point_feat' not in attr_name:
            gs_attr = getattr(self, f"{attr_name}")
            gs_attr = gs_attr.detach().cpu()
            if attr_name.endswith('dynamic'):
                return [self.as_grid_img_dynamic(gs_attr)]
            return [self.as_grid_img(gs_attr)]
        else:
            imgs = []
            for i in range(self.keyframe_num):
                gs_attr = self._point_feats[:, i, :]
                gs_attr = gs_attr.detach().cpu()

                img = self.as_grid_img_dynamic(gs_attr)  # width, length, 16
                img = self.frame_maker(img, self.dynamic_grid_sidelen, self.dynamic_grid_sidelen)  # width * 4, length * 4
                imgs.append(img)
            return imgs

    def set_attr_from_grid_img(self, attr_name, img, original_gaussian=None):
        if self.optimizer is not None:
            raise "Overwriting Gaussians during training not implemented yet! - Consider pruning method implementations"

        attr_shapes = {
            "_xyz": (3,),
            "_features_dc": (1, 3),
            "_features_rest": (3, 3),
            "_rotation": (4,),
            "_scaling": (3,),
            "_opacity": (self.opacity_dim,),
            "_language_feature": (self.lang_feat_dim,),

            "_xyz_dynamic": (3,),
            "_features_dc_dynamic": (1, 3),
            "_features_rest_dynamic": (3, 3),
            "_rotation_dynamic": (4,),
            "_scaling_dynamic": (3,),
            "_opacity_dynamic": (self.opacity_dim,),
            "_language_feature_dynamic": (self.lang_feat_dim,),

            "_xyz_dynamic_offset": (3,),
            "_xyz_offset": (3,),
        }
        if attr_name.endswith('dynamic'):
            self.dynamic_grid_sidelen = img.shape[0]
        self.grid_sidelen = img.shape[0]
        target_shape = attr_shapes[attr_name]
        img_shaped = img.reshape(-1, *target_shape)
        tensor = torch.tensor(img_shaped, dtype=torch.float, device="cuda")

        setattr(self, attr_name, tensor)

    def set_point_feat_from_grid_img(self, feat_imgs, keyframe_num, xyz_dynamic_mask):
        new_point_feats = None
        width = feat_imgs[0].shape[0] // 4
        height = feat_imgs[0].shape[1] // 4

        for idx, img in enumerate(feat_imgs):
            feats = np.zeros((width, height, self.feat_dim))
            x, y = 0, 0
            for i in range(self.feat_dim):
                if y >= img.shape[1]:
                    y = 0
                    x += width
                assert x + width <= img.shape[0], f"Tile_maker: not enough space, needs {x + width}x{y} but only have {width * 4}"
                feats[:, :, i] = img[x:x+width, y:y+height]
                y += height
            feats = feats.reshape(-1, self.feat_dim)
            tensor = torch.tensor(feats, dtype=torch.float, device="cuda")
            if new_point_feats is None:
                new_point_feats = torch.zeros((width*height, keyframe_num, self.feat_dim), device="cuda")
                new_point_feats[:, idx, :] = tensor
            else:
                new_point_feats[:, idx, :] = tensor
        new_point_feats = new_point_feats.contiguous()

        setattr(self, f'_point_feats', new_point_feats)

    @staticmethod
    def get_geo_attr_from_grid_img(attr_name, img, opacity_dim: int):
        attr_shapes = {
            "_xyz": (3,),
            "_features_dc": (1, 3),
            "_features_rest": (3, 3),
            "_rotation": (4,),
            "_scaling": (3,),
            "_opacity": (opacity_dim,),
            "_language_feature": (9,),

            "_xyz_dynamic": (3,),
            "_features_dc_dynamic": (1, 3),
            "_features_rest_dynamic": (3, 3),
            "_rotation_dynamic": (4,),
            "_scaling_dynamic": (3,),
            "_opacity_dynamic": (opacity_dim,),
            "_language_feature_dynamic": (9,),
        }

        target_shape = attr_shapes[attr_name]
        img_shaped = img.reshape(-1, *target_shape)
        tensor = torch.tensor(img_shaped, dtype=torch.float, device="cuda")
        return tensor

    def get_point_feat_from_grid_img(self, feat_imgs, keyframe_num):
        new_point_feats = None
        width = feat_imgs[0].shape[0] // 4
        height = feat_imgs[0].shape[1] // 4

        for idx, img in enumerate(feat_imgs):
            feats = np.zeros((width, height, self.feat_dim))
            x, y = 0, 0
            for i in range(self.feat_dim):
                if y >= img.shape[1]:
                    y = 0
                    x += width
                assert x + width <= img.shape[0], f"Tile_maker: not enough space, needs {x + width}x{y} but only have {width * 4}"
                feats[:, :, i] = img[x:x+width, y:y+height]
                y += height
            feats = feats.reshape(-1, self.feat_dim)
            tensor = torch.tensor(feats, dtype=torch.float, device="cuda")
            if new_point_feats is None:
                new_point_feats = torch.zeros((width*height, keyframe_num, self.feat_dim), device="cuda")
                new_point_feats[:, idx, :] = tensor
            else:
                new_point_feats[:, idx, :] = tensor
        new_point_feats = new_point_feats.contiguous()
        return new_point_feats

    def load_compressed_offset(self, path, skip=[]):
        import pickle
        with open(path,'rb') as f:
            data = pickle.load(f)
            latents = data['latents']
            decoder_state_dict = data['decoder_state_dict']
            decoder_args = data['decoder_args']

        for attribute in latents.keys():
            if attribute not in skip:
                self.quat_decoders[attribute[1:]] = LatentDecoder(**decoder_args[attribute]).cuda()
                self.quat_decoders[attribute[1:]].load_state_dict(decoder_state_dict[attribute])
                tensor = latents[attribute].uncompress().cuda()
                setattr(self, attribute, tensor)
