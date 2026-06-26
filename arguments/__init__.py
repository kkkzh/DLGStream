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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self.dataset_type = ""
        self.num_times = 300
        self._images = "images"
        self._resolution = -1
        self.load_memory = False
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process = False
        self.extension = ".png"
        self.near = 0.2
        self.far = 300.0
        self.feature_level = 1  # 1, 2, 3
        self.dy_threshold = 5.0

        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4
        self.enable_filter = True

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.normalize = False
        self.activated = False
        self.kernel_size = 5
        self.sigma = 3.0
        self.neighbor_loss_weight = {
            'xyz': 0.0,
            'features_dc': 0.0,
            'features_rest': 0.0,
            'opacity': 1.0,
            'scaling': 0.0,
            'rotation': 10.0,

            'xyz_dynamic': 0.0,
            'features_dc_dynamic': 0.0,
            'features_rest_dynamic': 0.0,
            'opacity_dynamic': 1.0,
            'scaling_dynamic': 0.0,
            'rotation_dynamic': 10.0
        }

        self.static_neighbor_loss_weight = {
            'xyz': 0.0,
            'features_dc': 0.0,
            'features_rest': 0.0,
            'opacity': 1.0,
            'scaling': 0.0,
            'rotation': 10.0,
        }

        self.dynamic_neighbor_loss_weight = {
            'xyz_dynamic': 0.0,
            'features_dc_dynamic': 0.0,
            'features_rest_dynamic': 0.0,
            'opacity_dynamic': 1.0,
            'scaling_dynamic': 0.0,
            'rotation_dynamic': 10.0
        }

        self.sorting_enabled = True
        self.sorting_normalize = True
        self.sorting_activated = True
        self.sorting_shuffle = True
        self.improvement_break = 0.0001
        self.sorting_weights = {
            'xyz': 1.0,
            'features_dc': 1.0,
            'features_rest': 0.0,
            'opacity': 0.0,
            'scaling': 1.0,
            'rotation': 0.0
        }

        self.sorting_dynamic_weights = {
            'xyz_dynamic': 1.0,
            'features_dc_dynamic': 1.0,
            'features_rest_dynamic': 0.0,
            'opacity_dynamic': 0.0,
            'scaling_dynamic': 1.0,
            'rotation_dynamic': 0.0
        }
        # --- self-organizing-gaussian

        super().__init__(parser, "ModelHiddenParams")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False

        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.refine_iterations = 12000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.language_feature_lr = 0.002

        self.temporal_feature_lr_init = 0.001  # 0.001 | 0.0025
        self.temporal_feature_lr_final = 0.00001  # 0.00001 | 0.000025
        self.temporal_feature_lr_delay_mult = 0.01
        self.temporal_feature_lr_steps = 30_000

        self.mlp_deform_lr_init = 0.005  # 0.005  0.01
        self.mlp_deform_lr_final = 0.0005  # 0.00005 0.0001
        self.mlp_deform_lr_delay_mult = 0.01
        self.mlp_deform_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_lang_lr_init = 0.001
        self.mlp_lang_lr_final = 0.00001
        self.mlp_lang_lr_delay_mult = 0.01
        self.mlp_lang_lr_max_steps = 30_000

        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.encoding_xyz_lr_init = 0.005
        self.encoding_xyz_lr_final = 0.00001
        self.encoding_xyz_lr_delay_mult = 0.33
        self.encoding_xyz_lr_max_steps = 30_000

        self.mlp_grid_lr_init = 0.005
        self.mlp_grid_lr_final = 0.00001
        self.mlp_grid_lr_delay_mult = 0.01
        self.mlp_grid_lr_max_steps = 30_000

        self.hac_opacity_lr_init = 0.002
        self.hac_opacity_lr_final = 0.00002
        self.hac_opacity_lr_delay_mult = 0.01
        self.hac_opacity_lr_max_steps = 30_000

        self.hac_cov_lr_init = 0.004
        self.hac_cov_lr_final = 0.004
        self.hac_cov_lr_delay_mult = 0.01
        self.hac_cov_lr_max_steps = 30_000

        self.hac_color_lr_init = 0.008
        self.hac_color_lr_final = 0.00005
        self.hac_color_lr_delay_mult = 0.01
        self.hac_color_lr_max_steps = 30_000

        self.success_threshold = 0.8
        self.step_flag1 = 3000
        self.step_flag2 = 6000

        self.feature_lr = 0.0025
        self.feature_lr_final = 0.00025
        self.opacity_lr = 0.05
        self.opacity_lr_final = 0.0005
        self.scaling_lr = 0.005
        self.scaling_lr_final = 0.0005
        self.rotation_lr = 0.001
        self.rotation_lr_final = 0.00005  # 0.00005

        self.rotation_dynamic_offset_lr = 0.05
        self.rotation_offset_lr = 0.05

        self.percent_dense = 0.01
        self.lambda_dssim = 0
        self.lambda_lpips = 0

        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.static_densification_interval = 100
        self.dynamic_densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.static_densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_static = 0.0002
        self.densify_grad_threshold_dynamic = 0.0002
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_static = 0.005
        self.opacity_threshold_dynamic = 0.005
        self.batch_size = 1

        self.position_erp = 'chip'  # chip
        self.rotation_erp = 'slerp'  # slerp

        # ex4dgs
        self.prune_every = 5
        self.error_base_prune_steps = 5000
        self.s_l1_thres = 0.08
        self.s_max_ssim = 0.6
        self.d_l1_thres = 0.08
        self.d_max_ssim = 0.6

        # regularization
        self.transformer_reg = 0.1
        self.neighbor_reg = 1.0
        self.temporal_reg = 1.0
        self.temporal_reg2 = 1.5
        self.opacity_reg = 0.01
        self.scaling_reg = 0.01
        self.static_latent_reg = 0.001  # 0.001
        self.static_latent_reg_start = 0
        self.dynamic_latent_reg = 0.0001  # 0.0001
        self.dynamic_latent_reg_start = 0

        self.coarse_gaussian_num = 80000
        self.num_gaussian = 150000
        self.num_gaussian2 = 80000
        # loss
        self.stage = [2000, 5000, 6000, 9000]
        self.s_ssim = 0.15
        self.d_ssim = 0.06
        self.d_ssim2 = 0.2
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
