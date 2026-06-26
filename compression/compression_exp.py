import sys
import os
import yaml
import json
import time

import numpy as np
import pandas as pd
import cv2
import torch
from argparse import ArgumentParser
from dataclasses import dataclass, asdict

# from scene.streamoff_gaussian_model import GaussianModel

from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.system_utils import do_system

from compression.jpeg_xl import JpegXlCodec
from compression.npz import NpzCodec
from compression.exr import EXRCodec
from compression.png import PNGCodec

codecs = {
    "jpeg-xl": JpegXlCodec,
    "npz": NpzCodec,
    "exr": EXRCodec,
    "png": PNGCodec,
}

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


@dataclass
class QuantEval:
    psnr: float
    ssim: float
    lpips: float


@dataclass
class Measurement:
    name: str
    path: str
    size_bytes: int
    quant_eval: QuantEval = None

    @property
    def human_readable_byte_size(self):
        if self.size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(np.floor(np.log(self.size_bytes) / np.log(1000)))
        p = np.power(1000, i)
        s = round(self.size_bytes / p, 2)
        return f"{s}{size_name[i]}"
    
    def to_dict(self):
        d = asdict(self)
        d.pop('quant_eval')
        if self.quant_eval is not None:
            d.update(self.quant_eval.__dict__)
        d['size'] = self.human_readable_byte_size
        return d


def log_transform(coords):
    positive = coords > 0
    negative = coords < 0
    zero = coords == 0

    transformed_coords = np.zeros_like(coords)
    transformed_coords[positive] = np.log1p(coords[positive])
    transformed_coords[negative] = -np.log1p(-coords[negative])
    # For zero, no change is needed as transformed_coords is already initialized to zeros

    return transformed_coords


def inverse_log_transform(transformed_coords):
    positive = transformed_coords > 0
    negative = transformed_coords < 0
    zero = transformed_coords == 0

    original_coords = np.zeros_like(transformed_coords)
    original_coords[positive] = np.expm1(transformed_coords[positive])
    original_coords[negative] = -np.expm1(-transformed_coords[negative])
    # For zero, no change is needed as original_coords is already initialized to zeros

    return original_coords


def get_attr_numpy(gaussians, attr_name):
    attr_tensors = gaussians.attr_as_grid_img(attr_name)
    attr_numpy = [attr_tensor.numpy() for attr_tensor in attr_tensors]
    return attr_numpy


def compress_attr(attr_config, gaussians, out_folder):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    attr_params = attr_config.get('params', {})
    attr_min = attr_config['min']
    attr_max = attr_config['max']
    unuse_threshold = False
    if attr_min == 0 and attr_max == 0:
        unuse_threshold = True
    
    if not attr_params:
        attr_params = {}
    
    codec = codecs[attr_method]()
    attr_nps = get_attr_numpy(gaussians, attr_name)
    if 'point_feat' in attr_name or 'temporal_feat' in attr_name:
        stacked_attr_np = np.concatenate(attr_nps)
        attr_min = np.min(stacked_attr_np)
        attr_max = np.max(stacked_attr_np)
    for i in range(len(attr_nps)):
        attr_np = attr_nps[i]
        file_name = f"{attr_name}.{codec.file_ending()}" if len(attr_nps) == 1 else f"{attr_name}_{i}.{codec.file_ending()}"
        if 'point_feat' in attr_name or 'temporal_feat' in attr_name:
            out_file = os.path.join(out_folder, 'feat_images', file_name)
        else:
            out_file = os.path.join(out_folder, file_name)

        if attr_config.get('contract', False):
            attr_np = log_transform(attr_np)

        if "quantize" in attr_config and gaussians.offset_mode == 0:
            quantization = attr_config["quantize"]
            attr_min_val = attr_np.min()
            attr_max_val = attr_np.max()
            val_range = attr_max_val - attr_min_val
            # no division by zero
            if val_range == 0:
                val_range = 1
            attr_np_norm = (attr_np - attr_min_val) / val_range
            qpow = 2 ** quantization
            attr_np_quantized = np.round(attr_np_norm * qpow) / qpow
            attr_np = attr_np_quantized * val_range + attr_min_val
            attr_np = attr_np.astype(np.float32)

        if attr_config.get('normalize', False):
            min_val, max_val = codec.encode_with_normalization(attr_np, attr_name, out_file, attr_min, attr_max, unuse_threshold=unuse_threshold, **attr_params)
            # return file_name, min_val, max_val
        else:
            codec.encode(attr_np, out_file, **attr_params)
            # return file_name, None, None
    if attr_config.get('normalize', False):
        return file_name, min_val, max_val
    else:
        return file_name, None, None


def run_single_compression(gaussians, experiment_out_path, experiment_config, gopid=None):
    compressed_min_vals = {}
    compressed_max_vals = {}

    compressed_files = {}

    total_size_bytes = 0

    for attribute in experiment_config['attributes']:
        attr_name = attribute['name']
        if (gopid == 0) or ('point_feat' in attr_name) or ('temporal_feat' in attr_name):
            compressed_file, min_val, max_mal = compress_attr(attribute, gaussians, experiment_out_path)

            compressed_files[attr_name] = compressed_file
            compressed_min_vals[attr_name] = min_val
            compressed_max_vals[attr_name] = max_mal

    if gaussians.offset_mode % 10 == 2:
        compress_offset = []
        xyz_dynamic_offset = experiment_config["dy_offset"][0]
        compress_offset.append(xyz_dynamic_offset)
        if gaussians.offset_mode == 22:
            xyz_offset = experiment_config["st_offset"][0]
            compress_offset.append(xyz_offset)
        for attribute in compress_offset:
            compressed_file, min_val, max_mal = compress_attr(attribute, gaussians, experiment_out_path)
            attr_name = attribute['name']
            compressed_files[attr_name] = compressed_file
            compressed_min_vals[attr_name] = min_val
            compressed_max_vals[attr_name] = max_mal

    compr_info = pd.DataFrame([compressed_min_vals, compressed_max_vals, compressed_files], index=["min", "max", "file"]).T
    compr_info.to_csv(os.path.join(experiment_out_path, "compression_info.csv"))

    if hasattr(gaussians, 'max_sh_degree'):
        experiment_config['max_sh_degree'] = gaussians.max_sh_degree
        experiment_config['active_sh_degree'] = gaussians.active_sh_degree
        experiment_config['disable_xyz_log_activation'] = gaussians.disable_xyz_log_activation
        experiment_config['time_embedding_num'] = gaussians.time_embedding_num

    return total_size_bytes, experiment_config


def run_compressions(gaussians, out_path, compr_exp_config, gopid=None, qp=20):
    from .decoders import CompressedLatents
    results = {}
    for experiment in compr_exp_config['experiments']:
        experiment_name = experiment['name']
        experiment_out_path = os.path.join(out_path, experiment_name)
        os.makedirs(experiment_out_path, exist_ok=True)
        feat_images_dir = os.path.join(experiment_out_path, 'feat_images')
        os.makedirs(feat_images_dir, exist_ok=True)

        size_bytes, config_ = run_single_compression(gaussians, experiment_out_path, experiment, gopid)

        # compress canonical gaussians offsets
        if gaussians.offset_mode % 10 == 2:
            from collections import OrderedDict
            import pickle
            pkl_path = os.path.join(experiment_out_path, 'offset_compressed.pkl')
            latents = OrderedDict()
            decoder_state_dict = OrderedDict()
            decoder_args = OrderedDict()
            for idx, attr in enumerate(gaussians.quat_attrbutes):
                latent = gaussians.__getattr__(f'_{attr}').detach().cpu()
                compressed_obj = CompressedLatents()
                compressed_obj.compress(latent)
                latents[f'_{attr}'] = compressed_obj
                ldecode_matrix = "learnable"
                decoder_args[f'_{attr}'] = {
                    'latent_dim': gaussians.quat_decoder_latent_dims[idx],
                    'feature_dim': gaussians.quat_decoder_feature_dims[idx],
                    'ldecode_matrix': ldecode_matrix,
                }
                decoder_state_dict[f'_{attr}'] = gaussians.quat_decoders[attr].state_dict().copy()
            # compress static gaussian offsets
            if gaussians.offset_mode == 22:
                for idx, attr in enumerate(gaussians.quat_static_attrbutes):
                    latent = gaussians.__getattr__(f'_{attr}').detach().cpu()
                    compressed_obj = CompressedLatents()
                    compressed_obj.compress(latent)
                    latents[f'_{attr}'] = compressed_obj
                    ldecode_matrix = "learnable"
                    decoder_args[f'_{attr}'] = {
                        'latent_dim': gaussians.quat_decoder_latent_dims[idx],
                        'feature_dim': gaussians.quat_decoder_feature_dims[idx],
                        'ldecode_matrix': ldecode_matrix,
                    }
                    decoder_state_dict[f'_{attr}'] = gaussians.quat_decoders[attr].state_dict().copy()

            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'latents': latents,
                    'decoder_state_dict': decoder_state_dict,
                    'decoder_args': decoder_args,
                }, f)

        do_system(f'ffmpeg -y -framerate 30 -i {feat_images_dir}/_point_feats_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {qp} {experiment_out_path}/_point_feats.mp4')
        results[f"size_bytes/cmpr_{experiment['name']}"] = size_bytes
        results[experiment_name] = config_
        results['out_path'] = experiment_out_path

        gaussians.save(experiment_out_path)
        gaussians.save_mlps(experiment_out_path)
    return results


def decompress_attr(gaussians, attr_config, compressed_file, min_val, max_val, original_gaussian):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    if 'point_feat' not in attr_name:
        codec = codecs[attr_method]()

        if attr_config.get('normalize', False):
            decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
        else:
            decompressed_attr = codec.decode(compressed_file)

        if attr_config.get('contract', False):
            decompressed_attr = inverse_log_transform(decompressed_attr)

        gaussians.set_attr_from_grid_img(attr_name, decompressed_attr, original_gaussian)
    else:
        compressed_dir = os.path.dirname(compressed_file)
        compressed_file = os.path.join(compressed_dir, '_point_feats.mp4')
        do_system(f"ffmpeg -y -i {compressed_file} -start_number 0 -pix_fmt gray16be {compressed_dir}/feat_images/_point_feats_%d_out.png")
        codec = codecs[attr_method]()
        feat_imgs = []
        for i in range(gaussians.time_embedding_num):
            compressed_file = os.path.join(compressed_dir, 'feat_images', f'_point_feats_{i}_out.png')
            if attr_config.get('normalize', False):
                decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
            else:
                decompressed_attr = codec.decode(compressed_file)
            feat_imgs.append(decompressed_attr)
            os.remove(compressed_file)
        if hasattr(gaussians, '_xyz_dynamic_mask'):
            dynamic_mask = gaussians._xyz_dynamic_mask
        else:
            dynamic_mask = None
        gaussians.set_point_feat_from_grid_img(feat_imgs, gaussians.time_embedding_num, dynamic_mask)


def run_single_decompression(compressed_dir, experiment_config, GaussianModel, original_gaussian=None, skip=[]):
    compr_info = pd.read_csv(os.path.join(compressed_dir, "compression_info.csv"), index_col=0)

    max_sh_degree = experiment_config['max_sh_degree']
    decompressed_gaussians = GaussianModel(experiment_config['max_sh_degree'], experiment_config['disable_xyz_log_activation'])
    decompressed_gaussians.active_sh_degree = experiment_config['active_sh_degree']
    decompressed_gaussians.time_embedding_num = experiment_config['time_embedding_num']
    if hasattr(original_gaussian, '_xyz_dynamic_mask'):
        decompressed_gaussians._xyz_dynamic_mask = original_gaussian._xyz_dynamic_mask

    for attribute in experiment_config['attributes']:
        attr_name = attribute["name"]
        if attr_name not in skip:
            compressed_file = os.path.join(compressed_dir, compr_info.loc[attr_name, "file"])
            decompress_attr(decompressed_gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"], original_gaussian)
        else:
            attr_tensor = getattr(original_gaussian, f"{attr_name}")
            setattr(decompressed_gaussians, attr_name, attr_tensor)

    if original_gaussian.offset_mode % 10 == 2:
        decompressed_gaussians.offset_mode = original_gaussian.offset_mode
        decompressed_gaussians.load_compressed_offset(os.path.join(compressed_dir, "offset_compressed.pkl"))

        compress_offset = []
        xyz_dynamic_offset = experiment_config["dy_offset"][0]
        compress_offset.append(xyz_dynamic_offset)
        if decompressed_gaussians.offset_mode == 22:
            xyz_offset = experiment_config["st_offset"][0]
            compress_offset.append(xyz_offset)
        for attribute in compress_offset:
            attr_name = attribute["name"]
            compressed_file = os.path.join(compressed_dir, compr_info.loc[attr_name, "file"])
            decompress_attr(decompressed_gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"], original_gaussian)

    # copy parameters of deformation field
    decompressed_gaussians.mlp_lang.load_state_dict(original_gaussian.mlp_lang.state_dict())
    decompressed_gaussians.mlp_deform.load_state_dict(original_gaussian.mlp_deform.state_dict())
    decompressed_gaussians.mlp_cov.load_state_dict(original_gaussian.mlp_cov.state_dict())
    decompressed_gaussians.mlp_opacity.load_state_dict(original_gaussian.mlp_opacity.state_dict())
    decompressed_gaussians.mlp_color.load_state_dict(original_gaussian.mlp_color.state_dict())
    if hasattr(decompressed_gaussians, 'sd_classify'):
        decompressed_gaussians.sd_classify.load_state_dict(original_gaussian.sd_classify.state_dict())

    return decompressed_gaussians


def run_decompressions(compressions_dir, experiment_configs, skip, original_gaussian=None):
    for compressed_dir in os.listdir(compressions_dir):
        compressed_dir_path = os.path.join(compressions_dir, compressed_dir)
        if not os.path.isdir(compressed_dir_path):
            continue
        yield os.path.basename(compressed_dir_path), run_single_decompression(compressed_dir_path, experiment_configs[str(compressed_dir)], original_gaussian, skip)


def decompress_attr_2(gaussians, attr_config, compressed_file, min_val, max_val, original_gaussian, time_feat_dir):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    if 'point_feat' not in attr_name:
        codec = codecs[attr_method]()

        if attr_config.get('normalize', False):
            decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
        else:
            decompressed_attr = codec.decode(compressed_file)

        if attr_config.get('contract', False):
            decompressed_attr = inverse_log_transform(decompressed_attr)

        gaussians.set_attr_from_grid_img(attr_name, decompressed_attr)
    else:
        compressed_dir = os.path.dirname(compressed_file)
        # compressed_file = os.path.join(compressed_dir, '_point_feats.mp4')
        # do_system(f"ffmpeg -y -i {compressed_file} -start_number 0 -pix_fmt gray16be {compressed_dir}/feat_images/_point_feats_%d_out.png")
        codec = codecs[attr_method]()
        feat_imgs = []
        for i in range(gaussians.time_embedding_num):
            compressed_file = os.path.join(compressed_dir, time_feat_dir, f'_point_feats_{i}_out.png')
            if not os.path.exists(compressed_file):
                print(f'{compressed_file} not found!')
                sys.exit(-1)
            if attr_config.get('normalize', False):
                decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
            else:
                decompressed_attr = codec.decode(compressed_file)
            feat_imgs.append(decompressed_attr)
        gaussians.set_point_feat_from_grid_img(feat_imgs, gaussians.time_embedding_num, None)


def decompress_attr_3(gaussians, attr_config, compressed_file, min_val, max_val, original_gaussian, temporal_features):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    if 'point_feat' not in attr_name:
        codec = codecs[attr_method]()

        if attr_config.get('normalize', False):
            decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
        else:
            decompressed_attr = codec.decode(compressed_file)

        if attr_config.get('contract', False):
            decompressed_attr = inverse_log_transform(decompressed_attr)

        gaussians.set_attr_from_grid_img(attr_name, decompressed_attr)
    else:
        assert len(temporal_features) == gaussians.time_embedding_num, f"feature nums={len(temporal_features)}, while target feature nums={gaussians.time_embedding_num}"
        feat_imgs = []
        for i in range(gaussians.time_embedding_num):
            if attr_config.get('normalize', False):
                decompressed_attr = temporal_features[i] / 65535 * (max_val - min_val) + min_val
            else:
                decompressed_attr = temporal_features[i] / 65535
            feat_imgs.append(decompressed_attr)
        gaussians.set_point_feat_from_grid_img(feat_imgs, gaussians.time_embedding_num, None)


def decompress_geo_attr(gaussians, attr_config, compressed_file, min_val, max_val, opacity_dim: int):
    attr_name = attr_config['name']
    attr_method = attr_config['method']

    if 'point_feat' not in attr_name:
        codec = codecs[attr_method]()

        if attr_config.get('normalize', False):
            decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
        else:
            decompressed_attr = codec.decode(compressed_file)

        if attr_config.get('contract', False):
            decompressed_attr = inverse_log_transform(decompressed_attr)

        attr_tensor = gaussians.get_geo_attr_from_grid_img(attr_name, decompressed_attr, opacity_dim)
    else:
        from test_fdsd import measure_decode_latency
        temporal_features = measure_decode_latency(compressed_file)
        assert len(temporal_features) == gaussians.time_embedding_num, f"feature nums={len(temporal_features)}, while target feature nums={gaussians.time_embedding_num}"
        feat_imgs = []
        for i in range(gaussians.time_embedding_num):
            if attr_config.get('normalize', False):
                decompressed_attr = temporal_features[i] / 65535 * (max_val - min_val) + min_val
            else:
                decompressed_attr = temporal_features[i] / 65535
            feat_imgs.append(decompressed_attr)
        attr_tensor = gaussians.get_point_feat_from_grid_img(feat_imgs, gaussians.time_embedding_num)

    return attr_tensor


def run_compressions_hac(gaussians, out_path, compr_exp_config, gopid=None, qp=20):
    results = {}
    for experiment in compr_exp_config['experiments']:
        experiment_name = experiment['name']
        experiment_out_path = os.path.join(out_path, experiment_name)
        os.makedirs(experiment_out_path, exist_ok=True)
        feat_images_dir = os.path.join(experiment_out_path, 'feat_images')
        os.makedirs(feat_images_dir, exist_ok=True)

        size_bytes, config_ = run_single_compression(gaussians, experiment_out_path, experiment, gopid)
        config_["N_dynamic"] = gaussians._anchor_dynamic.shape[0]

        do_system(f'ffmpeg -y -framerate 30 -i {feat_images_dir}/_temporal_feat_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {qp} {experiment_out_path}/_temporal_feat.mp4')
        results[f"size_bytes/cmpr_{experiment['name']}"] = size_bytes
        results[experiment_name] = config_
        results['out_path'] = experiment_out_path

        gaussians.save(os.path.join(experiment_out_path, 'model.pth'))
        gaussians.save_mlps(os.path.join(experiment_out_path, 'mlps.pth'))
    return results


def decompress_attr_hac(gaussians, attr_config, compressed_file, min_val, max_val, N_dynamic):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    if 'temporal_feat' not in attr_name:
        pass
    else:
        compressed_dir = os.path.dirname(compressed_file)
        compressed_file = os.path.join(compressed_dir, '_temporal_feat.mp4')
        do_system(f"ffmpeg -y -i {compressed_file} -start_number 0 -pix_fmt gray16be {compressed_dir}/feat_images/_temporal_feat_%d_out.png")
        codec = codecs[attr_method]()
        feat_imgs = []
        for i in range(gaussians.keyframe_num):
            compressed_file = os.path.join(compressed_dir, 'feat_images', f'_temporal_feat_{i}_out.png')
            if attr_config.get('normalize', False):
                decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
            else:
                decompressed_attr = codec.decode(compressed_file)
            feat_imgs.append(decompressed_attr)
            os.remove(compressed_file)

        gaussians.set_point_feat_from_grid_img(feat_imgs, gaussians.keyframe_num, N_dynamic)

def run_single_decompression_hac(compressed_dir, experiment_config, decompressed_gaussians, original_gaussian=None, skip=[]):
    compr_info = pd.read_csv(os.path.join(compressed_dir, "compression_info.csv"), index_col=0)

    N_dynamic = experiment_config['N_dynamic']
    for attribute in experiment_config['attributes']:
        attr_name = attribute["name"]
        if attr_name not in skip:
            compressed_file = os.path.join(compressed_dir, compr_info.loc[attr_name, "file"])
            decompress_attr_hac(decompressed_gaussians, attribute, compressed_file, compr_info.loc[attr_name, "min"], compr_info.loc[attr_name, "max"], N_dynamic)
        else:
            attr_tensor = getattr(original_gaussian, f"{attr_name}")
            setattr(decompressed_gaussians, attr_name, attr_tensor)

    # gaussian mlps
    decompressed_gaussians.mlp_grid.load_state_dict(original_gaussian.mlp_grid.state_dict())
    decompressed_gaussians.mlp_opacity.load_state_dict(original_gaussian.mlp_opacity.state_dict())
    decompressed_gaussians.mlp_cov.load_state_dict(original_gaussian.mlp_cov.state_dict())
    decompressed_gaussians.mlp_color.load_state_dict(original_gaussian.mlp_color.state_dict())
    decompressed_gaussians.x_bound_min = original_gaussian.x_bound_min
    decompressed_gaussians.x_bound_max = original_gaussian.x_bound_max
    if original_gaussian.mlp_language is not None:
        decompressed_gaussians.mlp_language.load_state_dict(original_gaussian.mlp_language.state_dict())
    # deformation mlps
    decompressed_gaussians.mlp_deform_xyz.load_state_dict(original_gaussian.mlp_deform_xyz.state_dict())
    decompressed_gaussians.mlp_deform_cov.load_state_dict(original_gaussian.mlp_deform_cov.state_dict())
    decompressed_gaussians.mlp_deform_color.load_state_dict(original_gaussian.mlp_deform_color.state_dict())
    decompressed_gaussians.mlp_deform_opacity.load_state_dict(original_gaussian.mlp_deform_opacity.state_dict())

    # unecessary to save
    decompressed_gaussians._rotation = original_gaussian._rotation
    decompressed_gaussians._rotation_dynamic = original_gaussian._rotation_dynamic
    decompressed_gaussians._opacity = original_gaussian._opacity
    decompressed_gaussians._opacity_dynamic = original_gaussian._opacity_dynamic
    return decompressed_gaussians



