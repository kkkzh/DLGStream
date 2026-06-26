import os
from argparse import ArgumentParser
import numpy as np
import ffmpeg
import subprocess
import json


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


def measure_key_frame_size(input_file):
    # probe = ffmpeg.probe(input_file, show_frames=None, select_streams='v:0', format='json')
    command = [
        'ffprobe',
        '-i', input_file,
        '-show_frames',
        '-select_streams', 'v:0',
        '-print_format', 'json',
    ]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print('ffprobe', out, err)
    probe = json.loads(out.decode('utf-8'))
    frames = probe['frames']
    key_frame_info = frames[0]
    pkt_size = key_frame_info['pkt_size']  # bytes
    return int(pkt_size) / 1024  # kB


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", default='/home/kzh/3DGS/Work3-results/Ours/cook_spinach', type=str)
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--nokey", action="store_true")
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--gopids", nargs="+", type=int, default=[])
    parser.add_argument("--qp", default=6, type=int)
    parser.add_argument("--type", type=str, default='3dgs', help='3dgs or hac')
    parser.add_argument("--language", action="store_true")
    args = parser.parse_args()

    if args.type == 'dec':
        import torch

        base_dir = os.path.join(args.datadir, f'gop1_v_256_2048', 'compression', 'best', 'png_quant')
        mlp_path = os.path.join(base_dir, 'mlps.pth')
        # mlp_path = '/home/kzh/3DGS/work4-results/Rebuttal/mlps.pth'
        mlps = torch.load(mlp_path)
        deformation_mlp = {
            'x_bound_min': mlps['x_bound_min'],
            'x_bound_max': mlps['x_bound_max'],
            'mlp_deform_xyz': mlps['mlp_deform_xyz'],
            'mlp_deform_cov': mlps['mlp_deform_cov'],
            'mlp_deform_color': mlps['mlp_deform_color'],
            'mlp_deform_opacity': mlps['mlp_deform_opacity'],
        }
        torch.save(deformation_mlp, os.path.join(base_dir, 'deformation_mlp.pth'))
        file_size = get_file_size_in_kB(os.path.join(base_dir, 'deformation_mlp.pth')) / args.gop
        print(f'deformation size: {file_size}kB')
        if 'ntc_mlp' in mlps.keys():
            ntc_mlp = {
                'ntc_mlp': mlps['ntc_mlp']
            }
            torch.save(ntc_mlp, os.path.join(base_dir,'ntc_mlp.pth'))
            file_size = get_file_size_in_kB(os.path.join(base_dir,'ntc_mlp.pth')) / args.gop
            print(f'ntc_mlp size: {file_size}kB')
        else:
            attr_mlp = {
                'mlp_opacity': mlps['mlp_opacity'],
                'mlp_cov': mlps['mlp_cov'],
                'mlp_color': mlps['mlp_color'],
                'mlp_grid': mlps['mlp_grid']
            }
            torch.save(attr_mlp, os.path.join(base_dir, 'attr_mlp.pth'))
            file_size = get_file_size_in_kB(os.path.join(base_dir, 'attr_mlp.pth')) / args.gop
            print(f'attr_mlp size: {file_size}kB')

    if len(args.gopids) == 0:
        gop_nums = 300 // args.gop
        start = 1 if args.nokey else 0
        gop_list = range(start, gop_nums, 1)
    else:
        gop_list = args.gopids
        if args.nokey:
            gop_list = [x for x in gop_list if x != 0]
    print(f"Processing gop", gop_list)

    if args.type == '3dgs':
        gaussian_attrs = ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest',
                          '_xyz_dynamic', '_scaling_dynamic', '_rotation_dynamic', '_opacity_dynamic', '_features_dc_dynamic', '_features_rest_dynamic']
        if args.language:
            gaussian_attrs.extend(['_language_feature', '_language_feature_dynamic'])
        gaussian_offset_attrs = ['_xyz_offset', '_xyz_dynamic_offset']
        mlps = ['mlp_deform', 'mlp_cov', 'mlp_opacity', 'mlp_color']

        total = []
        total_attr_size, total_video_size, total_mlp_size, total_residual_size = [], [], [], []
        for i in gop_list:
            if i == 0:
                # if args.postfix in ["mu_lerp", "rot_lerp"]:
                #     base_dir = os.path.join(args.datadir, f'gop0_{args.postfix}', 'compression', 'best', 'jxl_quant')
                # else:
                #     base_dir = os.path.join(args.datadir, 'gop0_1', 'compression', 'best', 'jxl_quant')
                base_dir = os.path.join(args.datadir, f'gop0_{args.postfix}', 'compression', 'best', 'jxl_quant')
                if not os.path.exists(base_dir):
                    base_dir = os.path.join(args.datadir, 'gop0_1', 'compression', 'best', 'jxl_quant')
            else:
                base_dir = os.path.join(args.datadir, f'gop{i}' if args.postfix is None else f'gop{i}_{args.postfix}', 'compression', 'best', 'jxl_quant')
            gop_size = 0
            gop_attr_size = 0
            gop_residual_size = 0
            gop_mlp_size = 0
            gop_video_size = 0
            # temporal_feats_path = os.path.join(args.datadir, f'gop{i}', "jxl_quant_sh", f'_point_feats.mp4' if args.qp == 20 else f'_point_feats_{args.qp}.mp4')
            temporal_feats_path = os.path.join(base_dir, f'_point_feats.mp4' if args.qp == 6 else f'_point_feats_{args.qp}.mp4')
            file_size = get_file_size_in_kB(temporal_feats_path) / args.gop
            gop_video_size += file_size
            gop_size += file_size
            gop_attr_size = measure_key_frame_size(temporal_feats_path)
            if i == 0:
                for file in gaussian_attrs:
                    # file_path = os.path.join(args.datadir, f'gop{i}', "jxl_quant_sh", f'{file}.jxl')
                    file_path = os.path.join(base_dir, f'{file}.jxl')
                    attr_file_size = get_file_size_in_kB(file_path) / args.gop
                    gop_attr_size += attr_file_size
                    gop_size += attr_file_size
            else:
                for file in gaussian_offset_attrs:
                    if args.postfix == "wo_st_refine" and file == '_xyz_offset':
                        continue

                    file_path = os.path.join(base_dir, f'{file}.jxl')
                    residual_attr_file_size = get_file_size_in_kB(file_path) / args.gop
                    gop_attr_size += residual_attr_file_size
                    gop_size += residual_attr_file_size
                file_path = os.path.join(base_dir, 'offset_compressed.pkl')
                residual_attr_file_size = get_file_size_in_kB(file_path) / args.gop
                gop_residual_size += residual_attr_file_size
                gop_size += residual_attr_file_size
            for file in mlps:
                file_path = os.path.join(base_dir, f'{file}.pth')
                file_size = get_file_size_in_kB(file_path) / args.gop
                gop_mlp_size += file_size
                gop_size += file_size
            compress_file_path = os.path.join(base_dir, f'compression_info.csv')
            file_size = get_file_size_in_kB(compress_file_path) / args.gop
            gop_size += file_size
            total.append(gop_size)
            total_attr_size.append(gop_attr_size)
            total_video_size.append(gop_video_size)
            total_mlp_size.append(gop_mlp_size)
            total_residual_size.append(gop_residual_size)
        save_path = os.path.join(args.datadir, "experiments", f'results' if args.postfix is None else f'results_{args.postfix}', f'metrics_qp{args.qp}')
        os.makedirs(save_path, exist_ok=True)
        size_ = np.mean(np.array(total))
        np.save(os.path.join(save_path, 'size.npy'), size_)
        print(f'Average frame size: {size_}kB')  # f'Key frame size: {np.mean(np.array(key_frame)) / 1024}MB

        avg_video_size = np.mean(np.array(total_video_size))
        avg_mlp_size = np.mean(np.array(total_mlp_size))
        avg_residual_size = np.mean(np.array(total_residual_size))
        avg_attr_size = size_ - avg_video_size - avg_mlp_size - avg_residual_size
        print(f'Gaussian attr size: {avg_attr_size}kB, temporal feature size: {avg_video_size}, mlp size: {avg_mlp_size}, GOP residual size: {avg_residual_size}')
    elif args.type == 'hac':
        total = []
        total_attr_size, total_video_size, total_mlp_size, total_residual_size = [], [], [], []
        for i in gop_list:
            gop_size = 0
            gop_attr_size = 0
            gop_residual_size = 0
            gop_mlp_size = 0
            gop_video_size = 0

            if i == 0:
                attr_dir = os.path.join(args.datadir, f'gop0_cubic', 'compression', 'best', 'bitstreams')
            else:
                attr_dir = os.path.join(args.datadir, f'gop{i}_{args.postfix}', 'compression', 'best', 'bitstreams')
            attr_size = get_directory_size_in_kB(attr_dir) / args.gop
            gop_size += attr_size

            if i == 0:
                gop_attr_size += attr_size
            else:
                gop_residual_size += attr_size

            if i == 0:
                base_dir = os.path.join(args.datadir, f'gop0_cubic', 'compression', 'best', 'png_quant')
            else:
                base_dir = os.path.join(args.datadir, f'gop{i}_{args.postfix}', 'compression', 'best', 'png_quant')

            if i == 0:
                video_base_dir = os.path.join(args.datadir, f'gop1_{args.postfix}', 'compression', 'best', 'png_quant')
            else:
                video_base_dir = base_dir
            temporal_feats_path = os.path.join(video_base_dir, f'_temporal_feat.mp4' if args.qp == 6 else f'_temporal_feat_{args.qp}.mp4')
            file_size = get_file_size_in_kB(temporal_feats_path) / args.gop
            gop_video_size += file_size
            gop_size += file_size

            temporal_feats_path = os.path.join(base_dir, f'mlps.pth')
            file_size = get_file_size_in_kB(temporal_feats_path) / args.gop
            gop_mlp_size += file_size
            gop_size += file_size

            compress_file_path = os.path.join(base_dir, f'compression_info.csv')
            file_size = get_file_size_in_kB(compress_file_path) / args.gop
            gop_size += file_size

            total.append(gop_size)
            total_attr_size.append(gop_attr_size)
            total_video_size.append(gop_video_size)
            total_mlp_size.append(gop_mlp_size)
            total_residual_size.append(gop_residual_size)

        save_path = os.path.join(args.datadir, "experiments", f'results' if args.postfix is None else f'results_{args.postfix}', f'metrics_qp{args.qp}')
        os.makedirs(save_path, exist_ok=True)
        size_ = np.mean(np.array(total))
        np.save(os.path.join(save_path, 'size.npy'), size_)
        print(f'Average frame size: {size_}kB')

        avg_attr_size = np.mean(np.array(total_attr_size))
        avg_video_size = np.mean(np.array(total_video_size))
        avg_mlp_size = np.mean(np.array(total_mlp_size))
        avg_residual_size = np.mean(np.array(total_residual_size))
        print(f'Gaussian attr size: {avg_attr_size:.5f}kB, temporal feature size: {avg_video_size:.5f}, mlp size: {avg_mlp_size:.5f}, GOP residual size: {avg_residual_size:.5f}')