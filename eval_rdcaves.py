import os
import sys
from argparse import ArgumentParser

import numpy as np


def do_system(arg, exit=True):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        if exit:
            sys.exit(err)

def eval_n3dv_hac(args, rds, n3dv):
    for scene in n3dv:
        for qp in rds:
            eval_cmd = (f"python test_hac.py -s /home/kzh/mountpoint/dataset/N3DV/hexgs/{scene} --configs arguments/n3dv/default_hac.py "
                        f"--compre_config hac_gop60.yaml --checkpoint {args.datadir}/{scene} --postfix {args.postfix} --qp {qp} "
                        f" --gopids 0 1 2 3 4 --disable_ssim --disable_lpips")
            # do_system(eval_cmd)
            size_cmd = f"python read_size.py --datadir {args.datadir}/{scene} --type hac --postfix {args.postfix} --gopids 0 1 2 3 4 --qp {qp} "
            # do_system(size_cmd)

    # calc
    print(f"N3DV dataset")
    for qp in rds:
        total_psnr = []
        total_size = []
        for scene in n3dv:
            # psnr_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "psnr.npy" if qp == 6 else f"psnr_{qp}.npy"))
            psnr_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "psnr.npy"))
            size_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "size.npy"))
            total_psnr.append(psnr_)
            total_size.append(size_)
        total_psnr = np.array(total_psnr)
        total_size = np.array(total_size)
        print(f"qp = {qp}, size = {total_size.mean()}, psnr = {total_psnr.mean()}")


def eval_n3dv_3dgs(args, rds, n3dv):
    for scene in n3dv:
        for qp in rds:
            eval_cmd = (f"python test_offset.py -s /home/kzh/mountpoint/dataset/N3DV/hexgs/{scene} --configs arguments/n3dv/default.py "
                        f"--compre_config fsd_gop60.yaml --checkpoint {args.datadir}/{scene} --postfix {args.postfix} --qp {qp} "
                        f" --gopids 0 1 2 3 4 --all --disable_ssim --disable_lpips")
            if args.language:
                eval_cmd = eval_cmd + f"--language "
            # do_system(eval_cmd)
            size_cmd = f"python read_size.py --datadir {args.datadir}/{scene} --type 3dgs --postfix {args.postfix} --gopids 0 1 2 3 4 --qp {qp} "
            if args.language:
                size_cmd = size_cmd + f"--language "
            # do_system(size_cmd)

    # calc
    print(f"N3DV dataset")
    for qp in rds:
        total_psnr = []
        total_size = []
        for scene in n3dv:
            psnr_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "psnr.npy"))
            size_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "size.npy"))
            total_psnr.append(psnr_.mean())
            total_size.append(size_)
        total_psnr = np.array(total_psnr)
        total_size = np.array(total_size)
        print(f"qp = {qp}, size = {total_size.mean()}, psnr = {total_psnr.mean()}")


def eval_meet_hac(args, rds, meet):
    for scene in meet:
        for qp in rds:
            eval_cmd = (f"python test_hac.py -s /home/kzh/mountpoint/dataset/MeetRoom/{scene} --configs arguments/meeting/default_hac.py "
                        f"--compre_config hac_gop60.yaml --checkpoint {args.datadir}/{scene} --postfix {args.postfix} --qp {qp} "
                        f" --gopids 0 1 2 3 4 --disable_ssim --disable_lpips")
            if args.language:
                eval_cmd = eval_cmd + f"--language "
            do_system(eval_cmd)
            size_cmd = f"python read_size.py --datadir {args.datadir}/{scene} --type hac --postfix {args.postfix} --gopids 0 1 2 3 4 --qp {qp}"
            if args.language:
                size_cmd = size_cmd + f"--language "
            do_system(size_cmd)

    print(f"MeetRoom dataset")
    for qp in rds:
        total_psnr = []
        total_size = []
        for scene in meet:
            psnr_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "psnr.npy"))
            size_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "size.npy"))
            total_psnr.append(psnr_)
            total_size.append(size_)
        total_psnr = np.array(total_psnr)
        total_size = np.array(total_size)
        print(f"qp = {qp}, size = {total_size.mean()}, psnr = {total_psnr.mean()}")


def eval_meet_3dgs(args, rds, meet):
    for scene in meet:
        for qp in rds:
            eval_cmd = (f"python test_offset.py -s /home/kzh/mountpoint/dataset/MeetRoom/{scene} --configs arguments/meeting/default.py "
                        f"--compre_config fsd_gop60.yaml --checkpoint {args.datadir}/{scene} --postfix {args.postfix} --qp {qp} "
                        f" --gopids 0 1 2 3 4 --all --disable_ssim --disable_lpips")
            # do_system(eval_cmd)
            size_cmd = f"python read_size.py --datadir {args.datadir}/{scene} --type 3dgs --postfix {args.postfix} --gopids 0 1 2 3 4 --qp {qp} "
            # do_system(size_cmd)

    # calc
    print(f"MeetRoom dataset")
    for qp in rds:
        total_psnr = []
        total_size = []
        for scene in meet:
            # psnr_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "psnr.npy" if qp == 6 else f"psnr_{qp}.npy"))
            psnr_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "psnr.npy"))
            size_ = np.load(os.path.join(f"{args.datadir}/{scene}/experiments/results_{args.postfix}", f'metrics_qp{qp}', "size.npy"))
            total_psnr.append(psnr_)
            total_size.append(size_)
        total_psnr = np.array(total_psnr)
        total_size = np.array(total_size)
        print(f"qp = {qp}, size = {total_size.mean()}, psnr = {total_psnr.mean()}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--type", type=str, default='3dgs')
    parser.add_argument("--language", action="store_true")
    args = parser.parse_args()

    rds = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    n3dv = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon', 'flame_steak', 'sear_steak']
    # n3dv = ['cut_roasted_beef', 'flame_salmon']
    # for scene in n3dv:
    #     for qp in rds:
    #         eval_cmd = (f"python test_hac.py -s /home/kzh/mountpoint/dataset/N3DV/hexgs/{scene} --configs arguments/n3dv/default_hac.py "
    #                     f"--compre_config hac_gop60.yaml --checkpoint output/4DLangSplat/hac/{scene} --postfix default1 --qp {qp} "
    #                     f" --gopids 0 1 2 3 4 --disable_ssim --disable_lpips")
    #         # do_system(eval_cmd)
    #         size_cmd = f"python read_size.py --datadir output/4DLangSplat/hac/{scene} --type hac --postfix default1 --gopids 0 1 2 3 4 --qp {qp} "
    #         # do_system(size_cmd)
    #
    # # calc
    # print(f"N3DV dataset")
    # for qp in rds:
    #     total_psnr = []
    #     total_size = []
    #     for scene in n3dv:
    #         # psnr_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "psnr.npy" if qp == 6 else f"psnr_{qp}.npy"))
    #         psnr_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "psnr.npy"))
    #         size_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "size.npy"))
    #         total_psnr.append(psnr_)
    #         total_size.append(size_)
    #     total_psnr = np.array(total_psnr)
    #     total_size = np.array(total_size)
    #     print(f"qp = {qp}, size = {total_size.mean()}, psnr = {total_psnr.mean()}")
    if args.type == '3dgs':
        eval_n3dv_3dgs(args, rds, n3dv)
    elif args.type == 'hac':
        eval_n3dv_hac(args, rds, n3dv)

    meet = ['discussion', 'trimming', 'vrheadset']
    # for scene in meet:
    #     for qp in rds:
    #         eval_cmd = (f"python test_hac.py -s /home/kzh/mountpoint/dataset/MeetRoom/{scene} --configs arguments/meeting/default_hac.py "
    #                     f"--compre_config hac_gop60.yaml --checkpoint output/4DLangSplat/hac/{scene} --postfix default1 --qp {qp} "
    #                     f" --gopids 0 1 2 3 4 --disable_ssim --disable_lpips")
    #         # do_system(eval_cmd)
    #
    #         size_cmd = f"python read_size.py --datadir output/4DLangSplat/hac/{scene} --type hac --postfix default1 --gopids 0 1 2 3 4 --qp {qp}"
    #         # do_system(size_cmd)
    #
    # print(f"MeetRoom dataset")
    # for qp in rds:
    #     total_psnr = []
    #     total_size = []
    #     for scene in meet:
    #         # psnr_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "psnr.npy" if qp == 6 else f"psnr_{qp}.npy"))
    #         psnr_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "psnr.npy"))
    #         size_ = np.load(os.path.join(f"output/4DLangSplat/hac/{scene}/experiments/results_default1", f'metrics_qp{qp}', "size.npy"))
    #         total_psnr.append(psnr_)
    #         total_size.append(size_)
    #     total_psnr = np.array(total_psnr)
    #     total_size = np.array(total_size)
    #     print(f"qp = {qp}, size = {total_size.mean()}, psnr = {total_psnr.mean()}")
    if args.type == '3dgs':
        eval_meet_3dgs(args, rds, meet)
    elif args.type == 'hac':
        eval_meet_hac(args, rds, meet)
