import os
import sys
import argparse


def do_system(arg, exit=True):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        if exit:
            sys.exit(err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--scenes", nargs="+", type=str, default=['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_steak', 'flame_salmon', 'sear_steak'])
    parser.add_argument("--configs", type=str, required=True)
    parser.add_argument("--compre_config", type=str, default="fsd_gop60")
    parser.add_argument("--work_path", type=str, default="/home/kzh/runtime/4DLangSplat/sog")
    parser.add_argument("--coarse_postfix", type=str, default=None)
    parser.add_argument("--coarse_full", type=int, default=1)
    parser.add_argument("--gop0_postfix", type=str, default=None)
    parser.add_argument("--postfix", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--language", action="store_true")
    parser.add_argument("--compres_thres", nargs="+", type=float, default=[33.1])
    parser.add_argument("--qp", type=int, default=6)
    parser.add_argument("--gopids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--type", type=str, default='3dgs', help="3dgs / hac")
    parser.add_argument("--lmbda", type=float, default=0.001, help='only for hac')
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    # config_path = f'arguments/n3dv/{args.configs}.py'
    dataset_path = args.dataset
    work_path = args.work_path
    compres_thres = args.compres_thres
    assert len(compres_thres) == len(args.scenes)
    config_path = args.configs
    compression_config = args.compre_config

    if args.type == '3dgs':
        main_file = 'train_fdsd'
    elif args.type == 'hac':
        main_file = 'train_hac'
        lmbda = args.lmbda
        assert 'hac' in work_path
        assert 'hac' in config_path
        assert 'hac' in compression_config
    else:
        raise NotImplementedError

    gopids = " ".join([str(j) for j in args.gopids])

    for idx, scene in enumerate(args.scenes):
        source_path = os.path.join(dataset_path, scene)
        model_path = os.path.join(work_path, scene)
        if args.coarse:
            checkpoint_path = os.path.join(work_path, scene, f"coarse" if args.coarse_postfix is None else f"coarse_{args.coarse_postfix}")
            if os.path.exists(os.path.join(checkpoint_path, "coarse_sd.pth")):
                print(f'coarse model trained, skip!')
                break
            training_cmd = (f"CUDA_VISIBLE_DEVICES={args.gpuid} python {main_file}.py -s {source_path} "
                      f"--configs {config_path} "
                      f"--compre_config {compression_config} "
                      f"--checkpoint_path {checkpoint_path} "
                      f"--expname {model_path} --coarse ")
            if args.coarse_full:
                training_cmd = training_cmd + f"--coarse_full "
            if args.coarse_postfix is not None:
                training_cmd = training_cmd + f"--postfix {args.coarse_postfix} "
            if args.language:
                training_cmd = training_cmd + f"--language "
            if args.type == 'hac':
                training_cmd = training_cmd + f"--lmbda {lmbda} "
            do_system(training_cmd)
        else:
            for _, gopid in enumerate(args.gopids):
                if gopid == 0:
                    checkpoint_path = os.path.join(model_path, f"coarse" if args.coarse_postfix is None else f"coarse_{args.coarse_postfix}")
                    # train coarse model if coarse is None
                    if not os.path.exists(os.path.join(checkpoint_path, "coarse_sd.pth")):
                        checkpoint_path = os.path.join(work_path, scene, f"coarse" if args.coarse_postfix is None else f"coarse_{args.coarse_postfix}")
                        training_cmd = (f"CUDA_VISIBLE_DEVICES={args.gpuid} python {main_file}.py -s {source_path} "
                                        f"--configs {config_path} "
                                        f"--compre_config {compression_config} "
                                        f"--checkpoint_path {checkpoint_path} "
                                        f"--expname {model_path} --coarse ")
                        if args.coarse_full:
                            training_cmd = training_cmd + f"--coarse_full "
                        if args.coarse_postfix is not None:
                            training_cmd = training_cmd + f"--postfix {args.coarse_postfix} "
                        if args.language:
                            training_cmd = training_cmd + f"--language "
                        if args.skip != 0:
                            training_cmd = training_cmd + f"--skip {args.skip} "
                        if args.type == 'hac':
                            training_cmd = training_cmd + f"--lmbda {lmbda} "
                        do_system(training_cmd)
                # elif gopid == 1:
                #     checkpoint_path = os.path.join(model_path, f"gop0_{args.postfix}/compression/best/jxl_quant")
                else:
                    if args.type == '3dgs':
                        if 'pt' in args.postfix:
                            checkpoint_path = os.path.join(model_path, f"gop0_{args.coarse_postfix}/compression/best/jxl_quant")  # simulate parallel training
                        else:
                            if gopid == 1 and args.gop0_postfix is not None:
                                checkpoint_path = os.path.join(model_path, f"gop0_{args.gop0_postfix}/compression/best/jxl_quant")
                            else:
                                checkpoint_path = os.path.join(model_path, f"gop{gopid - 1}_{args.postfix}/compression/best/jxl_quant")

                    elif args.type == 'hac':
                        if 'pt' in args.postfix:
                            checkpoint_path = os.path.join(model_path, f"gop0_{args.coarse_postfix}/compression/best/png_quant")  # simulate parallel training
                        else:
                            if gopid == 1 and args.gop0_postfix is not None:
                                checkpoint_path = os.path.join(model_path, f"gop0_{args.postfix}/compression/best/png_quant")
                            else:
                                checkpoint_path = os.path.join(model_path, f"gop{gopid - 1}_{args.postfix}/compression/best/png_quant")

                    else:
                        raise NotImplementedError

                output_path = os.path.join(model_path, f"gop{gopid}_{args.postfix}")

                # if os.path.exists(output_path):
                #     print(f'gop {gopid} trained, skip!')
                #     continue
                if gopid > 0 and args.type == 'hac':
                    if 'n3dv' in args.configs:
                        config_path = 'arguments/n3dv/default_hac_following.py'
                    elif 'meet' in args.configs:
                        config_path = 'arguments/meeting/default_hac_following.py'
                    else:
                        config_path = 'arguments/others/default_hac.py'
                training_cmd = (f"CUDA_VISIBLE_DEVICES={args.gpuid} python {main_file}.py -s {source_path} "
                                f"--configs {config_path} "
                                f"--compre_config {compression_config} "
                                f"--checkpoint_path {checkpoint_path} "
                                f"--expname {model_path} "
                                f"--gopids {gopid} "
                                f"--compres_thres {compres_thres[idx]} "
                                f"--qp {args.qp} --gop {args.gop} ")
                if args.postfix is not None:
                    training_cmd = training_cmd + f"--postfix {args.postfix} "
                if args.language:
                    training_cmd = training_cmd + f"--language "
                if args.skip != 0:
                    training_cmd = training_cmd + f"--skip {args.skip} "
                if args.type == 'hac':
                    training_cmd = training_cmd + f"--lmbda {lmbda} "
                if args.dynamic:
                    training_cmd = training_cmd + f"--dynamic"
                do_system(training_cmd)
