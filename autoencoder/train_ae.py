import os
import sys
import argparse
import glob
import shutil


def do_system(arg, exit=True):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        if exit:
            sys.exit(err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, required=True)
    parser.add_argument("--scenes", nargs="+", type=str, default=['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_steak', 'flame_salmon'])
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--clip_dim", type=int, default=3)

    args = parser.parse_args()

    for scene in args.scenes:
        feature_path = os.path.join(args.feature_path, scene, 'cam00')

        training_cmd = (f"python train.py --lr 7e-4 "
                        f"--dataset_path {feature_path} "
                        f"--model_name {scene}_clip "
                        f"--feature_dims 512 "
                        f"--encoder_dims 256 128 64 32 {args.clip_dim} "
                        f"--decoder_dims 16 32 64 128 256 512 "
                        f"--hidden_dims {args.clip_dim} "
                        f"--language_name clip_features")
        do_system(training_cmd)

        inference_cmd = (f"python test.py --dataset_path {feature_path} "
                         f"--model_name {scene}_clip "
                         f"--feature_dims 512 "
                         f"--encoder_dims 256 128 64 32 {args.clip_dim} "
                         f"--decoder_dims 16 32 64 128 256 512 "
                         f"--hidden_dims {args.clip_dim} "
                         f"--language_name clip_features")

        do_system(inference_cmd)

        # link to dataset dir
        videos = glob.glob(os.path.join(args.feature_path, scene, "cam*"))
        videos = sorted(videos)
        print('camera number:', len(videos))

        for index, video_path in enumerate(videos):
            source_path = os.path.join(video_path, "clip_features-language_features_dim3")

            camid = video_path.split("/")[-1]
            target_path = os.path.join(args.target_path, scene, camid, "clip_features")
            print(f"link {source_path} to {target_path}")
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            os.symlink(source_path, target_path)

