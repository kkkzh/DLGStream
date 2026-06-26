import os
import shutil
import argparse
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    args = parser.parse_args()

    videos = glob.glob(os.path.join(args.source_path, "cam*"))
    videos = sorted(videos)
    print('camera number:', len(videos))

    for index, video_path in enumerate(videos):
        if index == 0:
            continue

        source_path = os.path.join(video_path, "clip_features-language_features_dim3")

        camid = video_path.split("/")[-1]
        target_path = os.path.join(args.target_path, camid, "clip_features")
        print(f"link {source_path} to {target_path}")
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        os.symlink(source_path, target_path)
