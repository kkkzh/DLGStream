import os
import glob
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", default='data/dynerf/cut_roasted_beef', type=str)

    args = parser.parse_args()
    videos = glob.glob(os.path.join(args.datadir, "*.mp4"))
    videos = sorted(videos)

    # rename videos
    for video in videos:
        print(f"checking {video.split('/')[-1]}")
        os.system(f"ffmpeg -v error -i {video} -f null -")
