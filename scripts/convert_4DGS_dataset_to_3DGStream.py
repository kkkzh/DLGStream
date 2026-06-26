import argparse
import os
import glob
import shutil
from tqdm import tqdm

from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    videos = glob.glob(os.path.join(args.root_dir, "cam[0-9][0-9]"))
    videos = sorted(videos)

    for real_time in tqdm(range(args.start, args.end + 1), total=int(args.end - args.start)):
        if args.start == 0:
            timestamp = real_time
        else:
            timestamp = real_time - args.start  # enforce start from zero
        image_paths = []
        image_name = str(real_time).zfill(4)
        for index, video_path in enumerate(videos):
            image_path = os.path.join(video_path, "images", f"{image_name}.png")
            # try:  # check image
            #     img = Image.open(image_path)
            #     img = img.resize([2560, 1440], Image.LANCZOS)
            # except:
            #     print(f"{image_path}")
            image_paths.append(image_path)

        goal_dir = os.path.join(args.root_dir, "frames", f"{timestamp:04d}", 'images')
        if os.path.exists(goal_dir):
            shutil.rmtree(goal_dir)
        os.makedirs(goal_dir)
        undist_dir = os.path.join(args.root_dir, "frames_undist", f"{timestamp:04d}", 'images')
        if os.path.exists(undist_dir):
            shutil.rmtree(undist_dir)
        os.makedirs(undist_dir)

        image_name_list = []
        for index, image in enumerate(image_paths):
            image_name = image.split("/")[-1].split('.')
            cam_name = image.split("/")[-3]
            image_name[0] = cam_name
            image_name = ".".join(image_name)
            image_name_list.append(image_name)
            goal_path = os.path.join(goal_dir, image_name)
            os.symlink(image, goal_path)
            undist_path = os.path.join(undist_dir, image_name)
            os.symlink(image, undist_path)