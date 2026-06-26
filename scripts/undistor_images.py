import os
import argparse
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="")
    args = parser.parse_args()

    # re-organize dir
    videos = glob.glob(os.path.join(args.root_dir, "cam[0-9][0-9]"))
    videos = sorted(videos)
    if not os.path.exists(os.path.join(args.root_dir, "frames")):
        for timestamp in range(0, 300):
            image_paths = []
            image_name = str(timestamp).zfill(4)
            for index, video_path in enumerate(videos):
                image_path = os.path.join(video_path, "images", f"{image_name}.png")
                image_paths.append(image_path)

            goal_dir = os.path.join(args.root_dir, "frames", f"{timestamp:04d}")
            os.makedirs(goal_dir)

            image_name_list = []
            for index, image in enumerate(image_paths):
                image_name = image.split("/")[-1].split('.')
                cam_name = image.split("/")[-3]
                image_name[0] = cam_name
                image_name = ".".join(image_name)
                image_name_list.append(image_name)
                goal_path = os.path.join(goal_dir, image_name)
                os.symlink(image, goal_path)

    # image undistorter
    sparse_dir = os.path.join(args.root_dir, "colmap/time_0/distorted", "sparse", "0")
    for timestamp in range(0, 300):
        goal_dir = os.path.join(args.root_dir, "frames", f"{timestamp:04d}")
        frame_dir = os.path.join(args.root_dir, 'frames_undist', f"{timestamp:04d}")
        os.makedirs(frame_dir)
        os.system(f"colmap image_undistorter --image_path {goal_dir} --input_path {sparse_dir} --output_path {frame_dir} --output_type COLMAP --min_scale 1 --max_scale 1")