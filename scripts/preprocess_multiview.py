import argparse
import os
from tqdm import tqdm
import cv2
from PIL import Image
from multiprocessing import Pool, cpu_count


def process_single_video(args):
    video_path, index, duration, downsample, img_wh, base_path = args
    countss = duration[1] - duration[0]
    count = 0
    video_images_path = os.path.join(base_path, f"cam{index:02d}")
    cam_id = index
    image_path = os.path.join(video_images_path, "images")
    print(f"Processing cam{cam_id}: saved path {image_path}")
    # 检查目标文件夹是否已存在，如果存在则跳过（可以根据需求调整，例如强制覆盖或跳过）
    # if os.path.exists(image_path) and os.listdir(image_path):
    #     print(f"Directory {image_path} already exists and is not empty. Skipping.")
    #     return
    progress_bar = tqdm(range(duration[0], duration[1]), desc=f"extract cam{cam_id}", leave=False) # leave=False 避免进度条在多进程中混乱
    os.makedirs(image_path, exist_ok=True)
    this_count = 0
    video_frames = cv2.VideoCapture(video_path)
    if not video_frames.isOpened():
        print(f"Error opening video file {video_path} for cam{cam_id}")
        return
    while video_frames.isOpened():
        ret, video_frame = video_frames.read()
        if this_count >= countss:
            break
        if ret:
            image_name_path = os.path.join(image_path, "%04d.png" % count)
            # 可以在这里添加检查，如果图片已经存在则跳过
            if os.path.exists(image_name_path):
                count += 1
                this_count += 1
                progress_bar.update(1)
                continue
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_frame = Image.fromarray(video_frame)
            if downsample != 1.0:
                img = video_frame.resize(img_wh, Image.LANCZOS)
            else:
                img = video_frame
            img.save(image_name_path)
            progress_bar.update(1)
            count += 1
            this_count += 1
        else:
            break
    video_frames.release() # 释放视频对象
    progress_bar.close()
    print(f"Finished processing cam{cam_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--reso", type=int, required=True, nargs="+")
    parser.add_argument("--downsample", type=int, required=True)
    parser.add_argument("--duration", type=int, nargs="+", default=[0, 300])
    parser.add_argument("--threads", type=int, default=-1)
    args = parser.parse_args()

    duration = args.duration
    downsample = args.downsample
    img_wh = args.reso
    start = args.start

    videos_to_process = []
    for i in range(args.start, args.end+1, 1):
        video_id = f'{i:02d}'
        base_path = args.root_dir
        video_filename = os.path.join(base_path, f"cam{video_id}.mp4")
        if os.path.exists(video_filename):
            if start != 0:
                index = int(i - start)
            else:
                index = i
            videos_to_process.append((video_filename, index, duration, downsample, img_wh, base_path))
        else:
            print(f"Warning: Video file {video_filename} not found. Skipping.")

    # 使用 multiprocessing.Pool 来并行执行任务
    if args.threads == -1:
        num_processes = cpu_count() - 2
    else:
        num_processes = args.threads
    with Pool(processes=num_processes) as pool:
        # map 方法会将 iterable 中的每个元素作为参数传递给 process_single_video 函数
        # 它会按照输入顺序返回结果，但执行是并行的
        pool.map(process_single_video, videos_to_process)

    # for index, video_path in enumerate(videos):
    #     count = 0
    #     video_images_path = os.path.join(base_path, f"cam{index:02d}")
    #     cam_id = index
    #     image_path = os.path.join(video_images_path, "images")
    #     print(f"saved path {image_path}")
    #     if not os.path.exists(image_path):
    #         progress_bar = tqdm(range(duration[0], duration[1]), desc=f"extract cam{cam_id}")
    #         os.makedirs(image_path, exist_ok=True)
    #         this_count = 0
    #         video_frames = cv2.VideoCapture(video_path)
    #         if not video_frames.isOpened():
    #             print(f"Error opening video file {video_path}")
    #             break
    #         while video_frames.isOpened():
    #             ret, video_frame = video_frames.read()
    #             if this_count >= countss: break
    #             if ret:
    #                 image_name_path = os.path.join(image_path, "%04d.png" % count)
    #                 if not os.path.exists(image_name_path):
    #                     video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    #                     video_frame = Image.fromarray(video_frame)
    #                     if downsample != 1.0:
    #                         img = video_frame.resize(img_wh, Image.LANCZOS)
    #                     else:
    #                         img = video_frame
    #
    #                     img.save(os.path.join(image_path, "%04d.png" % count))
    #                 progress_bar.update(1)
    #                 # img = transform(img)
    #                 count += 1
    #                 this_count += 1
    #             else:
    #                 break
    #         progress_bar.close()