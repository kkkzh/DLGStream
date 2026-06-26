import os
import argparse
import sys
import json
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image as PILImage


def calc_std_gs(frame_path_root, std_path_root, reso, camid, frame_start=0, n_frame=300):
    cam = camid
    cam_id = 'cam' + str(cam).zfill(2)
    std_path = os.path.join(std_path_root, cam_id)

    frames = []
    for frame in tqdm(range(frame_start, frame_start + n_frame), total=n_frame):
        frame_id = str(frame).zfill(4)
        frame_path = os.path.join(frame_path_root, cam_id, 'images', frame_id + '.png')
        if os.path.exists(frame_path):
            frame = PILImage.open(frame_path).convert('RGB')
            if frame.width != reso[0] or frame.height != reso[1]:
                downsampled_frame = frame.resize(reso)
            else:
                downsampled_frame = frame
            frame = np.array(downsampled_frame, dtype=np.float32) / 255.
            frames.append(frame)
        # else:
        #     print(f"Not found image in {frame_path}")
        #     sys.exit(-1)

    if len(frames) == 0:
        print(f"camera {cam_id} not exists!")
        pass
    elif len(frames) > 0 and len(frames) != n_frame:
        print("load image error! The number of loaded image is unequal to n_frame")
        return
    else:
        frames = np.stack(frames, axis=0)
        frames = frames.std(axis=0).mean(axis=-1)
        frames = (cv2.GaussianBlur(frames, (31, 31), 0)).astype(np.float32)
        np.save(std_path + '_std.npy', frames)
        tqdm.write(std_path + '_std.npy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="", help="input path to the video")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=300)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument("--type", default="n3dv", help="n3dv, msth, meet, immersive")
    parser.add_argument('--threads', type=int, default=1, help="only support n3dv, meet, immersive")
    args = parser.parse_args()
    print(args.datadir)
    if args.type == 'n3dv':
        w, h = 2704, 2028
        downsample = 2
        calc_std = calc_std_gs
        camera_start_id = 0
        camera_end_id = 21
    elif args.type == 'meet':
        w, h = 1280, 720
        downsample = 1
        calc_std = calc_std_gs
        camera_start_id = 0
        camera_end_id = 13
    elif args.type == 'vru':
        w, h = 1920, 1080
        downsample = 1
        calc_std = calc_std_gs
        camera_start_id = 0
        camera_end_id = 34
    elif args.type == 'ImViD':
        w, h = 2656, 1494
        downsample = 1
        calc_std = calc_std_gs
        camera_start_id = 0
        camera_end_id = 38 + 1
    elif args.type == 'widerange4d':
        w, h = 2560, 1440
        downsample = 1
        calc_std = calc_std_gs
        camera_start_id = 0
        camera_end_id = 59 + 1
    else:
        raise NotImplementedError

    target_resolution = (w // downsample, h // downsample)
    step = args.interval

    if args.threads <= 1:
        for i in range(args.start, args.end, step):
            frame_start = i
            n_frames = step
            std_path = os.path.join(args.datadir, f'stds_2_{step}', str((i + 1) // step))
            os.makedirs(std_path, exist_ok=True)
            for idx in range(camera_start_id, camera_end_id, 1):
                calc_std(args.datadir, std_path, reso=target_resolution, camid=idx, frame_start=frame_start, n_frame=n_frames)
    else:
        tasks = []
        for i in range(args.start, args.end, step):
            frame_start = i
            n_frames = step
            std_path = os.path.join(args.datadir, f'stds_2_{step}', str((frame_start + 1) // step))
            os.makedirs(std_path, exist_ok=True)
            for idx in range(camera_start_id, camera_end_id, 1):
                tasks.append((args.datadir, std_path, target_resolution, idx, frame_start, n_frames))
        print(f'dispatch tasks!')
        pool = mp.Pool(processes=args.threads)
        pool.starmap(calc_std, tasks)
        pool.close()
        pool.join()
