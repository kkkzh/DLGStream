from argparse import ArgumentParser
import sys
import os
import glob
import json
from multiprocessing import Pool

import cv2
from PIL import Image
import numpy as np
import tqdm


sys.path.append('../scene')
try:
    from neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    from msth_dataset_loader import MSTH_Dataset
except:
    Neural3D_NDC_Dataset = None
    print(f"Import neural_3D_dataset_NDC Error! can not deal with n3dv and meet datasets")


# https://github.com/Synthesis-AI-Dev/fisheye-distortion
def getdistortedflow(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                     mode: str, crop_output: bool = True,
                     crop_type: str = "corner", scale: float = 2, cxoffset=None, cyoffset=None, knew=None):
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    elif len(imshape) == 2:
        h, w = imshape
        chan = 1
    else:
        raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')

    imdtype = img.dtype
    dstW = int(w)
    dstH = int(h)

    # Get array of pixel co-ords
    xs = np.arange(dstW)
    ys = np.arange(dstH)

    xs = xs  # - 0.5 # + cxoffset / 2
    ys = ys  # - 0.5 # + cyoffset / 2

    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2), in undistorted image coordiante

    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff, None, knew)  # shape: (N, 1, 2)

    undistorted_px = undistorted_px.reshape((dstH, dstW, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    undistorted_px[:, :, 0] = undistorted_px[:, :, 0]  # +  0.5*cyoffset #- 0.25*cyoffset #orginalx (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1]  # +  0.5*cyoffset #- 0.25*cxoffset #orginaly (0, 1)

    undistorted_px[:, :, 0] = undistorted_px[:, :, 0] / (h - 1)  # (h-1) #orginalx (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1] / (w - 1)  # (w-1) #orginaly (0, 1)

    undistorted_px = 2 * (undistorted_px - 0.5)  # to -1 to 1 for gridsample

    undistorted_px[:, :, 0] = undistorted_px[:, :, 0]  # orginalx (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1]  # orginaly (0, 1)

    undistorted_px = undistorted_px[:, :, ::-1]  # yx to xy for grid sample
    return undistorted_px


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", default='data/dynerf/cut_roasted_beef', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=300)
    parser.add_argument("--type", default='n3dv', type=str, help="n3dv, msth, meet, immersive")
    parser.add_argument("--threads", default=1, type=int, help="multi-thread process for immersive")
    args = parser.parse_args()

    if args.type == 'n3dv':
        Dataset = Neural3D_NDC_Dataset
        img_wh = [2704, 2028]
        pre_downsample = 2.0
    elif args.type == 'meet':
        Dataset = Neural3D_NDC_Dataset
        img_wh = [1280, 720]
        pre_downsample = 1.0
        # rename videos
        videos = glob.glob(os.path.join(args.datadir, '*.mp4'))
        videos = sorted(videos)
        for video in videos:
            videos_full_name = video.split('/')[-1]
            videos_name = videos_full_name.split('.')[0]
            if '_' in videos_name:
                camid = videos_name.split('_')[-1]
                camid = str(camid).zfill(2)
                new_video = os.path.join(args.datadir, f'cam{camid}.mp4')
                print(new_video)
                os.rename(video, new_video)
    else:
        raise NotImplementedError
    train_dataset = Dataset(args.datadir, img_wh, "train", 1.0, pre_downsample, time_scale=1,
                                         scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0], eval_index=0, duration=[args.start, args.end])
    test_dataset = Dataset(args.datadir, img_wh, "test", 1.0, pre_downsample, time_scale=1,
                                        scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0], eval_index=0, duration=[args.start, args.end])
