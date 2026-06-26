#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int
    mask: dict = None


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset)), desc="format camera infos"):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            try:
                R, T = dataset.load_pose(idx)
                focal_x = dataset.focal[0]
                focal_y = dataset.focal[1]
                width_o, height_o = image.shape[2], image.shape[1]
            except:
                R, T, focal_x, focal_y, cx, cy = dataset.load_pose(idx)
                # width_o, height_o = image.shape[2] * 2, image.shape[1] * 2
                width_o, height_o = image.shape[2], image.shape[1]
            FovX = focal2fov(focal_x, width_o)
            FovY = focal2fov(focal_y, height_o)
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


def format_render_poses(poses,data_infos, eval):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)

        # width = 1832
        # height = 1920
        # sx = width / image.shape[2]
        # sy = height / image.shape[1]
        # FovX = focal2fov(data_infos.focal[0]*sx, width)
        # FovY = focal2fov(data_infos.focal[0]*sy, height)
        width = image.shape[2]
        height = image.shape[1]
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height,
                            time = time, mask=None))
    return cameras


def readdynerfInfo(datadir, img_wh, pre_downsample, feature_level, duration=[0, 300], skip=0, load_memory=False):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
        datadir,
        img_wh,
        "train",
        1.0,
        pre_downsample=pre_downsample,
        time_scale=1,
        feature_level=feature_level,
        eval_index=0,
        duration=duration,
        load_memory=load_memory,
        skip=skip
    )
    test_dataset = Neural3D_NDC_Dataset(
        datadir,
        img_wh,
        "test",
        1.0,
        pre_downsample=pre_downsample,
        time_scale=1,
        feature_level=feature_level,
        eval_index=0,
        duration=duration,
        load_memory=load_memory
    )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset, eval)
    # val_cam_infos = None
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # load dynamic mask
    dynamic_mask_path_parent = os.path.join(datadir, 'stds_2_300')
    if os.path.exists(dynamic_mask_path_parent):
        num = os.listdir(dynamic_mask_path_parent)
        dynamic_mask = {}
        for i in range(0, len(num), 1):
            dynamic_mask[i] = {}
            dynamic_mask_path = os.path.join(dynamic_mask_path_parent, str(i))
            if not os.path.exists(dynamic_mask_path):
                raise Exception(f"Not found std files in {dynamic_mask_path}")
            dynamic_mask_files = os.listdir(dynamic_mask_path)
            dynamic_mask_files = sorted(dynamic_mask_files)

            for file in dynamic_mask_files:
                try:
                    mask = np.load(os.path.join(dynamic_mask_path, file))
                    mask = torch.from_numpy(mask).float()
                    mask = mask > 0.03
                except:
                    print("loading dynamic std error", file)
                    quit()
                camera, frame_idx = file.split("_")
                camera_idx = camera[3:5]
                camera_idx = int(camera_idx)
                dynamic_mask[i][camera_idx] = mask
    else:
        print(f"[Warning] dynamic mask not loaded!")
        dynamic_mask = None

    # xyz = np.load
    pcd = fetchPly(ply_path)

    print("ply points,", pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=duration[1]-duration[0],
                           mask=dynamic_mask
                           )
    return scene_info


def readMultiViewInfo(datadir, img_wh, pre_downsample, duration=[0, 300], load_memory=False):
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.multiview_dataset import MultiView_Dataset
    train_dataset = MultiView_Dataset(
    datadir,
    img_wh,
    "train",
    1.0,
    pre_downsample=pre_downsample,
    time_scale=1,
    eval_index=[0],
    duration=duration,
    load_memory=load_memory
    )
    test_dataset = MultiView_Dataset(
    datadir,
    img_wh,
    "test",
    1.0,
    pre_downsample=pre_downsample,
    time_scale=1,
    eval_index=[0],
    duration=duration,
    load_memory=load_memory
    )
    train_cam_infos = format_infos(train_dataset,"train")
    # val_cam_infos = format_render_poses(test_dataset.val_poses, test_dataset, eval)
    val_cam_infos = None
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # load dynamic mask
    dynamic_mask_path_parent = os.path.join(datadir, f'stds_2_{duration[1]}')
    if os.path.exists(dynamic_mask_path_parent):
        num = os.listdir(dynamic_mask_path_parent)
        dynamic_mask = {}
        for i in range(0, len(num), 1):
            dynamic_mask[i] = {}
            dynamic_mask_path = os.path.join(dynamic_mask_path_parent, str(i))
            if not os.path.exists(dynamic_mask_path):
                raise Exception(f"Not found std files in {dynamic_mask_path}")
            dynamic_mask_files = os.listdir(dynamic_mask_path)
            dynamic_mask_files = sorted(dynamic_mask_files)

            for file in dynamic_mask_files:
                try:
                    mask = np.load(os.path.join(dynamic_mask_path, file))
                    mask = torch.from_numpy(mask).float()
                    mask = mask > 0.03
                except:
                    print("loading dynamic std error", file)
                    quit()
                camera, frame_idx = file.split("_")
                camera_idx = camera[3:5]
                camera_idx = int(camera_idx)
                dynamic_mask[i][camera_idx] = mask
    else:
        print(f"[Warning] dynamic mask not loaded!")
        dynamic_mask = None

    pcd = fetchPly(ply_path)
    print("ply points,", pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=duration[1]-duration[0],
                           mask=dynamic_mask
                           )
    return scene_info

def fetchPly_normlized(path, radius):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    positions = positions / radius
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam


def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    return cam_infos, max_time, scene_radius 


def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info


sceneLoadTypeCallbacks = {
    "dynerf" : readdynerfInfo,
    "PanopticSports" : readPanopticSportsinfos,
    "multiview": readMultiViewInfo,
}
