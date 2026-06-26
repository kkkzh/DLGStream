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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.fdsd_gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset, TimedFourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, duration=[0, 300], skip=0, load_memory=False, timedordered=False, skip_init=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        # if os.path.exists(os.path.join(args.source_path, "sparse")):
        #     scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
        #     dataset_type="colmap"
        if args.dataset_type == 'msth':
            img_wh = [3840, 2160]
            pre_downsample = 2.0
            scene_info = sceneLoadTypeCallbacks["msth"](args.source_path, img_wh, pre_downsample, duration)
            dataset_type = "msth"
        elif args.dataset_type == 'n3dv':
            img_wh = [2704, 2028]
            pre_downsample = 2.0
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, img_wh, pre_downsample, args.feature_level, duration, skip=skip, load_memory=load_memory)
            dataset_type = "dynerf"
        elif args.dataset_type == 'meeting':
            img_wh = [1280, 720]
            pre_downsample = 1.0
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, img_wh, pre_downsample, args.feature_level, duration, load_memory)
            dataset_type = "dynerf"
        elif args.dataset_type == 'immersive':
            img_wh = [2560, 1920]
            pre_downsample = 1.0
            scene_info = sceneLoadTypeCallbacks["immersive"](args.source_path, img_wh, pre_downsample, duration)
            dataset_type = "immersive"
        elif args.dataset_type == 'technicolor':
            img_wh = [2048, 1088]
            pre_downsample = 1.0
            scene_info = sceneLoadTypeCallbacks["technicolor"](args.source_path, img_wh, pre_downsample, duration)
            dataset_type = "technicolor"
        elif args.dataset_type == 'widerange4d':
            img_wh = [2560, 1440]
            pre_downsample = 1.0
            scene_info = sceneLoadTypeCallbacks["multiview"](args.source_path, img_wh, pre_downsample, duration, load_memory)
            dataset_type = "multiview"
        elif args.dataset_type == 'ImViD':
            img_wh = [2656, 1494]
            pre_downsample = 1.0
            scene_info = sceneLoadTypeCallbacks["multiview"](args.source_path, img_wh, pre_downsample, duration, load_memory)
            dataset_type = "multiview"
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
        #     dataset_type="blender"
        # elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
        #     scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
        #     dataset_type="nerfies"
        # elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
        #     scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
        #     dataset_type="PanopticSports"
        # elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
        #     scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
        #     dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if not timedordered:
            # print("Loading Training Cameras")
            self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type, dymask=scene_info.mask, hgopid=duration[0]//300)
            # print("Loading Test Cameras")
            self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
            # print("Loading Video Cameras")
            self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)
        else:
            print("Loading Training Cameras")
            self.train_camera = scene_info.train_cameras
            print("Loading Test Cameras")
            self.test_camera = scene_info.test_cameras

        self.xyz_max = scene_info.point_cloud.points.max(axis=0)
        self.xyz_min = scene_info.point_cloud.points.min(axis=0)

        if self.loaded_iter or skip_init:
            # self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
            # self.gaussians.load_model(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)))
            self.point_cloud = scene_info.point_cloud
            pass
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getTimedTrainCameras(self):
        return TimedFourDGSdataset(self.train_camera, self.args, self.dataset_type)

    def getTimedTestCameras(self):
        return TimedFourDGSdataset(self.test_camera, self.args, self.dataset_type)

    def getVideoCameras(self, scale=1.0):
        return self.video_camera