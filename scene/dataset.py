from torch.utils.data import Dataset
from scene.cameras import Camera, Camerass
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        dymask=None,
        hgopid=None
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        self.dymask = dymask
        self.hgopid = hgopid
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type == "PanopticSports":
            return self.dataset[index]
        elif self.dataset_type == "immersive":
            image, pose, time, camidx_, fisheye_mapper = self.dataset[index]
            R, T, focal_x, focal_y, cx, cy = pose

            if self.dymask is not None:
                # gopid = time // 0.5 + self.hgopid * 2
                # gopid = 0
                gopid = self.hgopid
                mask = self.dymask[gopid][camidx_]
            else:
                mask = None

            width = None
            height = None
            width_o, height_o = image.shape[2] * 2, image.shape[1] * 2
            FovX = focal2fov(focal_x, width_o)
            FovY = focal2fov(focal_y, height_o)

            return Camerass(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, fisheye_mapper=fisheye_mapper,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                          mask=mask, width=width, height=height, cxr=cx, cyr=cy)
        elif self.dataset_type == "technicolor":
            image, pose, time, camidx_ = self.dataset[index]
            R, T, focal_x, focal_y, cx, cy = pose

            if self.dymask is not None:
                # gopid = time // 0.5 + self.hgopid * 2
                # gopid = 0
                gopid = self.hgopid
                mask = self.dymask[gopid][camidx_]
            else:
                mask = None

            width = None
            height = None
            FovX = focal2fov(focal_x, image.shape[2])
            FovY = focal2fov(focal_y, image.shape[1])

            return Camerass(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, fisheye_mapper=None,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                          mask=mask, width=width, height=height, cxr=cx, cyr=cy)
        elif self.dataset_type == "multiview":
            # try:
            image, pose, time, camidx_ = self.dataset[index]
            R, T, focal_x, focal_y, cx, cy = pose

            if self.dymask is not None:
                gopid = self.hgopid
                mask = self.dymask[gopid][camidx_]
            else:
                mask = None

            width = None
            height = None
            FovX = focal2fov(focal_x, image.shape[2])
            FovY = focal2fov(focal_y, image.shape[1])
            # except:
            #     caminfo = self.dataset[index]
            #     image = caminfo.image
            #     R = caminfo.R
            #     T = caminfo.T
            #     FovX = caminfo.FovX
            #     FovY = caminfo.FovY
            #     time = caminfo.time
            #     width = caminfo.width
            #     height = caminfo.height

            return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                          mask=mask, width=width, height=height)
            # return Camerass(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, fisheye_mapper=None,
            #                 image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
            #                 mask=mask, width=width, height=height, cxr=cx, cyr=cy)
        else:
            try:
                image, lf_mask, w2c, time, camidx_ = self.dataset[index]
                R,T = w2c

                if self.dymask is not None:
                    gopid = self.hgopid
                    mask = self.dymask[gopid][camidx_]
                else:
                    mask = None

                depth = None
                lf_map = lf_mask[0]
                seg_map = lf_mask[1]

                width = None
                height = None
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                width = caminfo.width
                height = caminfo.height

                mask = caminfo.mask
                depth = None
                lf_map = None
                seg_map = None
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask, depth=depth, width=width, height=height, lf_map=lf_map, seg_map=seg_map)

    def __len__(self):
        
        return len(self.dataset)


class TimedFourDGSdataset(Dataset):
    def __init__(
            self,
            dataset,
            args,
            dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type

        self.set_timestamp(0)

    def set_timestamp(self, timestamp):
        self.dataset.set_timestamp(timestamp)

    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R, T = w2c
                mask = None

                width = None
                height = None
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                width = caminfo.width
                height = caminfo.height

                mask = caminfo.mask
            return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                          mask=mask, width=width, height=height)
        else:
            return self.dataset[index]

    def __len__(self):

        return len(self.dataset)
