import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir, skip_test=True):
        # data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        # total_rows = 0
        # for i in tqdm(range(len(data_names))):
        #     features = np.load(data_names[i], mmap_mode='r')
        #     total_rows += features.shape[0]

        # first_sample = np.load(data_names[0], mmap_mode='r')
        # self.data = np.empty((total_rows, first_sample.shape[1]), dtype=first_sample.dtype)

        # current_idx = 0
        # self.data_dic = {}
        # for i in tqdm(range(len(data_names))):
        #     features = np.load(data_names[i])
        #     name = data_names[i].split('/')[-1].split('.')[0]
        #     rows = features.shape[0]
        #     self.data_dic[name] = rows

        #     self.data[current_idx:current_idx + rows] = features
        #     current_idx += rows
        root_dir = os.path.dirname(os.path.dirname(data_dir))
        feature_type = data_dir.split("/")[-1]
        videos = glob.glob(os.path.join(root_dir, "cam*"))
        videos = sorted(videos)

        feature_paths = []
        for index, video_path in enumerate(videos):
            if (index == 0) and skip_test:
                continue

            camera_path = video_path.split('.')[0]
            feature_dir = os.path.join(camera_path, feature_type)
            feature_path = os.listdir(feature_dir)
            for idx, path in enumerate(feature_path):
                if '_f.npy' in path:
                    feature_paths.append(os.path.join(feature_dir, path))

        assert len(feature_paths) > 0, print(len(feature_paths))
        total_rows = 0
        features = []
        self.data_dic = {}
        for path in tqdm(feature_paths, total=len(feature_paths)):
            feature = np.load(path, mmap_mode="r")
            features.append(feature)
            rows = feature.shape[0]
            total_rows += rows

            match = re.search(r"cam\d{2}", path)
            cam_id = match.group(0)
            name = path.split("/")[-1].split(".")[0]

            key = cam_id + "-" + name
            self.data_dic[key] = rows

        first_sample = np.load(feature_paths[0], mmap_mode="r")
        self.data = np.empty((total_rows, first_sample.shape[1]), dtype=first_sample.dtype)

        current_idx = 0
        for feature in features:
            rows = feature.shape[0]
            self.data[current_idx : current_idx + rows] = feature
            current_idx += rows

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 
