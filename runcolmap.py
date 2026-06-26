import argparse
import os
import numpy as np
import glob
import sys
import shutil
import sqlite3
import json
import csv

import open3d as o3d

from scripts.colmap_converter import read_points3D_binary
from scene.dataset_readers import storePly


def array_to_blob(array):
    return array.tostring()


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.fromstring(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid


def camTodatabase(database_path, txt_path):
    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    if not os.path.exists(database_path):
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txt_path, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                # if len(strLists) == 0:
                #     continue
                # else:
                #     print(strLists)
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        if camera_id != idList[i]:
            print(f"cameraId={camera_id}, idList={idList[i]}")
            break
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    db.close()


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


blender2opencv = np.eye(4)


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def run_colmap_n3dv(root_dir, colmap_dir, downsample, timestamp, start_timestamp=0, dense_reconstruction=True):
    # prepare images and poses
    poses_arr = np.load(os.path.join(root_dir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]
    videos = glob.glob(os.path.join(root_dir, "cam[0-9][0-9]"))
    videos = sorted(videos)
    assert len(videos) == poses_arr.shape[0]
    H, W, focal = poses[0, :, -1]
    H /= downsample
    W /= downsample
    focal = focal / downsample
    focal = [focal, focal]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # videos = glob.glob(os.path.join(root_dir, "cam[0-9][0-9]"))
    # videos = sorted(videos)
    image_paths = []
    image_name = str(timestamp).zfill(4)
    for index, video_path in enumerate(videos):
        image_path = os.path.join(video_path, "images", f"{image_name}.png")
        image_paths.append(image_path)
    # print(image_paths)
    goal_dir = os.path.join(colmap_dir, "images")
    os.makedirs(goal_dir)

    image_name_list = []
    for index, image in enumerate(image_paths):
        image_name = image.split("/")[-1].split('.')
        # image_name[0] = "r_%03d" % index
        cam_name = image.split("/")[-3]
        image_name[0] = cam_name
        image_name = ".".join(image_name)
        image_name_list.append(image_name)
        goal_path = os.path.join(goal_dir, image_name)
        # shutil.copy(image, goal_path)
        os.symlink(image, goal_path)

    # write image information.
    sparse_dir = os.path.join(colmap_dir, "sparse_custom")
    os.makedirs(sparse_dir)
    object_images_file = open(os.path.join(sparse_dir, "images.txt"), "w")
    for idx, pose in enumerate(poses):
        # pose_44 = np.eye(4)
        R = pose[:3, :3]
        R = -R
        R[:, 0] = -R[:, 0]
        T = pose[:3, 3]

        R = np.linalg.inv(R)
        T = -np.matmul(R, T)
        T = [str(i) for i in T]
        qevc = [str(i) for i in rotmat2qvec(R)]
        print(idx + 1, " ".join(qevc), " ".join(T), 1, image_name_list[idx], "\n", file=object_images_file)

    # write camera infomation.
    object_cameras_file = open(os.path.join(sparse_dir, "cameras.txt"), "w")
    print(1, "SIMPLE_PINHOLE", int(W), int(H), focal[0], W / 2, H / 2, file=object_cameras_file)  #
    object_point_file = open(os.path.join(sparse_dir, "points3D.txt"), "w")

    object_cameras_file.close()
    object_images_file.close()
    object_point_file.close()

    # spare reconstruction
    db_path = os.path.join(colmap_dir, "database.db")
    os.system(f"colmap feature_extractor --database_path {db_path} --image_path {goal_dir} "
              f"--SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1")

    # os.system(f"python database.py --database_path {db_path} --txt_path {os.path.join(sparse_dir, 'cameras.txt')}")
    camTodatabase(db_path, os.path.join(sparse_dir, 'cameras.txt'))

    os.system(f"colmap exhaustive_matcher --database_path {db_path}")

    new_sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    os.makedirs(new_sparse_dir)
    os.system(f"colmap point_triangulator --database_path {db_path} --image_path {goal_dir} --input_path {sparse_dir} --output_path {new_sparse_dir} --clear_points 1")

    points3D = read_points3D_binary(os.path.join(new_sparse_dir, "points3D.bin"))
    xyz = []
    rgb = []
    for k in points3D:
        xyz.append(points3D[k].xyz)
        rgb.append(points3D[k].rgb)
    xyz = np.array(xyz)
    rgb = np.array(rgb)
    ply_output_dir = os.path.join(root_dir, "plys")
    os.makedirs(ply_output_dir, exist_ok=True)
    storePly(os.path.join(ply_output_dir, f"sparse_points3D_{timestamp}.ply"), xyz, rgb)

    if dense_reconstruction:
        dense_dir = os.path.join(colmap_dir, "dense")
        os.makedirs(dense_dir)
        os.system(f"colmap image_undistorter --image_path {goal_dir} --input_path {new_sparse_dir} --output_path {dense_dir}")
        os.system(f"colmap patch_match_stereo --workspace_path {dense_dir}")

        ply_path = os.path.join(dense_dir, 'fused.ply')
        os.system(f"colmap stereo_fusion --workspace_path {dense_dir} --output_path {ply_path}")

        # downsample point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Total points: {len(pcd.points)}")

        # 通过点云下采样将输入的点云减少
        # voxel_size = 0.02
        # while len(pcd.points) > 40000:
        #     pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        #     print(f"Downsampled points: {len(pcd.points)}")
        #     voxel_size += 0.01
        num_points = len(pcd.points)
        target_points = 35000
        sampling_interval = max(1, num_points // target_points)
        pcd = pcd.uniform_down_sample(every_k_points=sampling_interval)
        print(f"Downsampled points: {len(pcd.points)}")

        o3d.io.write_point_cloud(os.path.join(ply_output_dir, f"point3D_downsample2_{timestamp}.ply"), pcd)
        shutil.copy(ply_path, os.path.join(ply_output_dir, f"points3D_{timestamp}.ply"))
        if timestamp == 0:
            o3d.io.write_point_cloud(os.path.join(root_dir, f"points3D_downsample2.ply"), pcd)

        # remove dense dir
        os.system(f"rm -rf {dense_dir}")


def run_colmap_ImViD(root_dir, colmap_dir, downsample, timestamp, start_timestamp=0, dense_reconstruction=True):
    # prepare images
    goal_dir = os.path.join(colmap_dir, "images")
    if not os.path.exists(goal_dir):
        videos = glob.glob(os.path.join(root_dir, "cam[0-9][0-9]"))
        videos = sorted(videos)
        image_paths = []
        image_name = str(timestamp).zfill(4)
        for index, video_path in enumerate(videos):
            image_path = os.path.join(video_path, "images", f"{image_name}.png")
            image_paths.append(image_path)
        os.makedirs(goal_dir)

        image_name_list = []
        for index, image in enumerate(image_paths):
            image_name = image.split("/")[-1].split('.')
            # image_name[0] = "r_%03d" % index
            cam_name = image.split("/")[-3]
            image_name[0] = cam_name
            image_name = ".".join(image_name)
            image_name_list.append(image_name)
            goal_path = os.path.join(goal_dir, image_name)
            # shutil.copy(image, goal_path)
            os.symlink(image, goal_path)
    else:
        print("[INFO] skip copy images!")

    # write image information.
    sparse_dir = os.path.join(colmap_dir, "sparse_custom")
    # os.makedirs(sparse_dir)

    object_point_file = open(os.path.join(sparse_dir, "points3D.txt"), "w")
    object_point_file.close()

    # spare reconstruction
    db_path = os.path.join(colmap_dir, "database.db")
    os.system(f"colmap feature_extractor --database_path {db_path} --image_path {goal_dir} --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV "
              f"--SiftExtraction.max_image_size 4096 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1")

    # os.system(f"python database.py --database_path {db_path} --txt_path {os.path.join(sparse_dir, 'cameras.txt')}")
    camTodatabase(db_path, os.path.join(sparse_dir, 'cameras.txt'))

    os.system(f"colmap exhaustive_matcher --database_path {db_path}")

    new_sparse_dir = os.path.join(colmap_dir, "distorted", "sparse")
    os.makedirs(new_sparse_dir)
    os.system(f"colmap point_triangulator --database_path {db_path} --image_path {goal_dir} --input_path {sparse_dir} --output_path {new_sparse_dir} --clear_points 1")

    points3D = read_points3D_binary(os.path.join(new_sparse_dir, "points3D.bin"))
    xyz = []
    rgb = []
    for k in points3D:
        xyz.append(points3D[k].xyz)
        rgb.append(points3D[k].rgb)
    xyz = np.array(xyz)
    rgb = np.array(rgb)
    ply_output_dir = os.path.join(root_dir, "plys")
    os.makedirs(ply_output_dir, exist_ok=True)
    storePly(os.path.join(ply_output_dir, f"sparse_points3D_{timestamp}.ply"), xyz, rgb)

    if dense_reconstruction:
        dense_dir = os.path.join(colmap_dir, "dense")
        os.makedirs(dense_dir)
        os.system(f"colmap image_undistorter --image_path {goal_dir} --input_path {new_sparse_dir} --output_path {dense_dir}")
        os.system(f"colmap patch_match_stereo --workspace_path {dense_dir}")

        ply_path = os.path.join(dense_dir, 'fused.ply')
        os.system(f"colmap stereo_fusion --workspace_path {dense_dir} --output_path {ply_path}")

        # downsample point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Total points: {len(pcd.points)}")

        # 通过点云下采样将输入的点云减少
        # voxel_size = 0.02
        # while len(pcd.points) > 40000:
        #     pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        #     print(f"Downsampled points: {len(pcd.points)}")
        #     voxel_size += 0.01
        num_points = len(pcd.points)
        target_points = 35000
        sampling_interval = max(1, num_points // target_points)
        pcd = pcd.uniform_down_sample(every_k_points=sampling_interval)
        print(f"Downsampled points: {len(pcd.points)}")

        o3d.io.write_point_cloud(os.path.join(ply_output_dir, f"point3D_downsample2_{timestamp}.ply"), pcd)
        shutil.copy(ply_path, os.path.join(ply_output_dir, f"points3D_{timestamp}.ply"))
        if timestamp == 0:
            o3d.io.write_point_cloud(os.path.join(root_dir, f"points3D_downsample2.ply"), pcd)

        # remove dense dir
        os.system(f"rm -rf {dense_dir}")

def run_colmap_multiview(root_dir, colmap_dir, downsample, timestamp, start_timestamp=0, dense_reconstruction=True):
    # prepare images
    goal_dir = os.path.join(colmap_dir, "images")
    if not os.path.exists(goal_dir):
        videos = glob.glob(os.path.join(root_dir, "cam[0-9][0-9]"))
        videos = sorted(videos)
        image_paths = []
        image_name = str(timestamp).zfill(4)
        for index, video_path in enumerate(videos):
            image_path = os.path.join(video_path, "images", f"{image_name}.png")
            image_paths.append(image_path)
        os.makedirs(goal_dir)

        image_name_list = []
        for index, image in enumerate(image_paths):
            image_name = image.split("/")[-1].split('.')
            cam_name = image.split("/")[-3]
            image_name[0] = cam_name
            image_name = ".".join(image_name)
            image_name_list.append(image_name)
            goal_path = os.path.join(goal_dir, image_name)
            shutil.copy(image, goal_path)
            # os.symlink(image, goal_path)
    else:
        print("[INFO] skip copy images!")

    # spare reconstruction
    ply_output_dir = os.path.join(root_dir, "plys")
    os.makedirs(ply_output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(colmap_dir, "distorted", "sparse", "0")):
        db_path = os.path.join(colmap_dir, "database.db")
        os.system(f"colmap feature_extractor --database_path {db_path} --image_path {goal_dir} "
                  f"--ImageReader.single_camera 1 --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --SiftExtraction.use_gpu 1")  #  SIMPLE_PINHOLE

        os.system(f"colmap exhaustive_matcher --database_path {db_path} --SiftMatching.use_gpu 1")

        new_sparse_dir = os.path.join(colmap_dir, "distorted", "sparse")
        os.makedirs(new_sparse_dir)
        os.system(f"colmap mapper --database_path {db_path} --image_path {goal_dir} --output_path {new_sparse_dir}")

        points3D = read_points3D_binary(os.path.join(new_sparse_dir, "0", "points3D.bin"))
        xyz = []
        rgb = []
        for k in points3D:
            xyz.append(points3D[k].xyz)
            rgb.append(points3D[k].rgb)
        xyz = np.array(xyz)
        rgb = np.array(rgb)
        storePly(os.path.join(ply_output_dir, f"sparse_points3D_{timestamp}.ply"), xyz, rgb)
    else:
        print(f"skip sparse reconstruction!")

    if dense_reconstruction:
        sparse_dir = os.path.join(colmap_dir, "distorted", "sparse", "0")
        dense_dir = os.path.join(colmap_dir, "dense")
        os.makedirs(dense_dir)
        os.system(f"colmap image_undistorter --image_path {goal_dir} --input_path {sparse_dir} --output_path {dense_dir} --output_type COLMAP")
        os.system(f"colmap patch_match_stereo --workspace_path {dense_dir} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true")

        ply_path = os.path.join(dense_dir, 'fused.ply')
        os.system(f"colmap stereo_fusion --workspace_path {dense_dir} --workspace_format COLMAP --input_type geometric --output_path {ply_path}")

        # downsample point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"Total points: {len(pcd.points)}")

        # 通过点云下采样将输入的点云减少
        num_points = len(pcd.points)
        target_points = 35000
        sampling_interval = max(1, num_points // target_points)
        pcd = pcd.uniform_down_sample(every_k_points=sampling_interval)
        print(f"Downsampled points: {len(pcd.points)}")

        o3d.io.write_point_cloud(os.path.join(ply_output_dir, f"points3D_downsample2_{timestamp}.ply"), pcd)
        shutil.copy(ply_path, os.path.join(ply_output_dir, f"points3D_{timestamp}.ply"))
        if timestamp == 0:
            o3d.io.write_point_cloud(os.path.join(root_dir, f"points3D_downsample2.ply"), pcd)

        # remove dense dir
        # os.system(f"rm -rf {dense_dir}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=300)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument("--type", type=str, default="n3dv", help="n3dv, msth, meet, immersive")
    parser.add_argument('--dense', type=int, default=0)
    args = parser.parse_args()

    if args.type == 'n3dv':
        run_colmap = run_colmap_n3dv
        downsample = 2
    elif args.type == 'meet':
        run_colmap = run_colmap_n3dv
        downsample = 1
    elif args.type == 'vru':
        run_colmap = run_colmap_multiview
        downsample = 1
    elif args.type == 'imvid':
        run_colmap = run_colmap_ImViD
        downsample = 2
    else:
        run_colmap = run_colmap_multiview
        downsample = 1

    for timestamp in range(args.start, args.end, args.interval):
        colmap_dir = os.path.join(args.root_dir, "colmap", f"time_{timestamp}")
        if os.path.exists(colmap_dir):
            os.system(f"rm -rf {os.path.join(colmap_dir, 'sparse')}")
            os.system(f"rm -rf {os.path.join(colmap_dir, 'database.db')}")
        else:
            os.makedirs(colmap_dir)

        run_colmap(args.root_dir, colmap_dir, downsample, timestamp, args.start, args.dense)
