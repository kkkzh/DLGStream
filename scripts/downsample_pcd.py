import os
import argparse

import open3d as o3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, default="The path of point cloud, include file name")
    parser.add_argument("--target_points", type=int, default="The number of target points")
    args = parser.parse_args()

    # downsample point cloud
    pcd = o3d.io.read_point_cloud(args.ply_path)
    print(f"Total points: {len(pcd.points)}")
    ply_output_dir = os.path.dirname(args.ply_path)
    timestamp = args.ply_path.split("/")[-1].split("_")[-1].split(".")[0]

    # 通过均匀下采样将输入的点云减少
    num_points = len(pcd.points)
    sampling_interval = max(1, num_points // args.target_points)
    pcd = pcd.uniform_down_sample(every_k_points=sampling_interval)
    print(f"Downsampled points: {len(pcd.points)}")

    o3d.io.write_point_cloud(os.path.join(ply_output_dir, f"point3D_downsample2_{timestamp}.ply"), pcd)