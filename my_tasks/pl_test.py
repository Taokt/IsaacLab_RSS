import numpy as np

# 读取单个 .npy 文件
file_path = "./distance_to_image_plane_59_0.npy"
data = np.load(file_path)

# 打印数据的信息
print("Data shape:", data.shape)
print("Data type:", data.dtype)
print("Data preview:", data)

import open3d as o3d

def depth_to_pointcloud(depth, intrinsic_matrix):
    """
    将深度图转换为点云
    depth: 深度图 (H, W)
    intrinsic_matrix: 相机内参 (3, 3)
    """
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)  # 去掉最后一维，变成 (H, W)
    h, w = depth.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # 构造像素网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # 构造点云
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

# 示例：加载深度图并生成点云
intrinsic_matrix = np.array([[732.9993,   0.0000, 320.0000],
        [  0.0000, 732.9993, 240.0000],
        [  0.0000,   0.0000,   1.0000]])  # 示例内参
depth = np.load(file_path)
pointcloud_data = depth_to_pointcloud(depth, intrinsic_matrix)

# 转换为 Open3D 格式
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud_data)

# 可视化点云
o3d.visualization.draw_geometries([pcd])

