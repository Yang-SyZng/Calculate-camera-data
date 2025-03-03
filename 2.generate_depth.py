import numpy as np
import os
import json
from tqdm.auto import tqdm
import open3d as o3d
from PIL import Image
from scripts.pull_data import read_info
from ip_basic.ip_basic import ip
import cv2

"""
created by yang @ 2024/11/07
restructured by yang @ 2025/03/2
"""

def render_depth(points_cloud, camera_info, photo_info, radius):
    # [width, height, fx, fy, cx, cy, intrinsic]
    # [id, img_name, quaternion, rotation, camera_position, extrinsic]
    # 预处理内外参
    width, height = camera_info[0].astype(int), camera_info[1].astype(int)
    # 点云的剔除
    pured_points = pure_point_cloud(points_cloud, np.array(photo_info[4]), radius=radius)

    # 提取点云的坐标和颜色
    points = np.asarray(pured_points.points)  # 世界坐标系下的点云坐标
    colors = np.asarray(pured_points.colors) if pured_points.has_colors() else None  # RGB 值（如果存在）

    # 点云 世界坐标 -> 相机坐标
    pc_camera_coordinate = world_to_camera(points, photo_info[-1]).T
    # 提取相机坐标下点云的深度信息
    # 过滤相机背后的点云
    valid_camera = (pc_camera_coordinate[:, 2] >= 0)
    pc_camera_coordinate = pc_camera_coordinate[valid_camera]
    points = points[valid_camera]
    if colors is not None:
        colors = colors[valid_camera]  # 同步过滤颜色

    Z_c = pc_camera_coordinate[:, 2]

    # 点云投影到图像上
    pc_camera_2d_coordinate = camera_to_2D(pc_camera_coordinate, camera_info[-1])
    u, v = pc_camera_2d_coordinate[:, 0], pc_camera_2d_coordinate[:, 1]

    # 进一步过滤：只保留在图像边界内的点
    valid_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_image].astype(int)
    v = v[valid_image].astype(int)
    Z_c = Z_c[valid_image]
    points = points[valid_image]
    if colors is not None:
        colors = colors[valid_image]  # 同步过滤颜色

    # 创建过滤后的点云对象
    filtered_points = o3d.geometry.PointCloud()
    filtered_points.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        filtered_points.colors = o3d.utility.Vector3dVector(colors)  # 保留 RGB 值
    # 转换成图像深度
    Z_c_normalized = (Z_c - Z_c.min()) / (Z_c.max() - Z_c.min()) * 255
    # 生成深度图
    depth_map_normalized = np.full((height, width), 0, dtype=np.uint8)
    depth_map = np.full((height, width), 0, dtype=np.float32)  # 单通道原始深度图
    depth_map_colored = np.zeros((height, width, 3), dtype=np.uint8)  # 三通道彩色深度图
    for i in range(len(u)):
        depth_map_normalized[int(v[i]), int(u[i])] = Z_c_normalized[i]
        depth_map[int(v[i]), int(u[i])] = Z_c[i]
        if colors is not None:
            depth_map_colored[v[i], u[i]] = (colors[i] * 255).astype(np.uint8)  # 将颜色值（0-1范围）转为 0-255

    return depth_map, depth_map_normalized, depth_map_colored.astype(np.uint8), filtered_points

def pure_point_cloud(point_cloud, camera_location: np.ndarray, radius: float):
    _, pt_map = point_cloud.hidden_point_removal(camera_location=camera_location.reshape((3, 1)), radius=radius)
    pcd = point_cloud.select_by_index(pt_map)
    return pcd

def world_to_camera(world_point: np.ndarray, extrinsic: np.ndarray):
    """
    Args:
        world_point: np.ndarray 格式的世界坐标系的点云
        extrinsic: 相机外参（Rt01矩阵）

    Returns: 相机坐标下的点云

    Raises:
        ValueError: 点云、外参形状不符合要求
    """
    if world_point.shape[1] != 3:
        raise ValueError(f"Expected world_points to have shape (N, 3), but got (N, {world_point.shape[1]})")
    if extrinsic.shape != (4, 4):
        raise ValueError(f"Expected extrinsic to have shape (4, 4), but got {extrinsic.shape}")

    # extend world_point
    # from [x, 3] 2 [x, 4]
    world_point = np.concatenate((world_point, np.ones((len(world_point), 1))), axis=1)

    # word 2 camera
    # [R, t]    ·   [pc 1]
    # [0, 1]
    camera_point = np.dot(extrinsic, world_point.T)

    return camera_point

def camera_to_2D(camera_points: np.ndarray, intrinsic: np.ndarray):
    """
    Args:
        camera_points: np.ndarray 格式的相机坐标系的点云
        intrinsic: 相机内参

    Returns: 2维图像上的点云（可能包含图像坐标外的点）

    Raises:
        ValueError: 点云、内参形状不符合要求
    """
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic to have shape (3, 3), but got {intrinsic.shape}")

    # extend intrinsic
    # form [3, 3] 2 [3, 4]
    intrinsic = np.concatenate((intrinsic, np.array([0., 0., 0.]).reshape(3, 1)), axis=1)
    # camera_points 3d 2 2d
    photo_points = np.dot(intrinsic, camera_points.T)
    # 提取 Z 得到[u, v, 1]
    photo_points = photo_points / photo_points[2, :]

    return photo_points.T[:, :2]

def convert_in_ex(intrinsic_info: list, extrinsic_info: list):
    """
    Args:
        intrinsic_info: [width, height, f, S, cx, cy]
        extrinsic_info: [img_name, [R, t]]

    Returns: camera_intrinsic_matrix, camera_extrinsic_matrix

    Raises:
        ValueError: 内外参数信息不符合要求
    """
    if len(intrinsic_info) != 6:
        raise ValueError(f"Expected intrinsic to have amount 6, but got {len(intrinsic_info)}")
    if len(intrinsic_info) != 2:
        raise ValueError(f"Expected extrinsic to have amount 2, but got {len(extrinsic_info)}")
    if isinstance(intrinsic_info[0], np.ndarray):
        raise TypeError(f"Expected intrinsic type np.ndarray, but got {type(intrinsic_info)}")
    if isinstance(extrinsic_info[0], np.ndarray):
        raise TypeError(f"Expected extrinsic type np.ndarray, but got {type(extrinsic_info[0])}")
    if extrinsic_info[1].shape != (1, 3):
        raise ValueError(f"Expected extrinsic to have shape (1, 3), but got {extrinsic_info[1].shape}")
    if extrinsic_info[0].shape != (3, 3):
        raise ValueError(f"Expected extrinsic to have shape (3, 3), but got {extrinsic_info[0].shape}")


    return _convert_intrinsic_(intrinsic_info), _convert_extrinsic_(extrinsic_info)


def _convert_intrinsic_(camera_intrinsic: list):
    AspectRatio = 1
    w, h = camera_intrinsic[:2]
    f = camera_intrinsic[2]
    S = camera_intrinsic[3]
    cx, cy = camera_intrinsic[4:]

    # 传感器尺寸
    sensor_width = AspectRatio * (S / w)
    sensor_height = (S / w) / AspectRatio

    fx = f / sensor_width
    fy = f / sensor_height

    # 计算相机内参矩阵
    intrinsic = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

    return intrinsic


def _convert_extrinsic_(camera_extrinsic: list):
    roration = camera_extrinsic[0]
    xyz = camera_extrinsic[1].T
    xyz_offset = np.dot(-roration, xyz)
    # 计算相机外参矩阵
    extrinsic = np.concatenate((np.concatenate((roration, xyz_offset), axis=1), np.array([0, 0, 0, 1]).reshape(1, -1)),
                               axis=0)

    return extrinsic

def colorful(output_dir,output_colorful_dir):
    # modified by zc @ 2024/02/26
    """可视化深度图，生成深度图片

    Args:
        output_dir (str): 深度信息文件路径
        output_colorful_dir (str): 深度图片输出路径
    """
    # end modified
    os.makedirs(output_colorful_dir, exist_ok=True)
    input_depth_dir = os.path.expanduser(output_dir)
    image_list = []
    depth_pathes = os.listdir(input_depth_dir)
    for file in depth_pathes:
        if file[-3:] == "npz" or file[-3:] == "npy":
            image_list.append(input_depth_dir + '/' + file)
    for depth_image in image_list:
        depth = np.load(depth_image)
        if depth_image[-3:] == "npz":
            depth = depth[depth.files[0]].astype(np.float32)
        img = cv2.applyColorMap(np.uint8(depth / np.amax(depth) * 255), cv2.COLORMAP_JET)
        img_name = depth_image.split('/')[-1].split('.')[0]
        outpath = output_colorful_dir + '/' + img_name + '.png'
        cv2.imwrite(outpath, img)
        print(outpath)

# end created @ jxf

def c2d(input_dir: str, output_dir: str, cameras_info, photos_info, targetWidth):
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    color = os.path.join(input_dir, 'color')

    origin_depth = os.path.join(depth_dir, "origin_depth")
    origin_depth_colored = os.path.join(depth_dir, "origin_depth_colored")
    resized_depth = os.path.join(depth_dir, "resized_depth")
    colorful_points = os.path.join(depth_dir, "depth_color")

    points_cloud = o3d.io.read_point_cloud(os.path.join(color, os.listdir(color)[0]))
    # 计算radius
    diameter = np.linalg.norm(np.asarray(points_cloud.get_max_bound()) - np.asarray(points_cloud.get_min_bound()))

    for i, camera_info in tqdm(enumerate(cameras_info), desc="Processing photo groups", leave=True, position=0):
        os.makedirs(os.path.join(origin_depth, f'{i}'), exist_ok=True)
        os.makedirs(os.path.join(origin_depth_colored, f'{i}'), exist_ok=True)
        os.makedirs(os.path.join(resized_depth, f'{i}'), exist_ok=True)
        os.makedirs(os.path.join(colorful_points, f'{i}'), exist_ok=True)
        for j, photo_info in tqdm(enumerate(photos_info[i]), desc=f"Rendering depths for group {i + 1}", leave=True, position=1):
            photo_depth, photo_depth_normalized, photo_depth_colored, filtered_points =\
                render_depth(points_cloud, camera_info, photo_info, radius=diameter*1000)
            photo_name = photo_info[1].split('.')[0]
            # origin_depth
            np.save(origin_depth + f'/{i}' +  f'/depth-{photo_name}', photo_depth)
            # origin_depth_colored
            Image.fromarray(photo_depth_colored).save(origin_depth_colored + f'/{i}' + f'/depth-{photo_name}.png')
            # resized_depth
            ip(photo_depth, resized_depth + f'/{i}' + f'/resized-depth-{photo_name}', targetWidth)
            # colorful_points
            o3d.io.write_point_cloud(colorful_points + f'/{i}' + f'/depth-{photo_name}.pcd', filtered_points)
if __name__ == '__main__':
    input_dir = './input'
    output_dir = './output'

    with open('config/config.json') as file:
        data = json.load(file)
    targetWidth = int(data['imageWidth'])

    c2d(input_dir, output_dir, *read_info(input_dir), targetWidth)