from lxml import etree
import os
import open3d as o3d
import tqdm
from PIL import Image

"""
created by yang @ 2024/11/07
restructured by yang @ 2025/02/28
"""

def generate_depth(tree: etree._ElementTree = None):
    depth_path = './depth'
    photo_path = 'data/undis'
    photo_names = os.listdir(photo_path)

    try:
        os.scandir(depth_path)
    except FileNotFoundError:
        os.mkdir(depth_path)
    if tree:
        tree = tree

    root = tree.getroot()
    block = root.find("Block")
    photogroups = block.find("Photogroups")
    photogroup_lists = photogroups.findall("Photogroup")

    pc = o3d.io.read_point_cloud(self.point_cloud)

    # 照片组数
    cam_num = len(photogroup_lists)
    # 相机信息、图像参数信息
    cameras_info = []
    photos_info = []
    for i in range(cam_num):
        cameras_info.append(self.find_camera_info(photogroup_lists[i]))
        photos = photogroup_lists[i].findall("Photo")
        photos_info.append(self.find_photo_info(photo_names, photos))

    for i, photos_group in enumerate(tqdm(photos_info, desc="Processing photo groups")):
        try:
            os.scandir(depth_path + f'/{i}')
        except FileNotFoundError:
            os.mkdir(depth_path + f'/{i}')
        for photo_info in tqdm(photos_group, desc=f"Rendering depths for group {i + 1}", leave=False):
            photo_depth, pured_points_cloud = self.render_depth(pc, cameras_info[i], photo_info)
            photo_name = photo_info[0].split('.')[0]
            Image.fromarray(photo_depth).save(depth_path + f'/{i}' + f'/depth-{photo_name}.png')
            o3d.io.write_point_cloud(depth_path + f'/{i}' + f'/depth-{photo_name}.pcd', pured_points_cloud)


def render_depth(points_cloud: o3d.cpu.pybind.geometry.PointCloud, camera_info, photo_info: list):
    # 计算radius
    diameter = np.linalg.norm(np.asarray(points_cloud.get_max_bound()) - np.asarray(points_cloud.get_min_bound()))
    # 预处理内外参
    c_intrinsic_matrix, c_extrinsic_matrix = self.process(camera_info, photo_info[1])
    width, height = camera_info[0].astype(int), camera_info[1].astype(int)
    # 点云的剔除
    pured_points_cloud = self.pure_point_cloud(points_cloud, photo_info[1][1], radius=diameter * 1000)

    # 点云 世界坐标 -> 相机坐标
    pc_camera_coordinate = self.calculate_camera_coordinate(np.array(pured_points_cloud.points), c_extrinsic_matrix).T
    # 提取相机坐标下点云的深度信息
    # 过滤相机背后的点云
    valid = (pc_camera_coordinate[:, 2] >= 0)
    pc_camera_coordinate = pc_camera_coordinate[valid]
    Z_c = pc_camera_coordinate[:, 2]
    # 点云投影到图像上
    pc_camera_2d_coordinate = self.calculate_2d_coordinate(pc_camera_coordinate, c_intrinsic_matrix)

    u, v = pc_camera_2d_coordinate[:, 0], pc_camera_2d_coordinate[:, 1]
    # 进一步过滤：只保留在图像边界内的点
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid].astype(int)
    v = v[valid].astype(int)
    Z_c = Z_c[valid]
    # 转换成图像深度
    Z_c_normalized = (Z_c - Z_c.min()) / (Z_c.max() - Z_c.min()) * 255
    # 生成深度图
    depth_map = np.full((height, width), 0)
    for i in range(len(u)):
        depth_map[int(v[i]), int(u[i])] = Z_c_normalized[i]

    return depth_map.astype(np.uint8), pured_points_cloud

def pure_point_cloud(point_cloud, camera_location: np.ndarray, radius: float):
    _, pt_map = point_cloud.hidden_point_removal(camera_location=camera_location.reshape((3, 1)), radius=radius)
    pcd = point_cloud.select_by_index(pt_map)
    return pcd

if __name__ == '__main__':
    pass