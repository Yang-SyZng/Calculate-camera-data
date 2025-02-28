import numpy as np
from lxml import etree
import open3d as o3d
from PIL import Image
import os
from tqdm import tqdm

class matrix:
    def __init__(self, camera_data_path: str):
        self.tree = etree.parse(camera_data_path)
        self.point_cloud = './color/color_SS_0.002_nf_RADIUS0.06_REL0.5_sor_100_1.5_sor_200_2_all.pcd.pcd'

    def generate_depth(self, tree: etree._ElementTree = None):
        depth_path = './depth'
        try:
            os.scandir(depth_path)
        except FileNotFoundError:
            os.mkdir(depth_path)

        photo_path = './undis'
        photo_names = os.listdir(photo_path)
        if tree:
            self.tree = tree

        root = self.tree.getroot()
        block = root.find("Block")
        photogroups = block.find("Photogroups")
        photogroup_lists = photogroups.findall("Photogroup")

        pc = o3d.io.read_point_cloud(self.point_cloud)

        # cam_num
        cam_num = len(photogroup_lists)
        # 相机内参
        cameras_info = []
        for i in range(cam_num):
            cameras_info.append(self.find_camera_info(photogroup_lists[i]))
        # 相机外参
        photos_info = []
        for i in range(cam_num):
            photos = photogroup_lists[i].findall("Photo")
            photos_info.append(self.find_photo_info(photo_names, photos))

        for i, photos_group in enumerate(tqdm(photos_info, desc="Processing photo groups")):
            try:
                os.scandir(depth_path + f'/{i}')
            except FileNotFoundError:
                os.mkdir(depth_path + f'/{i}')
            for photo_info in tqdm(photos_group, desc=f"Rendering depths for group {i + 1}", leave=False):
                photo_depth = self.render_depth(pc, cameras_info[i], photo_info)
                photo_name = photo_info[0].split('.')[0]
                Image.fromarray(photo_depth).save(depth_path + f'/{i}' + f'/depth-{photo_name}.png')
            exit(0)
    def find_photo_info(self, photo_names: list, root: etree._ElementTree):
        photos_info_list = []
        for i in root:
            if i.find("ImagePath").text.split('/')[-1] in photo_names:
                # 图像
                Rotation = i.find("Pose").find("Rotation")
                rotations = np.array([[Rotation[0].text, Rotation[1].text, Rotation[2].text],
                                      [Rotation[3].text, Rotation[4].text, Rotation[5].text],
                                      [Rotation[6].text, Rotation[7].text, Rotation[8].text],
                                      ], dtype=np.float64)
                Center = i.find("Pose").find("Center")
                camera_position = np.array([[Center[0].text, Center[1].text, Center[2].text]], dtype=np.float64)
                # 合并为 3x4 矩阵
                camera_extrinsic = [rotations, camera_position]
                photos_info_list.append([i.find("ImagePath").text.split('/')[-1], camera_extrinsic])
        return photos_info_list

    def render_depth(self, points_cloud: o3d.cpu.pybind.geometry.PointCloud, camera_info: list, photo_info: list):
        c_intrinsic_matrix, c_extrinsic_matrix = self.process(camera_info, photo_info[1])
        width, height = camera_info[0].astype(int), camera_info[1].astype(int)
        pured_points_cloud = self.pure_point_cloud(points_cloud, photo_info[1][1], radius=1000)
        # world_coordinate 2 camera_coordinate
        pc_camera_coordinate = self.calculate_camera_coordinate(np.array(pured_points_cloud.points), c_extrinsic_matrix).T

        # 提取相机坐标下点云的深度信息
        Z_c = pc_camera_coordinate[:, 2]
        pc_camera_2d_coordinate = self.calculate_2d_coordinate(pc_camera_coordinate, c_intrinsic_matrix)

        u, v = pc_camera_2d_coordinate[:, 0], pc_camera_2d_coordinate[:, 1]
        # 进一步过滤：只保留在图像边界内的点
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[valid].astype(int)
        v = v[valid].astype(int)
        Z_c = Z_c[valid]
        Z_c_normalized = (Z_c - Z_c.min()) / (Z_c.max() - Z_c.min()) * 255
        # 生成深度图
        depth_map = np.full((height, width), 0)
        for i in range(len(u)):
            depth_map[int(v[i]), int(u[i])] = Z_c_normalized[i]

        depth_normalized = depth_map.astype(np.uint8)
        return depth_normalized

    def pure_point_cloud(self, point_cloud, camera_location: np.ndarray, radius: float=1500.):
        _, pt_map = point_cloud.hidden_point_removal(camera_location=camera_location.reshape((3, 1)), radius=radius)
        pcd = point_cloud.select_by_index(pt_map)
        return pcd
    def calculate_camera_coordinate(self, world_point_cloud: np.ndarray, camera_extrinsic_matrix: np.ndarray):
        assert world_point_cloud.shape[1] == 3,\
            f"Shape Error, we need (x, 3), but your point cloud are (x, {world_point_cloud.shape[1]})"
        assert camera_extrinsic_matrix.shape == (4, 4),\
            f"Shape Error, we need (4, 4), but your are {camera_extrinsic_matrix.shape}"

        # extend world_point_cloud 2 (x, 4)
        world_point_cloud = np.concatenate((world_point_cloud, np.ones_like(world_point_cloud)[:, :1]), axis=1)

        # word 2 camera
        camera_point_cloud = np.dot(camera_extrinsic_matrix, world_point_cloud.T)

        return camera_point_cloud

    def calculate_2d_coordinate(self, camera_point_cloud: np.ndarray, camera_intrinsic_matrix: np.ndarray):
        assert camera_intrinsic_matrix.shape == (3, 3), \
            f"Shape Error, we need (3, 3), but your are {camera_intrinsic_matrix.shape}"
        # extend camera_intrinsic_matrix 2 (3, 4)
        camera_intrinsic_matrix = np.concatenate((camera_intrinsic_matrix, np.array([0., 0., 0.]).reshape(3, 1)),
                                                 axis=1)
        # camera 3d 2 2d
        camera_2d_point = np.dot(camera_intrinsic_matrix, camera_point_cloud.T)
        # normalization
        camera_2d_point = camera_2d_point / camera_2d_point[2, :]

        return camera_2d_point.T[:, :2]

    def find_camera_info(self, root: etree._ElementTree):
        # 图像大小
        ImageDimensions = root.find("ImageDimensions")
        width = ImageDimensions.find("Width").text
        height = ImageDimensions.find("Height").text
        # 焦距
        FocalLength = root.find("FocalLength")
        f = FocalLength.text
        # 传感器尺寸
        SensorSize = root.find("SensorSize")
        S = SensorSize.text
        # 主点坐标
        PrincipalPoint = root.find("PrincipalPoint")
        cx = PrincipalPoint.find("x").text
        cy = PrincipalPoint.find("y").text

        camera_info = np.array([width, height, f, S, cx, cy], dtype=np.float64)

        return camera_info

    def calculate_depth_2_point_cloud(self, depth: np.ndarray,  camera_intrinsic_matrix: np.ndarray, camera_extrinsic_matrix: np.ndarray) -> np.ndarray:
        assert len(depth.shape) == 2
        assert camera_intrinsic_matrix.shape == (3, 3), f"Shape Error, we need (3, 3), but your are {camera_intrinsic_matrix.shape}"
        assert camera_extrinsic_matrix.shape == (4, 4), f"Shape Error, we need (4, 4), but your are {camera_extrinsic_matrix.shape}"
        fx = camera_intrinsic_matrix[0][0]
        fy = camera_intrinsic_matrix[1][1]
        cx = camera_intrinsic_matrix[0][2]
        cy = camera_intrinsic_matrix[1][2]
        camera_extrinsic_inv = np.linalg.inv(camera_extrinsic_matrix)
        R = camera_extrinsic_inv[:3, :3]
        t = camera_extrinsic_inv[:3, -1]

        depth_height, depth_width = depth.shape
        # 生成每个像素的坐标
        u, v = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
        u = u.flatten()
        v = v.flatten()
        z = depth.flatten()
        z /= 2000
        x = (u - cx) * z / fx  # 计算X坐标
        y = (v - cy) * z / fy  # 计算Y坐标
        # 生成点云（每个点的坐标为 (X, Y, Z)）
        points = np.dot(R, np.array([x, y, z])).T + t

        return points

    def process(self, camera_intrinsic: list, camera_extrinsic: list):
        assert len(camera_intrinsic)==6, f"Amount Error, we need 6, but your are {len(camera_intrinsic)}"
        assert len(camera_extrinsic)==2, f"Amount Error, we need 2, but your are {len(camera_extrinsic)}"
        assert isinstance(camera_extrinsic[0], np.ndarray), f"TypeError"
        assert isinstance(camera_extrinsic[1], np.ndarray), f"TypeError"
        assert camera_extrinsic[0].shape==(3, 3), f"Shape Error, we need (3, 3), but your are {camera_extrinsic[0].shape}"
        assert camera_extrinsic[1].shape==(1, 3), f"Shape Error, we need (1, 3), but your are {camera_extrinsic[1].shape}"

        return self.__process_intrinsic__(camera_intrinsic), self.__process_extrinsic__(camera_extrinsic)
        

    def __process_intrinsic__(self, camera_intrinsic: list):
        AspectRatio = 1
        w, h = camera_intrinsic[:2]
        f = camera_intrinsic[2]
        S = camera_intrinsic[3]
        cx, cy = camera_intrinsic[4:]

        # 还没写完，假装能计算
        sensor_width, sensor_height = self.__calculate_sensor_sizes__(w, h, S)

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

    def __process_extrinsic__(self, camera_extrinsic: list):
        roration = camera_extrinsic[0]
        xyz = camera_extrinsic[1].T
        xyz_offset = np.dot(-roration, xyz)
        # 计算相机外参矩阵
        extrinsic = np.concatenate((np.concatenate((roration, xyz_offset), axis=1), np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

        return extrinsic

    def __calculate_sensor_sizes__(self, width, height, sensor_size):


        return 0, 0

def calculate_2d_coordinate(world_point_cloud: np.ndarray, camera_intrinsic_matrix: np.ndarray, camera_extrinsic_matrix: np.ndarray):
    assert world_point_cloud.shape[1] == 3, f"Shape Error, we need (x, 3), but your point cloud are (x, {world_point_cloud.shape[1]})"
    assert camera_extrinsic_matrix.shape == (4, 4), f"Shape Error, we need (4, 4), but your are {camera_extrinsic_matrix.shape}"
    assert camera_intrinsic_matrix.shape == (3, 3), f"Shape Error, we need (3, 3), but your are {camera_intrinsic_matrix.shape}"

    # extend camera_intrinsic_matrix 2 (3, 4)
    camera_intrinsic_matrix = np.concatenate((camera_intrinsic_matrix, np.array([0., 0., 0.]).reshape(3, 1)), axis=1)
    # extend world_point_cloud 2 (x, 4)
    world_point_cloud = np.concatenate((world_point_cloud, np.ones_like(world_point_cloud)[:, :1]), axis=1)
    
    # word 2 camera
    camera_point_cloud = np.dot(camera_extrinsic_matrix, world_point_cloud.T)
    # camera 3d 2 2d
    camera_2d_point = np.dot(camera_intrinsic_matrix, camera_point_cloud)
    # normalization
    camera_2d_point = camera_2d_point / camera_2d_point[2, :]

    return camera_2d_point.T[:, :2]

def process_intrinsic(camera_intrinsic: list):
        
        w, h = camera_intrinsic[:2]
        f = camera_intrinsic[2]
        S = camera_intrinsic[3]
        cx, cy = camera_intrinsic[4:]
        
        fx = f * w / S
        fy = f * h / S
        
        # 计算相机内参矩阵
        intrinsic = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

        return intrinsic   

def process_extrinsic(camera_extrinsic: list):
    roration = camera_extrinsic[0]
    xyz = camera_extrinsic[1].T
    xyz_offset = np.dot(-roration, xyz)
    # 计算相机外参矩阵
    extrinsic = np.concatenate((np.concatenate((roration, xyz_offset), axis=1), np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

    return extrinsic

def source2target(source_intrinsic, source_extrinsic, target_intrinsic, target_extrinsic):
    pass


def s2t_extrinsic(source: np.ndarray, target: np.ndarray):
    assert isinstance(source, np.ndarray), "matrix type must be numpy.ndarray"
    assert isinstance(target, np.ndarray), "matrix type must be numpy.ndarray"
    assert source.shape==(4, 4), f"shape Error, your shape is {source.shape}"
    assert target.shape==(4, 4), f"shape Error, your shape is {target.shape}"

    source_R = source[:3, :3]
    source_T = source[:3, 3:]

    target_R = target[:3, :3]
    target_T = target[:3, 3:]


    target_R_inv = np.linalg.inv(target_R)
    T = np.concatenate((np.concatenate((np.dot(target_R_inv, source_R), np.dot(target_R_inv, (source_T - target_T))), axis=1),  np.array([[0, 0, 0, 1]])), axis=0)
                 
    
    return T

def process_front(camera_extrinsic: list):
    pass