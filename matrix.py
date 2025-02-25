import numpy as np
from lxml import etree

class matrix:
    def __init__(self, camera_data_path: str):
        self.tree = etree.parse(camera_data_path)

    def generate_depth(self, save_path: str):
        print(self.tree.getroot())

    def find_xml(self, tree, photo_file_path: str):
        assert isinstance(tree.tree, etree._ElementTree), "TypeError"
        tree = tree.tree
        photo_file_name = photo_file_path.split('/')[-1]

        photo_number = int(photo_file_name.split('-')[0])
        camera_number = photo_file_name.split('-')[1].split('.')[0]

        root = tree.getroot().find("Block").find("Photogroups")[int(camera_number[-1])]

        # 图像大小
        ImageDimensions = root.find("ImageDimensions")
        width = ImageDimensions.find("Width").text
        height = ImageDimensions.find("Height").text
        # 焦距
        FocalLength = root.find("FocalLength")
        f = FocalLength.text
        # 主点坐标
        PrincipalPoint = root.find("PrincipalPoint")
        cx = PrincipalPoint.find("x").text
        cy = PrincipalPoint.find("y").text
        # 传感器尺寸
        SensorSize = root.find("SensorSize")
        S = SensorSize.text

        camera_intrinsic = np.array([width, height, f, S, cx, cy], dtype=np.float64)

        # 图像
        Photo = root[12 + photo_number]
        Rotation = Photo.find("Pose").find("Rotation")
        rotations = np.array([[Rotation[0].text, Rotation[1].text, Rotation[2].text],
                            [Rotation[3].text, Rotation[4].text, Rotation[5].text],
                            [Rotation[6].text, Rotation[7].text, Rotation[8].text],
                            ], dtype=np.float64)

        Center = Photo.find("Pose").find("Center")
        camera_position = np.array([[Center[0].text, Center[1].text, Center[2].text]], dtype=np.float64)

        camera_extrinsic = [rotations, camera_position]


        return camera_intrinsic, camera_extrinsic
    def calculate_2d_coordinate(self, world_point_cloud: np.ndarray, camera_intrinsic_matrix: np.ndarray, camera_extrinsic_matrix: np.ndarray):
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