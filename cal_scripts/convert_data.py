import numpy as np

"""
created by yang @ 2024/11/07
restructured by yang @ 2025/02/28
"""

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
        raise ValueError(f"Expected intrinsic to have shape (3, 3), but got{intrinsic.shape}")

    # extend intrinsic
    # form [3, 3] 2 [3, 4]
    intrinsic = np.concatenate((intrinsic, np.array([0., 0., 0.]).reshape(3, 1)), axis=1)
    # camera_points 3d 2 2d
    photo_points = np.dot(intrinsic, camera_points.T)
    # 提取 Z 得到[u, v, 1]
    photo_points = photo_points / photo_points[2, :]

    return photo_points.T[:, :2]

def process(camera_intrinsic: list, camera_extrinsic: list):
    assert len(camera_intrinsic) == 6, f"Amount Error, we need 6, but your are {len(camera_intrinsic)}"
    assert len(camera_extrinsic) == 2, f"Amount Error, we need 2, but your are {len(camera_extrinsic)}"
    assert isinstance(camera_extrinsic[0], np.ndarray), f"TypeError"
    assert isinstance(camera_extrinsic[1], np.ndarray), f"TypeError"
    assert camera_extrinsic[0].shape == (3, 3), f"Shape Error, we need (3, 3), but your are {camera_extrinsic[0].shape}"
    assert camera_extrinsic[1].shape == (1, 3), f"Shape Error, we need (1, 3), but your are {camera_extrinsic[1].shape}"

    return _process_intrinsic_(camera_intrinsic), _process_extrinsic_(camera_extrinsic)


def _process_intrinsic_(camera_intrinsic: list):
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


def _process_extrinsic_(camera_extrinsic: list):
    roration = camera_extrinsic[0]
    xyz = camera_extrinsic[1].T
    xyz_offset = np.dot(-roration, xyz)
    # 计算相机外参矩阵
    extrinsic = np.concatenate((np.concatenate((roration, xyz_offset), axis=1), np.array([0, 0, 0, 1]).reshape(1, -1)),
                               axis=0)

    return extrinsic

def calculate_depth_2_point_cloud(depth: np.ndarray, camera_intrinsic_matrix: np.ndarray,
                                  camera_extrinsic_matrix: np.ndarray) -> np.ndarray:
    assert len(depth.shape) == 2
    assert camera_intrinsic_matrix.shape == (
        3, 3), f"Shape Error, we need (3, 3), but your are {camera_intrinsic_matrix.shape}"
    assert camera_extrinsic_matrix.shape == (
        4, 4), f"Shape Error, we need (4, 4), but your are {camera_extrinsic_matrix.shape}"
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

if __name__ == '__main__':
    pc = np.ones((10, 3))
    ex = np.ones((4, 4))
    world_to_camera(pc, ex)