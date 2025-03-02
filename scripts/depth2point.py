import numpy as np

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

if __name__ == '__main__':
    pass
