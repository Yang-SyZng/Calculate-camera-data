import numpy as np

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
