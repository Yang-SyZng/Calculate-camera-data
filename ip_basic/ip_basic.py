import numpy as np
from . import depth_map_utils
import cv2
from PIL import Image

def ip(photo_depth, output_dir, targetWidth):
    photo_depth = photo_depth.astype(np.float32)
    fill_type = 'fast'
    extrapolate = True
    blur_type = 'bilateral'
    if fill_type == 'fast':
        final_depth = depth_map_utils.fill_in_fast(photo_depth, extrapolate=extrapolate, blur_type=blur_type)
    elif fill_type == 'multiscale':
        final_depth, process_dict = depth_map_utils.fill_in_multiscale(photo_depth, extrapolate=extrapolate, blur_type=blur_type)
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))

    photo_depth[photo_depth < 1] = final_depth[photo_depth < 1]
    depth = cv2.resize(photo_depth, (targetWidth, int(targetWidth / photo_depth.shape[1] * photo_depth.shape[0])))

    np.save(output_dir, depth)

