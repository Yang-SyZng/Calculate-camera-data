import glob
import os
import numpy as np
import depth_map_utils
import cv2

def ip(input_depth_dir, output_dir,imageWidth):
    os.makedirs(output_dir, exist_ok=True)
    fill_type = 'fast'
    extrapolate = True
    blur_type = 'bilateral'

    images_to_use = sorted(glob.glob(input_depth_dir + '/*npz'))
    num_images = len(images_to_use)
    for i in range(num_images):
        depth_image = images_to_use[i]
        depth = np.load(depth_image)
        depth = depth[depth.files[0]].astype(np.float32)
        depth_copy = depth.copy()
        if fill_type == 'fast':
            final_depth = depth_map_utils.fill_in_fast(
                depth_copy, extrapolate=extrapolate, blur_type=blur_type)
        elif fill_type == 'multiscale':
            final_depth, process_dict = depth_map_utils.fill_in_multiscale(
                depth_copy, extrapolate=extrapolate, blur_type=blur_type)
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))

        depth[depth<1] = final_depth[depth<1]
        new_width = imageWidth
        depth = cv2.resize(depth, (new_width, int(new_width / depth.shape[1] * depth.shape[0])))
        filename = depth_image.split('\\')[-1]
        np.savez(os.path.join(output_dir, filename), depth)
        print(os.path.join(output_dir, filename))
        print(output_dir)
        print(filename)
        # print("finish,ip:"+ str(i))
