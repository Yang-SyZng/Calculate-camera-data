import numpy as np
from lxml import etree
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
import json

"""
created by yang @ 2024/11/07
restructured by yang @ 2025/03/2
"""

def find_photo_info(photo_names: list, root: etree._ElementTree):
    photos_info_list = []
    for i in root:
        if i.find("ImagePath").text.split('/')[-1] in photo_names:
            # R矩阵
            Rotation = i.find("Pose").find("Rotation")
            rotation = np.array([[Rotation[0].text, Rotation[1].text, Rotation[2].text],
                                  [Rotation[3].text, Rotation[4].text, Rotation[5].text],
                                  [Rotation[6].text, Rotation[7].text, Rotation[8].text],
                                  ], dtype=np.float64)
            rot = R.from_matrix(rotation)
            # 获取四元数 (w, x, y, z)
            quaternion = rot.as_quat()

            # t矩阵
            Center = i.find("Pose").find("Center")
            camera_position = np.array([[Center[0].text, Center[1].text, Center[2].text]], dtype=np.float64)
            # 计算外参
            xyz = camera_position.T
            xyz_offset = np.dot(-rotation, xyz)
            # 计算相机外参矩阵
            extrinsic = np.concatenate((np.concatenate((rotation, xyz_offset), axis=1),
                                        np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

            photos_info_list.append([int(i.find("ImagePath").text.split('/')[-1].split('.')[0][-1]) + 1, i.find("ImagePath").text.split('/')[-1], quaternion, rotation, camera_position, extrinsic])

    return photos_info_list


def find_camera_info(root: etree._ElementTree):
    # 图像大小
    ImageDimensions = root.find("ImageDimensions")
    width = np.float64(ImageDimensions.find("Width").text)
    height = np.float64(ImageDimensions.find("Height").text)
    # 焦距
    FocalLength = root.find("FocalLength")
    f = np.float64(FocalLength.text)
    # 传感器尺寸
    SensorSize = root.find("SensorSize")
    S = np.float64(SensorSize.text)
    # 主点坐标
    PrincipalPoint = root.find("PrincipalPoint")
    cx = np.float64(PrincipalPoint.find("x").text)
    cy = np.float64(PrincipalPoint.find("y").text)

    AspectRatio = 1
    # 传感器尺寸
    sensor_width = AspectRatio * (S / width)
    sensor_height = (S / width) / AspectRatio

    fx = f / sensor_width
    fy = f / sensor_height
    # 计算相机内参矩阵
    intrinsic = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

    camera_info = [width, height, fx, fy, cx, cy, intrinsic]

    return camera_info

def read_info(input_dir: str):
    CC = os.path.join(input_dir, 'CC')
    color = os.path.join(input_dir, 'color')
    undis = os.path.join(input_dir, 'undis')
    try:
        os.scandir(CC)
    except FileNotFoundError:
        print(f"Can not found {CC}, please check data")
        exit(-1)
    try:
        os.scandir(color)
    except FileNotFoundError:
        print(f"Can not found {color}, please check data")
        exit(-1)
    try:
        os.scandir(undis)
    except FileNotFoundError:
        print(f"Can not found {undis}, please check data")
        exit(-1)

    xml_files = os.listdir(CC)
    xml_file = etree.parse(os.path.join(CC, f'{xml_files[0]}'))

    undis_groups = os.listdir(undis)
    img_name = []
    for undis_group in undis_groups:
        img_name.append(os.listdir(os.path.join(undis, undis_group)))

    root = xml_file.getroot()
    block = root.find("Block")
    photogroups = block.find("Photogroups")
    photogroup_lists = photogroups.findall("Photogroup")

    # 相机信息、图像参数信息
    cameras_info = []
    photos_info = []
    for i in range(len(undis_groups)):
        cameras_info.append(find_camera_info(photogroup_lists[i]))
        photos = photogroup_lists[i].findall("Photo")
        photos_info.append(find_photo_info(img_name[i], photos))
    return cameras_info, photos_info

def save_info(input_dir: str, output_dir: str, cameras_info, photos_info, targetWidth):
    undis = os.path.join(input_dir, 'undis')
    sparse_path = os.path.join(output_dir, 'sparse', '0')
    resize_images_path = os.path.join(output_dir, 'images')
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(resize_images_path, exist_ok=True)
    cameras_info_path = sparse_path + './cameras.txt'
    images_info_path = sparse_path + '/images.txt'

    with open(cameras_info_path, 'w') as f:
        for i, camera_info in enumerate(cameras_info):
            scale = camera_info[0] / targetWidth
            targetheight = int(camera_info[1] / scale)
            data = (f"{i + 1} {'PINHOLE'} {targetWidth} {targetheight}"
                    f" {camera_info[2] / scale} {camera_info[3] / scale}"
                    f" {camera_info[4] / scale} {camera_info[5] / scale}\n")
            f.write(data)
    index = 1
    # [id, img_name, quaternion, rotation, camera_position, extrinsic])
    with open(images_info_path, 'w') as file:
        for i, photo_info in enumerate(photos_info):
            for photo in photo_info:
                id = photo[0]
                img_n = photo[1]
                rotation = photo[3]
                xyz = photo[4]
                rot = R.from_matrix(rotation)
                # 获取四元数 (w, x, y, z)
                quaternion = rot.as_quat()

                data = (f"{index} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[2]} "
                        f"{xyz[0][0]} {xyz[0][1]} {xyz[0][2]} {id} {img_n}\n No Content \n")
                file.write(data)
                index += 1


                scale = cameras_info[id - 1][0] / targetWidth
                # 调整大小
                original_image = Image.open(os.path.join(undis, f'{i}', f'{photo[1]}'))
                new_size = (int(cameras_info[id - 1][0] / scale), int(cameras_info[id - 1][1] / scale))  # 替换为目标大小
                resized_image = original_image.resize(new_size)

                resized_image.save(os.path.join(resize_images_path, img_n))
if __name__ == '__main__':
    input_dir = 'input'
    output_dir = 'output'

    with open('config.json') as file:
        data = json.load(file)
    targetWidth = int(data['imageWidth'])

    print('====start pull data====')
    save_info(input_dir, output_dir, *read_info(input_dir), targetWidth)
    print('====pull data finished====')