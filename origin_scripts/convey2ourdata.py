# from scene.colmap_loader import read_extrinsics_binary
import os
import sys
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
import xml.etree.ElementTree as ET
import collections
from scipy.spatial.transform import Rotation as R
from PIL import Image
from collections import defaultdict

"""create by jxf @ 2023/01/24
   modified by zc @ 2024/02/26
将私有的点云和影像按照colmap的格式进行转换
"""

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "fx", "fy","px","py"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Models = "PINHOLE"


#add by zc @ 2024/02/26
def getReadFilePath(filePath,fileType):
    """读取文件夹下指定类型文件，若超出一个同类型的文件则提示异常

    Args:
        filePath (string): 文件所在文件夹路径
        fileType (string): 指定读取的文件类型

    Raises:
        ValueError: error

    Returns:
        string: 指定类型文件路径
    """
    file_count = 0  # 文件计数器
    file_name = ""
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(filePath):
        
        if(file_name.split('.')[-1].lower() == fileType):
            file_count += 1  # 每找到一个文件，计数器加1
        if file_count != 1:
            raise ValueError(f"Multiple {fileType} or None file found in the folder.Please check folder and filetype")
    return os.path.join(filePath, file_name)
# end add @ zc

def read_xml(path):
    # modified by zc @ 2024/02/26
    """读取CC xml文件，获取位姿优化成功的影像

    Args:
        path (string): CC xml路径

    Returns:
        camera: 相机列表
        images: 图片列表
        
    """
    # end modified @ zc
    camera={}
    images={}
    tree = ET.parse(path)
    root = tree.getroot()

    aim_root = root[1][3]
    camera_id = 0

    for children in aim_root:

        camera_id +=1
        image_width = int(children[3][0].text)
        image_height = int(children[3][1].text)
        focal_lenth = float(children[6].text)
        sensor_size = float(children[7].text)
        f_total = image_width*focal_lenth/sensor_size
        PrincipalPoint_x = float(children[9][0].text)
        PrincipalPoint_y = float(children[9][1].text)

        camera[camera_id] = Camera(id=camera_id, model=Models,
                                    width=image_width, height=image_height,
                                    fx=f_total,fy=f_total,px=PrincipalPoint_x,py=PrincipalPoint_y)

        for image in children.findall('Photo'):
            image_id = int(image[0].text) + 1
            image_path = image[1].text
            file_name = os.path.basename(image_path)

            numbers = [float(image[3][0][0].text), float(image[3][0][1].text), float(image[3][0][2].text),
                       float(image[3][0][3].text), float(image[3][0][4].text), float(image[3][0][5].text),
                       float(image[3][0][6].text), float(image[3][0][7].text), float(image[3][0][8].text)]
            rotation = np.array(numbers).reshape(3, 3)
            translation_w = [float(image[3][1][0].text), float(image[3][1][1].text), float(image[3][1][2].text)]
            translation_meta = [float(image[3][2][2][0].text), float(image[3][2][2][1].text), float(image[3][2][2][2].text)]
            if translation_w[0]-translation_meta[0]==0:
                print('CC OPTIMIZATION FAIL! DELETE THE POSE!')
            else:
                translation = np.array(translation_w).reshape(3, 1)
                translation = -np.dot(rotation,translation)


                r3 = R.from_matrix(rotation)
                qua = r3.as_quat()
                xys = [1,1]
                point3D_ids = 1

                images[image_id] = BaseImage(id=image_id, qvec=qua,
                                           tvec=translation, camera_id=camera_id,
                                           name=file_name, xys=xys, point3D_ids=point3D_ids)
    return camera,images

def save_colmap(cameras,images,path):
    # modified by zc @ 2024/02/26
    """将数据转化为colmap格式。
        <location>
        |---images
        |   |---<image 0>
        |   |---<image 1>
        |   |---...
        |---sparse
            |---0
                |---cameras.txt
                |---images.txt
                |---points3D.ply
    

    Args:
        cameras (Camera): 相机列表
        images (BaseImage): 图像列表
        path (str): 点云，相机参数存储路径
    """ 
    # end modified @ zc
    camera_path = path + '/cameras.txt'
    image_path = path + '/images.txt'
    os.makedirs(path, exist_ok=True)
    with open(camera_path, 'w') as file:
        for camera in cameras:
            if(targetWidth == 1):
                scale = 1
            else:
                scale = cameras[camera].width/targetWidth

            image_height = int(cameras[camera].height/scale)

            data = f"{cameras[camera].id} {cameras[camera].model} {int(cameras[camera].width/scale)} {image_height} {cameras[camera].fx/scale} {cameras[camera].fy/scale} {cameras[camera].px/scale} {cameras[camera].py/scale}\n"
            file.write(data)
    index = 1
    # 创建一个defaultdict来存储每个camera_id的图片计数
    image_count = defaultdict(int)
    with open(image_path, 'w') as file:
        for image in images:
            qw = images[image].qvec[3]
            qx = images[image].qvec[0]
            qy = images[image].qvec[1]
            qz = images[image].qvec[2]

            x = images[image].tvec[0][0]
            y = images[image].tvec[1][0]
            z = images[image].tvec[2][0]
            
            camera_id = images[image].camera_id
            
                # 每个相机的照片取数
            if  image_count[camera_id] <  20:
                data = f"{index} {qw} {qx} {qy} {qz} {x} {y} {z} {camera_id} {images[image].name}\n{images[image].xys[0]} {images[image].xys[1]} {images[image].point3D_ids}\n"
                file.write(data)
                 # 增加相应camera_id的图片计数
                image_count[camera_id] += 1
                index += 1

                original_image = Image.open(os.path.abspath(f"data/{datasetName}/input/undis/{images[image].name}"))
                if(targetWidth == 1):
                    scale = 1
                else:
                    scale = cameras[camera].width/targetWidth
                # 调整大小
                new_size = (int(original_image.width/scale), int(original_image.height/scale))  # 替换为目标大小
                resized_image = original_image.resize(new_size)
                region = resized_image
                # 保存调整大小后的图像
                os.makedirs(f"data/{datasetName}/images", exist_ok=True)
                region.save(f"data/{datasetName}/images/{images[image].name}")

    print('=====SAVE DONE!=========')

# modified by zc @ 2024/02/26           
# 读取JSON文件
with open('scripts/config.json') as file:
    data = json.load(file)
# 访问JSON中的配置
# target image width
targetWidth = int(data['imageWidth'])
# dataset name 
datasetName = data['datasetName']

cameras,images = read_xml(getReadFilePath(os.path.abspath(f"data/{datasetName}/input/CC"),"xml"))
save_colmap(cameras,images,os.path.abspath(f"data/{datasetName}/sparse/0"))
# end modified @ zc
