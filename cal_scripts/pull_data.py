import numpy as np
from lxml import etree


def find_photo_info(photo_names: list, root: etree._ElementTree):
    photos_info_list = []
    for i in root:
        if i.find("ImagePath").text.split('/')[-1] in photo_names:
            # R矩阵
            Rotation = i.find("Pose").find("Rotation")
            rotations = np.array([[Rotation[0].text, Rotation[1].text, Rotation[2].text],
                                  [Rotation[3].text, Rotation[4].text, Rotation[5].text],
                                  [Rotation[6].text, Rotation[7].text, Rotation[8].text],
                                  ], dtype=np.float64)
            Center = i.find("Pose").find("Center")
            # t矩阵
            camera_position = np.array([[Center[0].text, Center[1].text, Center[2].text]], dtype=np.float64)

            camera_extrinsic = [rotations, camera_position]

            photos_info_list.append([i.find("ImagePath").text.split('/')[-1], camera_extrinsic])
    return photos_info_list


def find_camera_info(root: etree._ElementTree):
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