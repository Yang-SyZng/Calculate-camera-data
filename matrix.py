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



