import os
import xml.dom.minidom
import json
from cloud2depth_new import c2d
from ip_basic_new import ip
from depthColorful_new import colorful


"""create by jxf @ 2023/01/24
"""

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

#modified by zc @ 2024/02/26
# 读取JSON文件
with open('scripts/config.json') as file:
    data = json.load(file)
# 访问JSON中配置
datasetName = data['datasetName']
imageWidth = int(data['imageWidth'])

main_path = os.path.abspath(f"data/{datasetName}")
depth_path = os.path.join(main_path, "depth")

# 获取undis目录中的所有文件数量
num = len(os.listdir(os.path.abspath(f"data/{datasetName}/input/undis")))
# 读取CC文件
cc = os.path.abspath(f"data/{datasetName}/input/CC")
dom = xml.dom.minidom.parse(getReadFilePath(cc,"xml"))
# 读取着色点云文件      
color = os.path.abspath(f"data/{datasetName}/input/color")
pcd_path = getReadFilePath(color,"pcd")
# end modified @ zc

root = dom.documentElement
path0 = os.path.join(depth_path, "depth_0")
output_dir = os.path.join(depth_path, "depth_resize")
output_colorful_dir = os.path.join(depth_path, "depth_color")


r = 10000
print("Generating depth file...")
for i in range(num):
    c2d(i, root, path0, pcd_path, r)
# modified by zc @ 2024/02/26 add depth imageWidth option
ip(path0, output_dir,imageWidth=imageWidth)
# end modified @ zc
colorful(output_dir,output_colorful_dir)


