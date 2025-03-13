import os
import shutil

# 根据点云着色后的文件夹格式，自动拉取undis去畸变图像，color着色点云，CC位姿优化文件到data目录中

# create by zc @ 2024/02/26
def copy_files(srcDir, desDir):
    """复制文件夹下的目录到另一个目录

    Args:
        srcDir (str): 源目录路径
        desDir (str): 目的目录路径
    """
    # 获取源目录中的所有文件和文件夹
    ls = os.listdir(srcDir)
    
    # 如果目标目录不存在，则创建它
    if not os.path.exists(desDir):
        os.makedirs(desDir)
    
    # 遍历源目录中的每个文件和文件夹
    for line in ls:
        # 构建文件的完整路径
        filePath = os.path.join(srcDir, line)
        
        # 检查路径对应的是否是文件
        if os.path.isfile(filePath):
            # 打印文件路径
            print(filePath)
            
            # 将文件复制到目标目录中
            shutil.copy(filePath, desDir)

if __name__ == "__main__":
    user_input = input("请输入路径：")
    source_dir = os.path.abspath(user_input)
    
    # 复制undis文件夹中0, 1, 2, 3文件夹里的内容
    undis_target_dir = os.path.abspath(os.path.join("data", os.path.basename(source_dir), "input", "undis"))
    for i in range(4):
        undis_source_dir = os.path.join(source_dir, "undis",str(i))
        copy_files(undis_source_dir, undis_target_dir)


    # 复制CC文件夹及其内容
    cc_source_dir = os.path.join(source_dir, "CC")
    cc_target_dir = os.path.abspath(os.path.join("data", os.path.basename(source_dir), "input","CC"))
    copy_files(cc_source_dir, cc_target_dir)

    # 复制color文件夹及其内容
    color_source_dir = os.path.join(source_dir, "color")
    color_target_dir = os.path.abspath(os.path.join("data", os.path.basename(source_dir), "input","color"))
    copy_files(color_source_dir, color_target_dir)
    
# end create @ zc
