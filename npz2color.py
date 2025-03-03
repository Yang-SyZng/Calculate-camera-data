import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')



def load_and_visualize_depth(npz_file):
    # 加载 .npz 文件
    data = np.load(npz_file)
    
    # 假设深度图存储在键名 'depth'，可以根据具体情况调整
    depth_key = 'depth' if 'depth' in data else list(data.keys())[0]
    depth_map = data[depth_key]
    
    # 可视化深度图
    plt.figure(figsize=(6,8))
    plt.imshow(depth_map, cmap='jet')  # 使用 'jet' 颜色映射
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 示例使用
if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import sys

    print("NumPy Version:", np.__version__)
    print("Matplotlib Version:", matplotlib.__version__)
    print("Matplotlib Backend:", matplotlib.get_backend())
    print("Python Version:", sys.version)
    npz_file = rf'111111/data/depth/depth_0/00001-cam0.npz'
    load_and_visualize_depth(npz_file)
