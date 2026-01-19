# visualization/plotter.py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os

# 显式导入参数类和常量
from parameters import VisualizationParameters
from constants import (GRAIN_AL, GRAIN_BOUNDARY_AL, PRECIPITATE, METAL_TYPE_3,
                       NEUTRAL_SOLUTION, CORROSIVE_SOLUTION, CORROSION_PRODUCT)

def visualize_corr_grid_cpu(corr_grid_cpu: np.ndarray, step: int, vis_params: VisualizationParameters):
    """
    可视化腐蚀网格的特定切片。
    """
    n_total, m, l = corr_grid_cpu.shape
    z_slice = vis_params.z_slice

    if z_slice is None:
        z_slice = l // 2
    elif z_slice < 0 or z_slice >= l:
        print(f"警告: Z 切片索引 {z_slice} 超出范围 [0, {l-1}]，将使用中间切片 {l//2}")
        z_slice = l // 2

    try:
        plt.figure(figsize=(8, 8 * n_total / m)) # 调整图形大小以匹配纵横比
        # 定义颜色映射 (确保颜色列表与常量值对应)
        # 0:灰, 1:蓝, 2:黄, 3:品红, 4:白, 5:亮绿, 6:黑
        cmap = ListedColormap(['gray', 'blue', 'yellow', 'magenta', 'white', 'lime', 'black'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] # 边界要覆盖所有整数值
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        im = plt.imshow(corr_grid_cpu[:, :, z_slice], cmap=cmap, norm=norm, interpolation='nearest', aspect='auto') # aspect='auto' 可能更合适
        plt.title(f"腐蚀模拟 - 步骤: {step} (Z 切片: {z_slice})")

        # 添加颜色条
        cbar = plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5, 6])
        cbar.set_ticklabels(['晶粒Al', '晶界Al', '析出相', '金属3', '中性溶液', '腐蚀溶液S', '腐蚀产物P'])

        if vis_params.save_plots:
            if not os.path.exists(vis_params.save_dir):
                os.makedirs(vis_params.save_dir)
                print(f"创建目录: {vis_params.save_dir}")
            save_path = os.path.join(vis_params.save_dir, f"corrosion_step_{step:06d}_z{z_slice}.png")
            plt.savefig(save_path)
            # print(f"图像已保存至: {save_path}") # 保存信息可以取消注释

        if vis_params.show_plots:
            plt.show()
        else:
            plt.close() # 关闭图形，防止在不显示时占用内存
    except Exception as e:
        print(f"可视化时出错 (步骤 {step}, 切片 {z_slice}): {e}")
        plt.close() # 确保关闭图形