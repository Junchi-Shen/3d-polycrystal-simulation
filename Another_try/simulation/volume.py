# simulation/volume.py
import cupy as cp
import numpy as np
import random
from constants import * # 导入常量

class VolumeData:
    """封装 3D 网格数据及其属性"""
    def __init__(self, shape: tuple, solution_thickness: int):
        self.n_total, self.m, self.l = shape
        self.solution_thickness = solution_thickness
        self.grid_gpu: cp.ndarray = cp.zeros(shape, dtype=cp.int32) # 初始化为 0，稍后填充

    def initialize_grid(self, bound_grid_cpu: np.ndarray, concentration_ratio: float):
        """用初始状态填充网格"""
        n_metal, m, l = bound_grid_cpu.shape
        if m != self.m or l != self.l or n_metal != (self.n_total - self.solution_thickness):
             raise ValueError("Bound grid shape mismatch")

        # 在 CPU 上创建 NumPy 数组
        corr_grid_cpu = np.full((self.n_total, self.m, self.l), NEUTRAL_SOLUTION, dtype=np.int32)

        # 计算 S 元胞数量并随机放置在溶液层
        num_solution_cells = self.solution_thickness * self.m * self.l
        num_S_cells = int(num_solution_cells * concentration_ratio)
        if num_S_cells > num_solution_cells:
            print("警告: S元胞浓度过高，设置为溶液层最大数量")
            num_S_cells = num_solution_cells

        if num_S_cells > 0:
            S_indices_flat = random.sample(range(num_solution_cells), num_S_cells)
            for flat_idx in S_indices_flat:
                x = flat_idx // (self.m * self.l)
                remainder = flat_idx % (self.m * self.l)
                y = remainder // self.l
                z = remainder % self.l
                corr_grid_cpu[x, y, z] = CORROSIVE_SOLUTION

        # 将金属部分复制到网格下半部分
        corr_grid_cpu[self.solution_thickness:, :, :] = bound_grid_cpu

        # 将 NumPy 数组传输到 GPU
        self.grid_gpu = cp.asarray(corr_grid_cpu)
        print("网格已初始化并传输到 GPU。")

    def get_grid_for_visualization(self) -> np.ndarray:
        """获取网格数据的 NumPy 副本用于可视化"""
        return cp.asnumpy(self.grid_gpu)

    @property
    def shape(self) -> tuple:
        return (self.n_total, self.m, self.l)