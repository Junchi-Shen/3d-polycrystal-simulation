# simulation/simulator.py
import cupy as cp
import numpy as np
from numba import cuda
import random

from parameters import SimulationParameters, CorrosionRule
from simulation.volume import VolumeData
from simulation.kernels import (random_walk_s_kernel_custom_rng, turn_n_to_s_kernel,
                                move_corrosion_products_kernel_custom_rng)
from utils.cuda_helpers import calculate_launch_config, initialize_rng_states
# 显式导入常量
from constants import GRAIN_AL, GRAIN_BOUNDARY_AL, PRECIPITATE, METAL_TYPE_3

class CorrosionSimulator:
    """负责管理和执行腐蚀模拟"""
    def __init__(self, sim_params: SimulationParameters, rule: CorrosionRule):
        self.sim_params = sim_params
        self.rule = rule

        # 确定网格形状
        if sim_params.grid_shape_metal is None:
             raise ValueError("SimulationParameters must be initialized with a valid grid_file.")
        n_metal, m, l = sim_params.grid_shape_metal
        self.n_total = n_metal + sim_params.solution_thickness
        self.shape = (self.n_total, m, l)

        # 创建 VolumeData 实例并初始化
        self.volume_data = VolumeData(self.shape, sim_params.solution_thickness)
        # 在 __init__ 中调用初始化，确保模拟器创建时就有有效数据
        self.volume_data.initialize_grid(sim_params.bound_grid_cpu, sim_params.concentration_ratio)

        # 配置 CUDA 启动参数
        self.threads_per_block = (8, 8, 4) # 可以调整
        self.blocks_per_grid, _, self.actual_threads_launched = calculate_launch_config(
            self.shape, self.threads_per_block
        )

        print(f"CUDA 配置: Threads/Block={self.threads_per_block}, Blocks/Grid={self.blocks_per_grid}")
        print(f"总共启动的 CUDA 线程数: {self.actual_threads_launched}")

        # 初始化 RNG 状态
        random.seed(sim_params.rng_seed) # 为 CPU 随机数设置种子 (用于初始化)
        self.kernel_rng_states = initialize_rng_states(self.actual_threads_launched, sim_params.rng_seed)

        # 初始化原子计数器
        self.s_to_create_count_gpu = cp.zeros(1, dtype=cp.int32)

        # 预加载规则参数到实例变量以便传递给内核 (使用导入的常量)
        self.p_corr_grain = np.float32(self.rule.corrosion_probabilities.get(GRAIN_AL, 0.0))
        self.p_corr_boundary = np.float32(self.rule.corrosion_probabilities.get(GRAIN_BOUNDARY_AL, 0.0))
        self.p_corr_precipitate = np.float32(self.rule.corrosion_probabilities.get(PRECIPITATE, 0.0))
        self.p_corr_type3 = np.float32(self.rule.corrosion_probabilities.get(METAL_TYPE_3, 0.0))
        self.p_pmove_rule = np.float32(self.rule.p_Pmove) # 确保类型为 float32

    def step(self):
        """执行一个模拟步骤"""
        # 清零计数器
        self.s_to_create_count_gpu[0] = 0

        # a. S 元胞随机游走和腐蚀
        random_walk_s_kernel_custom_rng[self.blocks_per_grid, self.threads_per_block](
            self.volume_data.grid_gpu, self.kernel_rng_states,
            self.n_total, self.shape[1], self.shape[2], self.sim_params.solution_thickness,
            self.p_corr_grain, self.p_corr_boundary, self.p_corr_precipitate, self.p_corr_type3,
            self.s_to_create_count_gpu
        )
        cuda.synchronize() # 确保内核完成

        # b. N 元胞转换为 S 元胞
        s_needed = int(self.s_to_create_count_gpu[0]) # 在同步后读取
        if s_needed > 0:
            turn_n_to_s_kernel[self.blocks_per_grid, self.threads_per_block](
                self.volume_data.grid_gpu,
                self.n_total, self.shape[1], self.shape[2], self.sim_params.solution_thickness,
                self.s_to_create_count_gpu
            )
            cuda.synchronize() # 确保内核完成

        # c. 腐蚀产物 P 移动
        move_corrosion_products_kernel_custom_rng[self.blocks_per_grid, self.threads_per_block](
            self.volume_data.grid_gpu, self.kernel_rng_states,
            self.n_total, self.shape[1], self.shape[2],
            self.p_pmove_rule # 使用 float32 概率
        )
        cuda.synchronize() # 确保内核完成

    def get_grid_for_visualization(self) -> np.ndarray:
        """获取用于可视化的网格数据副本"""
        return self.volume_data.get_grid_for_visualization()