# utils/cuda_helpers.py
import math
import cupy as cp
import numpy as np
import random

def calculate_launch_config(grid_shape: tuple, threads_per_block: tuple) -> tuple:
    """计算 CUDA 内核启动配置"""
    n_total, m, l = grid_shape
    blocks_per_grid_x = math.ceil(n_total / threads_per_block[0])
    blocks_per_grid_y = math.ceil(m / threads_per_block[1])
    blocks_per_grid_z = math.ceil(l / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    # 计算实际启动的总线程数，用于初始化 RNG 状态数组
    actual_threads_launched = blocks_per_grid_x * threads_per_block[0] * \
                              blocks_per_grid_y * threads_per_block[1] * \
                              blocks_per_grid_z * threads_per_block[2]
    return blocks_per_grid, threads_per_block, actual_threads_launched

def initialize_rng_states(num_threads: int, seed: int) -> cp.ndarray:
    """初始化自定义 RNG 状态数组"""
    print(f"为 {num_threads} 个线程初始化 RNG 状态...")
    # 设置 NumPy 的随机种子以确保可重复性
    np_rng = np.random.default_rng(seed)
    # 生成 [1, 2^32) 范围内的随机整数作为初始状态
    # 加 1 避免状态为 0
    np_rng_states = np_rng.integers(1, np.iinfo(np.uint32).max + 1, size=num_threads, dtype=np.uint32)
    kernel_rng_states_gpu = cp.asarray(np_rng_states)
    print("自定义 CUDA RNG 状态已初始化。")
    return kernel_rng_states_gpu