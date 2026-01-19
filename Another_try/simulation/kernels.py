# simulation/kernels.py
import numba
from numba import cuda
import numpy as np # Numba 内核中可能需要 np 类型

# 显式导入常量
from constants import (GRAIN_AL, GRAIN_BOUNDARY_AL, PRECIPITATE, METAL_TYPE_3,
                       NEUTRAL_SOLUTION, CORROSIVE_SOLUTION, CORROSION_PRODUCT)

# --- 自定义 Xorshift32 PRNG (CUDA 设备函数) ---
@cuda.jit(device=True)
def xorshift32_device(state):
    """简单的 Xorshift32 算法。返回 uint32 随机数。"""
    if state[0] == 0: state[0] = np.uint32(1) # 状态不能为0，使用 np.uint32 确保类型
    x = state[0]
    x ^= (x << np.uint32(13)) & np.uint32(0xFFFFFFFF)
    x ^= (x >> np.uint32(17)) & np.uint32(0xFFFFFFFF)
    x ^= (x << np.uint32(5)) & np.uint32(0xFFFFFFFF)
    state[0] = x
    return x

@cuda.jit(device=True)
def get_float_rand_device(state):
    """使用 xorshift32 生成 [0.0, 1.0) 范围内的伪随机浮点数。"""
    uint32_max = np.uint32(0xFFFFFFFF)
    return float(xorshift32_device(state)) / (float(uint32_max) + 1.0)

# --- CUDA 内核 ---

@cuda.jit(device=True)
def _handle_s_interaction(corr_grid, kernel_rng_states, tid, x, y, z, n_total, m, l,
                          p_corr_grain, p_corr_boundary, p_corr_precipitate, p_corr_type3,
                          s_to_create_count):
    """处理单个 S 元胞交互的设备函数 - 使用扁平化索引进行原子操作"""
    local_rng_state = cuda.local.array(shape=1, dtype=numba.uint32)
    local_rng_state[0] = kernel_rng_states[tid] # 加载状态

    # 1. 随机选择移动方向
    rand_val_dir = get_float_rand_device(local_rng_state)
    direction = int(rand_val_dir * 6)

    # 2. 计算新位置
    new_x, new_y, new_z = x, y, z
    if direction == 0: new_y = (y - 1 + m) % m
    elif direction == 1: new_y = (y + 1) % m
    elif direction == 2: new_x = max(0, x - 1)
    elif direction == 3: new_x = min(n_total - 1, x + 1)
    elif direction == 4: new_z = (z - 1 + l) % l
    elif direction == 5: new_z = (z + 1) % l

    # 3. 获取目标位置状态
    target_state = corr_grid[new_x, new_y, new_z]

    # 4. 处理移动/交互逻辑
    if target_state == NEUTRAL_SOLUTION:
        # --- 修改点：计算扁平化索引 ---
        flat_idx_s_n = new_x * m * l + new_y * l + new_z
        # --- 修改点：使用扁平化索引进行 CAS ---
        # original_target_value = cuda.atomic.compare_and_swap(corr_grid.ravel(), flat_idx_s_n, np.int32(NEUTRAL_SOLUTION), CORROSIVE_SOLUTION)
        # ^-- 注意这里可能需要用 .ravel()，如果 Numba 不支持直接在多维数组上用扁平索引的话
        #     但是 .ravel() 可能创建副本，导致原子操作无效。先尝试不用 .ravel()。
        #     如果报错，再尝试 corr_grid.ravel()。如果还不行，说明此路不通。
        #     我们先假设 Numba 允许在 3D 数组上用 1D 索引进行原子操作（虽然文档不明确）。
        #     更新：更标准的方式是 Numba 支持在多维数组上直接用整数索引，所以不用 .ravel()
        original_target_value = cuda.atomic.compare_and_swap(corr_grid, flat_idx_s_n, np.int32(NEUTRAL_SOLUTION), CORROSIVE_SOLUTION)

        if original_target_value == NEUTRAL_SOLUTION:
            corr_grid[x, y, z] = NEUTRAL_SOLUTION
            
    elif target_state == CORROSIVE_SOLUTION:
        pass
    elif target_state <= METAL_TYPE_3:
        corrosion_prob = 0.0
        if target_state == GRAIN_AL: corrosion_prob = p_corr_grain
        elif target_state == GRAIN_BOUNDARY_AL: corrosion_prob = p_corr_boundary
        elif target_state == PRECIPITATE: corrosion_prob = p_corr_precipitate
        elif target_state == METAL_TYPE_3: corrosion_prob = p_corr_type3

        rand_corr = get_float_rand_device(local_rng_state)
        if rand_corr < corrosion_prob:
            # --- 修改点：计算扁平化索引 ---
            flat_idx_s_metal = new_x * m * l + new_y * l + new_z
            # --- 修改点：使用扁平化索引进行 CAS ---
            original_metal_value = cuda.atomic.compare_and_swap(corr_grid, flat_idx_s_metal, np.int32(target_state), CORROSION_PRODUCT)
            if original_metal_value == target_state:
                corr_grid[x, y, z] = NEUTRAL_SOLUTION
                cuda.atomic.add(s_to_create_count, 0, 1)

    # 5. 保存更新后的 RNG 状态
    kernel_rng_states[tid] = local_rng_state[0]

@cuda.jit
def random_walk_s_kernel_custom_rng(corr_grid, kernel_rng_states, n_total, m, l, solution_thickness,
                                    p_corr_grain, p_corr_boundary, p_corr_precipitate, p_corr_type3,
                                    s_to_create_count):
    """ 主 CUDA 内核：调度 S 元胞的随机游走和腐蚀反应。 """
    x, y, z = cuda.grid(3)
    if x >= n_total or y >= m or z >= l: return

    if corr_grid[x, y, z] == CORROSIVE_SOLUTION:
        tid = cuda.grid(1)
        # 注意检查边界，防止 actual_threads_launched > n_total*m*l 时越界
        if tid >= kernel_rng_states.shape[0]: return

        # 调用设备函数处理交互 (m 和 l 作为参数传入)
        _handle_s_interaction(corr_grid, kernel_rng_states, tid, x, y, z, n_total, m, l,
                              p_corr_grain, p_corr_boundary, p_corr_precipitate, p_corr_type3,
                              s_to_create_count)

@cuda.jit
def turn_n_to_s_kernel(corr_grid, n_total, m, l, solution_thickness, s_to_create_count):
    """ CUDA 内核：根据计数器将溶液层中的 N 元胞转化为 S 元胞。(不变) """
    x, y, z = cuda.grid(3)
    if x >= solution_thickness or y >= m or z >= l: return

    if corr_grid[x, y, z] == NEUTRAL_SOLUTION:
        if cuda.atomic.sub(s_to_create_count, 0, 1) > 0:
            corr_grid[x, y, z] = CORROSIVE_SOLUTION
        else:
            cuda.atomic.add(s_to_create_count, 0, 1)

@cuda.jit(device=True)
def _handle_p_movement(corr_grid, kernel_rng_states, tid, x, y, z, n_total, m, l, p_Pmove):
    """处理单个 P 元胞移动的设备函数 - 使用扁平化索引进行原子操作"""
    local_rng_state = cuda.local.array(shape=1, dtype=numba.uint32)
    local_rng_state[0] = kernel_rng_states[tid]

    if get_float_rand_device(local_rng_state) < p_Pmove:
        valid_neighbors = cuda.local.array(shape=(26, 3), dtype=numba.int32)
        valid_neighbor_count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0: continue
                    nx, ny, nz = x + i, y + j, z + k
                    if 0 <= nx < n_total and 0 <= ny < m and 0 <= nz < l:
                        neighbor_state = corr_grid[nx, ny, nz]
                        if neighbor_state == NEUTRAL_SOLUTION or neighbor_state == CORROSIVE_SOLUTION:
                            if valid_neighbor_count < 26:
                                valid_neighbors[valid_neighbor_count, 0] = nx
                                valid_neighbors[valid_neighbor_count, 1] = ny
                                valid_neighbors[valid_neighbor_count, 2] = nz
                                valid_neighbor_count += 1

        if valid_neighbor_count > 0:
            rand_val_neighbor = get_float_rand_device(local_rng_state)
            chosen_index = int(rand_val_neighbor * valid_neighbor_count)
            chosen_index = min(chosen_index, valid_neighbor_count - 1)

            target_x = valid_neighbors[chosen_index, 0]
            target_y = valid_neighbors[chosen_index, 1]
            target_z = valid_neighbors[chosen_index, 2]
            target_state = corr_grid[target_x, target_y, target_z] # N=4 或 S=5

            # --- 修改点：计算扁平化索引 ---
            flat_idx_p = target_x * m * l + target_y * l + target_z
            # --- 修改点：使用扁平化索引进行 CAS ---
            original_target_value = cuda.atomic.compare_and_swap(corr_grid, flat_idx_p, np.int32(target_state), CORROSION_PRODUCT)

            if original_target_value == target_state:
                corr_grid[x, y, z] = target_state # 恢复目标原始状态

    kernel_rng_states[tid] = local_rng_state[0] # 写回状态

@cuda.jit
def move_corrosion_products_kernel_custom_rng(corr_grid, kernel_rng_states, n_total, m, l, p_Pmove):
    """ 主 CUDA 内核：调度腐蚀产物 P 的随机移动。 """
    x, y, z = cuda.grid(3)
    if x >= n_total or y >= m or z >= l: return

    if corr_grid[x, y, z] == CORROSION_PRODUCT:
        tid = cuda.grid(1)
        # 注意检查边界
        if tid >= kernel_rng_states.shape[0]: return

        # 调用设备函数处理移动 (m 和 l 作为参数传入)
        _handle_p_movement(corr_grid, kernel_rng_states, tid, x, y, z, n_total, m, l, p_Pmove)