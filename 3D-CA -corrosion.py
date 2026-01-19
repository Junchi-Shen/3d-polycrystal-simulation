import numpy as np
import random
import os
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 初始化腐蚀网格
def initialize_corr_grid(bound_grid, concentration_ratio, solution_thickness):
    corr_grid = 4 * np.ones((n + solution_thickness, m, l), dtype=int)  # 上半部分为溶液
    num_S_cells = int(solution_thickness * m * l * concentration_ratio)  # S元胞数量
    S_positions = random.sample(range(solution_thickness * m * l), num_S_cells)

    for pos in S_positions:
        x = pos // (m * l)  # 商为x坐标
        remainder = pos % (m * l)  # 余数为y和z的合成部分
        y = remainder // l  # y坐标
        z = remainder % l  # z坐标
        corr_grid[x, y, z] = 5  # 标记为S元胞

    corr_grid[solution_thickness:, :, :] = bound_grid  # 下半部分为金属元胞
    return corr_grid

# S元胞的随机游走
def random_walk_s(corr_grid, n, m, l):
    p_l, p_r, p_u, p_d, p_f, p_b = 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6  # 六个方向的概率相等
    S_positions = np.argwhere(corr_grid == 5)  # 获取所有S元胞的位置

    for pos in S_positions:
        x, y, z = pos
        direction = random.choices(['l', 'r', 'u', 'd', 'f', 'b'], weights=[p_l, p_r, p_u, p_d, p_f, p_b])[0]

        # 计算新的位置
        if direction == 'l':
            new_x, new_y, new_z = x, (y - 1) % m, z
        elif direction == 'r':
            new_x, new_y, new_z = x, (y + 1) % m, z
        elif direction == 'u':
            new_x, new_y, new_z = max(0, x - 1), y, z
        elif direction == 'd':
            new_x, new_y, new_z = min(n + 4, x + 1), y, z
        elif direction == 'f':
            new_x, new_y, new_z = x, y, (z - 1) % l  # 前移
        elif direction == 'b':
            new_x, new_y, new_z = x, y, (z + 1) % l  # 后移

        # 判断新位置是否为空或者S元胞
        if corr_grid[new_x, new_y, new_z] == 4 or corr_grid[new_x, new_y, new_z] == 5:
            corr_grid[x, y, z], corr_grid[new_x, new_y, new_z] = corr_grid[new_x, new_y, new_z], corr_grid[x, y, z]
        else:
            # 处理与金属元胞接触时的腐蚀反应
            if corr_grid[new_x, new_y, new_z] in [0, 1, 2, 3]:
                if corrosion_occurs(corr_grid, new_x, new_y, new_z):
                    corr_grid[new_x, new_y, new_z] = 6  # 腐蚀产物
                    corr_grid[x, y, z] = 4  # S元胞转变为N元胞
                    # 在corr_grid中随机选择一个N元胞变成S元胞
                    turn_random_n_to_s(corr_grid, n, m, l)

# 判断是否发生腐蚀
def corrosion_occurs(corr_grid, x, y, z):
    cell_type = corr_grid[x, y, z]
    corrosion_probability = corrosion_base_probabilities[cell_type]
    return random.random() < corrosion_probability

# 将随机一个N元胞转化为S元胞
def turn_random_n_to_s(corr_grid, n, m, l):
    N_positions = np.argwhere(corr_grid[:solution_thickness, :, :] == 4)
    if len(N_positions) > 0:
        random_pos = random.choice(N_positions)
        corr_grid[random_pos[0], random_pos[1], random_pos[2]] = 5  # 转化为S元胞

# 获取三维元胞的摩尔邻居
def get_moore_neighbors(corr_grid, x, y, z):
    neighbors = []
    n, m, l = corr_grid.shape
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue  # 自己不作为邻居
                new_x, new_y, new_z = x + i, y + j, z + k
                if 0 <= new_x < n and 0 <= new_y < m and 0 <= new_z < l:
                    neighbors.append((new_x, new_y, new_z))
    return neighbors

# 腐蚀产物元胞P的随机移动
def move_corrosion_products(corr_grid, n, m, l):
    P_positions = np.argwhere(corr_grid == 6)  # 获取所有腐蚀产物P的位置

    for pos in P_positions:
        x, y, z = pos
        if random.random() < p_Pmove:
            neighbors = get_moore_neighbors(corr_grid, x, y, z)
            valid_neighbors = [(nx, ny, nz) for nx, ny, nz in neighbors if corr_grid[nx, ny, nz] in [4, 5]]

            if valid_neighbors:
                new_x, new_y, new_z = random.choice(valid_neighbors)
                corr_grid[x, y, z], corr_grid[new_x, new_y, new_z] = corr_grid[new_x, new_y, new_z], corr_grid[x, y, z]

# 可视化腐蚀过程
def visualize_corr_grid(corr_grid, step, show_plot=False, save_plot=True):
    plt.figure(figsize=(6, 6))
    cmap = ListedColormap(['gray', 'blue', 'yellow', 'red', 'white', 'purple', 'black'])
    plt.imshow(corr_grid[:, :, 50], cmap=cmap, interpolation='nearest')  # 显示第一层作为示例
    plt.title(f"Step: {step}")
    plt.colorbar()

    if save_plot:
        plt.savefig(f"corrosion_step_{step}.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

# 主仿真函数
def simulate_corrosion(bound_grid, steps):

    for step in tqdm.tqdm(range(steps)):
        random_walk_s(corr_grid, n, m, l)
        move_corrosion_products(corr_grid, n, m, l)
        if step % 1000 == 0:
            visualize_corr_grid(corr_grid, step, show_plot=True, save_plot=False)

# 参数设置
if __name__ == '__main__':
    # 0: 晶粒铝 1: 晶界铝 2:析出相
    # 4:中性溶液 5: 腐蚀溶液 6: 腐蚀产物
    grid_size, num_grains, thickness, precipitate_ratio = 100, 100, 2, 0.8
    bound_grid = np.load(f"grid_alloy_100_20_0.8.npy")  # 加载 bound_grid 三维数组
    concentration_ratio = 0.2
    solution_thickness = 5
    corrosion_base_probabilities = {0: 0.0002, 1: 0.6, 2: 0.05, 3: 0.8}
    p_Pmove = 0.2

    n, m, l = bound_grid.shape
    corr_grid = initialize_corr_grid(bound_grid, concentration_ratio, solution_thickness)
    simulate_corrosion(bound_grid, steps=10000)