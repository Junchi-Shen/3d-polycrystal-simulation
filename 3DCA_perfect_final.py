import cupy as cp
import random
import os
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy

# 细胞类型常量
CELL_TYPE_M0 = 0  # Metal Grain
CELL_TYPE_M1 = 1  # Metal Boundary
CELL_TYPE_M2 = 2  # Metal Precipitate
CELL_TYPE_M_OTHER = 3 
CELL_TYPE_N = 4   # Neutral Solution
CELL_TYPE_S = 5   # Acidic Solute
CELL_TYPE_P = 6   # Corrosion Product

# --- 自定义CUDA内核定义 ---

# S细胞随机游走和腐蚀的CUDA内核
random_walk_s_kernel = cp.RawKernel(r'''
extern "C" __global__
void random_walk_s(int *grid, int n_total, int m, int l, 
                   float prob_M0, float prob_M1, float prob_M2,
                   int *random_dirs, float *random_vals, int *counter)
{
    // Get thread index (corresponding to a grid point)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_size = n_total * m * l;
    
    if (idx >= total_size) return;
    
    // Calculate 3D coordinates
    int x = idx / (m * l);
    int y = (idx % (m * l)) / l;
    int z = (idx % (m * l)) % l;
    
    // Only process S cells
    if (grid[idx] != 5) return; // 5 = S cell
    
    // Get random direction and probability value
    int direction = random_dirs[idx];
    float rand_val = random_vals[idx];
    
    // Calculate new position
    int new_x = x;
    int new_y = y;
    int new_z = z;
    
    // Update position based on direction
    if (direction == 0) new_y = (y - 1 + m) % m;      // Left
    else if (direction == 1) new_y = (y + 1) % m;     // Right
    else if (direction == 2) new_x = max(0, x - 1);   // Up
    else if (direction == 3) new_x = min(n_total - 1, x + 1); // Down
    else if (direction == 4) new_z = (z - 1 + l) % l; // Front
    else if (direction == 5) new_z = (z + 1) % l;     // Back
    
    // Calculate the index of the new position
    int new_idx = new_x * m * l + new_y * l + new_z;
    
    // Get target cell type
    int target_type = grid[new_idx];
    
    // Check if target is N or S cell (can be swapped)
    if (target_type == 4 || target_type == 5) { // 4 = N, 5 = S
        // Atomic operation to avoid race conditions
        atomicExch(&grid[idx], target_type);
        atomicExch(&grid[new_idx], 5); // 5 = S
    }
    // Check if target is metal and corrosion may occur
    else if (target_type <= 2) { // Metal types: 0, 1, 2
        float corr_prob = 0.0f;
        if (target_type == 0) corr_prob = prob_M0;
        else if (target_type == 1) corr_prob = prob_M1;
        else if (target_type == 2) corr_prob = prob_M2;
        
        // Check if corrosion occurs
        if (rand_val < corr_prob) {
            atomicExch(&grid[idx], 4);     // S -> N
            atomicExch(&grid[new_idx], 6); // M -> P
            atomicAdd(counter, 1);         // Increment corrosion counter
        }
    }
}
''', 'random_walk_s')

# N到S细胞转换的CUDA内核
turn_n_to_s_kernel = cp.RawKernel(r'''
extern "C" __global__
void turn_n_to_s(int *grid, int solution_thickness, int m, int l, 
                 float *random_vals, int num_to_convert, int *converted)
{
    // Get thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int solution_size = solution_thickness * m * l;
    
    if (idx >= solution_size || *converted >= num_to_convert) return;
    
    // Calculate 3D coordinates within the solution layer
    int x = idx / (m * l);
    int y = (idx % (m * l)) / l;
    int z = (idx % (m * l)) % l;
    
    // Ensure we're within the solution layer
    if (x >= solution_thickness) return;
    
    // Calculate actual grid index
    int grid_idx = x * m * l + y * l + z;
    
    // Only process N cells
    if (grid[grid_idx] == 4) { // 4 = N
        float threshold = (float)num_to_convert / solution_size;
        if (random_vals[idx] < threshold) {
            // Try to convert this N cell to S
            if (atomicCAS(&grid[grid_idx], 4, 5) == 4) { // 4 = N, 5 = S
                atomicAdd(converted, 1);
            }
        }
    }
}
''', 'turn_n_to_s')

# 腐蚀产物移动的CUDA内核
move_corrosion_products_kernel = cp.RawKernel(r'''
extern "C" __global__
void move_corrosion_products(int *grid, int n_total, int m, int l, 
                             float p_Pmove, float *random_move, int *random_dirs)
{
    // Get thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_size = n_total * m * l;
    
    if (idx >= total_size) return;
    
    // Calculate 3D coordinates
    int x = idx / (m * l);
    int y = (idx % (m * l)) / l;
    int z = (idx % (m * l)) % l;
    
    // Check if it's a P cell and decide whether to move
    if (grid[idx] == 6 && random_move[idx] < p_Pmove) { // 6 = P
        // Get random movement direction
        int direction = random_dirs[idx];
        
        // Calculate new position
        int new_x = x;
        int new_y = y;
        int new_z = z;
        
        // Update position based on direction
        if (direction == 0) new_x = max(0, x - 1);
        else if (direction == 1) new_x = min(n_total - 1, x + 1);
        else if (direction == 2) new_y = (y - 1 + m) % m;
        else if (direction == 3) new_y = (y + 1) % m;
        else if (direction == 4) new_z = (z - 1 + l) % l;
        else if (direction == 5) new_z = (z + 1) % l;
        
        // Calculate the index of the new position
        int new_idx = new_x * m * l + new_y * l + new_z;
        
        // Get target cell type
        int target_type = grid[new_idx];
        
        // Only move if target is N or S cell
        if (target_type == 4 || target_type == 5) { // 4 = N, 5 = S
            // Atomic operation to avoid race conditions
            atomicExch(&grid[idx], target_type);
            atomicExch(&grid[new_idx], 6); // 6 = P
        }
    }
}
''', 'move_corrosion_products')

# --- 主模拟函数（使用自定义CUDA内核）---

def initialize_corr_grid_gpu(bound_grid_np, concentration_ratio, solution_thickness):
    """初始化GPU上的腐蚀网格。"""
    n_metal, m, l = bound_grid_np.shape
    n_total = n_metal + solution_thickness
    grid_shape = (n_total, m, l)

    # 在GPU上创建网格，初始化上部为N
    corr_grid = cp.full(grid_shape, CELL_TYPE_N, dtype=cp.int32)

    # 计算S细胞的数量
    total_solution_cells = solution_thickness * m * l
    num_s_cells = int(total_solution_cells * concentration_ratio)

    if num_s_cells > 0:
        # 为S细胞生成随机的扁平索引
        s_indices_flat = cp.random.choice(cp.arange(total_solution_cells), size=num_s_cells, replace=False)

        # 将扁平索引转换为溶液层内的3D坐标
        x_indices = s_indices_flat // (m * l)
        remainder = s_indices_flat % (m * l)
        y_indices = remainder // l
        z_indices = remainder % l

        # 放置S细胞
        corr_grid[x_indices, y_indices, z_indices] = CELL_TYPE_S

    # 放置金属部分 - 将numpy数组传输到GPU
    bound_grid_gpu = cp.asarray(bound_grid_np, dtype=cp.int32)
    corr_grid[solution_thickness:, :, :] = bound_grid_gpu

    return corr_grid

def simulate_corrosion_gpu_with_kernel(bound_grid_np, steps, concentration_ratio, solution_thickness, corrosion_base_probabilities, p_Pmove, vis_interval=100, vis_slice=50):
    """使用自定义CUDA内核的主模拟循环。"""
    
    print("Initializing grid on GPU with CUDA kernels...")
    corr_grid_gpu = initialize_corr_grid_gpu(bound_grid_np, concentration_ratio, solution_thickness)
    print("Grid initialization complete.")
    
    n_total, m, l = corr_grid_gpu.shape
    total_size = n_total * m * l
    
    # 确保概率映射使用前面定义的整数键
    prob_map = {int(k): v for k, v in corrosion_base_probabilities.items()}
    prob_M0 = prob_map.get(CELL_TYPE_M0, 0.0)
    prob_M1 = prob_map.get(CELL_TYPE_M1, 0.0)
    prob_M2 = prob_map.get(CELL_TYPE_M2, 0.0)
    
    # 为每一步预分配随机数数组
    threads_per_block = 256
    blocks_per_grid = (total_size + threads_per_block - 1) // threads_per_block
    
    # 创建corrosion计数器
    counter = cp.zeros(1, dtype=cp.int32)
    
    print("Starting simulation...")
    for step in tqdm.tqdm(range(steps)):
        # 步骤1：生成本次迭代需要的随机数
        random_dirs = cp.random.randint(0, 6, total_size, dtype=cp.int32)
        random_vals = cp.random.rand(total_size, dtype=cp.float32)
        random_move = cp.random.rand(total_size, dtype=cp.float32)
        
        # 重置腐蚀计数器
        counter[0] = 0
        
        # 步骤2：启动S细胞随机游走和腐蚀的内核
        random_walk_s_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (corr_grid_gpu.ravel(), cp.int32(n_total), cp.int32(m), cp.int32(l),
             cp.float32(prob_M0), cp.float32(prob_M1), cp.float32(prob_M2),
             random_dirs, random_vals, counter)
        )
        
        # 步骤3：获取腐蚀计数并转换相应数量的N为S
        num_corroded = int(counter.get()[0])
        if num_corroded > 0:
            # 重置转换计数器
            converted = cp.zeros(1, dtype=cp.int32)
            # 生成新的随机数用于N到S的转换
            random_solution = cp.random.rand(solution_thickness * m * l, dtype=cp.float32)
            # 启动N到S转换内核
            solution_blocks = (solution_thickness * m * l + threads_per_block - 1) // threads_per_block
            turn_n_to_s_kernel(
                (solution_blocks,), (threads_per_block,),
                (corr_grid_gpu.ravel(), cp.int32(solution_thickness), cp.int32(m), cp.int32(l),
                 random_solution, cp.int32(num_corroded), converted)
            )
        
        # 步骤4：启动腐蚀产物移动内核
        move_corrosion_products_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (corr_grid_gpu.ravel(), cp.int32(n_total), cp.int32(m), cp.int32(l),
             cp.float32(p_Pmove), random_move, random_dirs)
        )
        
        # 步骤5：可视化（定期）
        if step % vis_interval == 0:
            visualize_corr_grid(corr_grid_gpu, step, z_slice_index=vis_slice, show_plot=False)
    
    print("Simulation finished.")
    visualize_corr_grid(corr_grid_gpu, steps, z_slice_index=vis_slice, show_plot=True) # 显示最终状态

def visualize_corr_grid(corr_grid_gpu, step, z_slice_index=50, show_plot=False, save_plot=True, save_dir="corrosion_plots_gpu"):
    """可视化网格的切片。将数据从GPU传输到CPU。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(6, 6))
    # 根据细胞类型值定义颜色图
    # 0:M0(灰色), 1:M1(蓝色), 2:M2(黄色), 3:M_OTHER?(红色), 4:N(白色), 5:S(紫色), 6:P(黑色)
    cmap = ListedColormap(['gray', 'blue', 'yellow', 'red', 'white', 'purple', 'black'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]  # 为离散颜色定义边界
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # 从GPU获取切片 -> 使用.get()或cp.asnumpy()传输到CPU
    grid_slice_cpu = cp.asnumpy(corr_grid_gpu[:, :, z_slice_index])

    plt.imshow(grid_slice_cpu, cmap=cmap, norm=norm, interpolation='nearest')
    plt.title(f"GPU Corrosion Step: {step} (Slice Z={z_slice_index})")

    # 创建带标签的颜色条
    cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(['M0', 'M1', 'M2', 'M?', 'N', 'S', 'P'])

    if save_plot:
        plt.savefig(os.path.join(save_dir, f"corrosion_step_{step:05d}.png"))

    if show_plot:
        plt.show()
    else:
        plt.close()

# --- 主参数和执行 ---
if __name__ == '__main__':
    # 确保CuPy已安装并且有兼容的GPU可用
    
    # 参数（根据需要调整）
    concentration_ratio = 0.2
    solution_thickness = 10  # 增加厚度以获得更好的可视化/效果
    corrosion_base_probabilities = {
        CELL_TYPE_M0: 0.0002, # 晶粒铝
        CELL_TYPE_M1: 0.6,    # 边界铝
        CELL_TYPE_M2: 0.05    # 沉淀物
    }
    p_Pmove = 0.2  # P细胞移动的概率
    simulation_steps = 5000  # 模拟步骤数
    visualization_interval = 100  # 每N步可视化一次
    visualization_slice_z = 50  # 要可视化的Z切片索引

    # 加载初始金属网格（NumPy数组）
    try:
        bound_grid_path = r"grid_alloy_100_20_0.8.npy"
        bound_grid_np = numpy.load(bound_grid_path)
        print(f"Loaded metal grid from {bound_grid_path} with shape: {bound_grid_np.shape}")

        # 如果网格尺寸较小，调整可视化切片
        if visualization_slice_z >= bound_grid_np.shape[2]:
            visualization_slice_z = bound_grid_np.shape[2] // 2  # 如果指定的超出范围，使用中间切片
            print(f"Adjusted visualization Z slice to: {visualization_slice_z}")

    except FileNotFoundError:
        print(f"Error: Metal grid file not found at {bound_grid_path}")
        print("Please ensure the .npy file exists and the path is correct.")
        exit()
    except Exception as e:
        print(f"Error loading grid: {e}")
        exit()

    # 运行模拟
    simulate_corrosion_gpu_with_kernel(
        bound_grid_np,
        steps=simulation_steps,
        concentration_ratio=concentration_ratio,
        solution_thickness=solution_thickness,
        corrosion_base_probabilities=corrosion_base_probabilities,
        p_Pmove=p_Pmove,
        vis_interval=visualization_interval,
        vis_slice=visualization_slice_z
    )
