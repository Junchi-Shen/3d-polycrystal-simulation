import os
import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import gc # 导入垃圾收集器
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# PyCUDA导入
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray # 导入gpuarray

# --- 设置CUDA环境变量 (请确保路径正确) ---
# 尝试自动查找或使用默认路径
cuda_path_found = None
possible_paths = [
    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8',
    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.7',
    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0',
    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8',
    'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7',
    # 可以添加更多路径
]
for path in possible_paths:
    if os.path.exists(path):
        cuda_path_found = path
        break

if cuda_path_found:
    print(f"找到CUDA路径: {cuda_path_found}")
    os.environ['CUDA_PATH'] = cuda_path_found
    # 动态添加bin目录到系统PATH和DLL搜索路径
    bin_path = os.path.join(cuda_path_found, 'bin')
    libnvvp_path = os.path.join(cuda_path_found, 'libnvvp')
    os.environ['PATH'] = f"{bin_path};{libnvvp_path};{os.environ['PATH']}"
    try:
        os.add_dll_directory(bin_path)
        print(f"已添加DLL目录: {bin_path}")
    except AttributeError:
        print("警告: os.add_dll_directory 不可用 (需要 Python 3.8+ on Windows). PATH环境变量已设置。")
    except FileNotFoundError:
        print(f"警告: 无法添加DLL目录，路径不存在: {bin_path}")
else:
    print("错误: 未找到有效的CUDA安装路径。请检查 possible_paths 或手动设置 CUDA_PATH 环境变量。")
    sys.exit(1)
# --- END CUDA环境变量设置 ---

# 验证CUDA环境
try:
    print(f"PyCUDA驱动版本: {cuda.get_driver_version()}")
    device_count = cuda.Device.count()
    if device_count == 0:
        raise RuntimeError("未检测到CUDA设备。")
    print(f"CUDA是否可用: True")
    print(f"GPU数量: {device_count}")
    device = cuda.Device(0)
    print(f"当前GPU: {device.name()}")
    print(f"GPU Compute Capability: {device.compute_capability()}")
    print(f"GPU内存大小: {device.total_memory() / (1024**3):.2f} GB")
    # 确定合适的arch
    major, minor = device.compute_capability()
    arch_flag = f"sm_{major}{minor}"
    print(f"将使用的编译架构: -arch={arch_flag}")
except Exception as e:
    print(f"CUDA初始化或环境检查错误: {e}")
    sys.exit(1)

def print_memory_usage():
    """打印当前应用程序的内存使用量"""
    process = psutil.Process(os.getpid())
    host_mem = process.memory_info().rss / (1024 * 1024)
    try:
        free, total = cuda.mem_get_info()
        gpu_mem_used = (total - free) / (1024 * 1024)
        print(f"内存使用: Host={host_mem:.2f} MB, GPU={gpu_mem_used:.2f} MB")
    except Exception as e:
        print(f"内存使用: Host={host_mem:.2f} MB (无法获取GPU内存: {e})")


# 修复后的CUDA代码 - 移除了extern "C"与正确使用随机数生成
cuda_code = """
// Basic XORShift random number generator
__device__ float rand_uniform(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    // Divide by the maximum possible value of a 32-bit unsigned integer + 1.0f
    return (float)x / 4294967296.0f;
}

// Kernel to find the closest seed for each grid point in a chunk
__global__ void get_closest_seed_kernel_optimized(
    const float *g_seeds,  // Pointer to global memory containing seed coordinates (x1, y1, z1, x2, y2, z2, ...)
    const int sx, const int sy, const int sz, // Start index of this chunk in the global grid
    const int bx, const int by, const int bz, // Size of this chunk (block dimensions)
    const int grid_size_total, // Total size of the global grid (e.g., 4000)
    const float domain_size,      // Physical size of the domain (e.g., 1.0)
    int *results,          // Output array to store the index of the closest seed for each point
    int num_seeds)         // Total number of seeds
{
    // Calculate thread's global indices within the chunk
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the bounds of the current chunk
    if(x >= bx || y >= by || z >= bz)
        return;

    // Calculate thread's global index in the entire grid
    int g_x = sx + x;
    int g_y = sy + y;
    int g_z = sz + z;

    // Calculate linear index within the chunk's result array
    int idx = (z * by + y) * bx + x;

    // Calculate the physical coordinates of the grid point
    // Step size between grid points
    float step = domain_size / (grid_size_total - 1); // Use total grid size for step calculation
    float px = g_x * step;
    float py = g_y * step;
    float pz = g_z * step;

    float min_dist_sq = 3.0f * domain_size * domain_size + 1.0f; // Initialize with a value larger than max possible squared distance
    int min_idx = -1; // Initialize with invalid index

    // Iterate through all seeds to find the closest one
    for (int i = 0; i < num_seeds; i++) {
        // Calculate squared Euclidean distance
        float dx = px - g_seeds[i*3 + 0]; // Seed x-coordinate
        float dy = py - g_seeds[i*3 + 1]; // Seed y-coordinate
        float dz = pz - g_seeds[i*3 + 2]; // Seed z-coordinate
        float dist_sq = dx*dx + dy*dy + dz*dz;

        // Update minimum distance and index if a closer seed is found
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            min_idx = i;
        }
    }

    // Store the index of the closest seed in the results array
    results[idx] = min_idx;
}


// Kernel to mark boundaries based on neighboring grain IDs (optimized with shared memory)
__global__ void mark_boundary_kernel_optimized(
    const int *polycrystalline, // Input: Chunk data with grain IDs
    int *boundary,              // Output: Chunk data marking boundaries (1 if boundary, 0 otherwise)
    int sx, int sy, int sz,     // Start index of this chunk in the global grid
    int bx, int by, int bz,     // Size of this chunk
    int gx_total, int gy_total, int gz_total) // Total size of the global grid
{
    // Declare shared memory for caching a portion of the polycrystalline data
    // Size should be blockDim.x * blockDim.y * blockDim.z
    extern __shared__ int s_poly[];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Global indices within the chunk
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int z = blockIdx.z * blockDim.z + tz;

    // Block dimensions
    int block_size_x = blockDim.x;
    int block_size_y = blockDim.y;
    int block_size_z = blockDim.z;

    // Linear index within the shared memory array
    int s_idx = (tz * block_size_y + ty) * block_size_x + tx;

    // Global position in the entire grid
    int gx_pos = sx + x;
    int gy_pos = sy + y;
    int gz_pos = sz + z;

    // Linear index within the chunk's global memory arrays (polycrystalline, boundary)
    int idx = (z * by + y) * bx + x;

    // Load data into shared memory if within chunk bounds
    if(x < bx && y < by && z < bz) {
        s_poly[s_idx] = polycrystalline[idx];
    } else {
        // Optional: Handle out-of-bounds threads in shared memory if needed,
        // e.g., by assigning a sentinel value. Here we rely on the later check.
        // s_poly[s_idx] = -1; // Example sentinel
    }

    // Synchronize threads within the block to ensure all shared memory is loaded
    __syncthreads();

    // Process the point if it's within the chunk bounds
    if(x < bx && y < by && z < bz) {
        boundary[idx] = 0; // Initialize boundary marker to 0 (not a boundary)

        int current_id = s_poly[s_idx];
        bool is_boundary = false;

        // --- Check neighbors using shared memory for intra-block checks ---
        // Check neighbor in -X direction (Left)
        // Condition: Not the first thread in x-dim AND not the first grid point globally
        if (!is_boundary && tx > 0 && gx_pos > 0) {
            int left_s_idx = (tz * block_size_y + ty) * block_size_x + (tx - 1);
            if (current_id != s_poly[left_s_idx]) {
                is_boundary = true;
            }
        }
        // Check neighbor in +X direction (Right)
        // Condition: Not the last thread in x-dim AND not the last grid point within the *chunk* AND not the last grid point globally
        if (!is_boundary && tx < block_size_x - 1 && x < bx - 1 && gx_pos < gx_total - 1) {
            int right_s_idx = (tz * block_size_y + ty) * block_size_x + (tx + 1);
            if (current_id != s_poly[right_s_idx]) {
                is_boundary = true;
            }
        }

        // Check neighbor in -Y direction (Back)
        if (!is_boundary && ty > 0 && gy_pos > 0) {
            int back_s_idx = (tz * block_size_y + (ty - 1)) * block_size_x + tx;
            if (current_id != s_poly[back_s_idx]) {
                is_boundary = true;
            }
        }
        // Check neighbor in +Y direction (Front)
        if (!is_boundary && ty < block_size_y - 1 && y < by - 1 && gy_pos < gy_total - 1) {
            int front_s_idx = (tz * block_size_y + (ty + 1)) * block_size_x + tx;
            if (current_id != s_poly[front_s_idx]) {
                is_boundary = true;
            }
        }

        // Check neighbor in -Z direction (Bottom)
        if (!is_boundary && tz > 0 && gz_pos > 0) {
            int bottom_s_idx = ((tz - 1) * block_size_y + ty) * block_size_x + tx;
            if (current_id != s_poly[bottom_s_idx]) {
                is_boundary = true;
            }
        }
        // Check neighbor in +Z direction (Top)
        if (!is_boundary && tz < block_size_z - 1 && z < bz - 1 && gz_pos < gz_total - 1) {
            int top_s_idx = ((tz + 1) * block_size_y + ty) * block_size_x + tx;
            if (current_id != s_poly[top_s_idx]) {
                is_boundary = true;
            }
        }

        // --- Handle boundary conditions at the edges of the shared memory block ---
        // If a point is at the edge of the shared memory block (e.g., tx == 0),
        // it might need to check a neighbor that wasn't loaded into shared memory.
        // This simplified version relies on the global grid checks implicitly.
        // A more robust implementation might require loading ghost cells into shared memory.

        // If any neighbor has a different ID, mark as boundary
        if (is_boundary) {
            boundary[idx] = 1;
        }
    }
}

// Kernel to mark precipitates on boundary points based on a probability
__global__ void mark_precipitate_kernel_optimized(
    const int *boundary, // Input: Chunk data marking boundaries (1 or 0)
    int *alloy,          // Output: Alloy microstructure (0=grain interior, 1=boundary, 2=precipitate)
    int total_size,      // Total number of points in the chunk
    unsigned int seed,   // Base random seed for the kernel
    float precipitate_ratio) // Probability of forming a precipitate on a boundary
{
    // Global thread index across the entire 1D launch grid for this chunk
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        // Initialize thread-local random state (unique per thread)
        unsigned int rand_state = seed + idx; // Simple way to get different seeds per thread

        // Generate a random number between 0.0 and 1.0
        float rand_val = rand_uniform(&rand_state);

        // If it's a boundary point (boundary[idx] == 1) AND random value is below the ratio, mark as precipitate (2)
        // Otherwise, keep the value from the boundary array (0 or 1)
        alloy[idx] = (boundary[idx] == 1 && rand_val < precipitate_ratio) ? 2 : boundary[idx];
    }
}


// --- Combined Kernel ---
// Processes grain boundary marking and precipitate formation in one go.
__global__ void process_complete_chunk_kernel(
    const float *g_seeds,       // Seed data (used by rand_uniform, not directly here otherwise) - CAN BE REMOVED if rand_uniform doesn't need it
    const int *poly_data,       // Input: Grain IDs for the chunk
    int *boundary_data,         // Output: Boundary markers (0 or 1)
    int *alloy_data,            // Output: Final alloy structure (0, 1, or 2)
    int sx, int sy, int sz,     // Chunk start index
    int bx, int by, int bz,     // Chunk size
    int gx_total, int gy_total, int gz_total, // Global grid size
    unsigned int seed,          // Base random seed
    float precipitate_ratio,    // Precipitate probability
    int num_seeds)              // Number of seeds (passed but not used directly in this kernel version)
{
    extern __shared__ int s_poly[]; // Shared memory for grain IDs

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int z = blockIdx.z * blockDim.z + tz;

    int block_size_x = blockDim.x;
    int block_size_y = blockDim.y;
    int block_size_z = blockDim.z;

    int s_idx = (tz * block_size_y + ty) * block_size_x + tx; // Shared memory index
    int idx = (z * by + y) * bx + x;                         // Global memory index (within chunk)

    int gx_pos = sx + x; // Global grid position
    int gy_pos = sy + y;
    int gz_pos = sz + z;

    // Load grain ID into shared memory
    if(x < bx && y < by && z < bz) {
        s_poly[s_idx] = poly_data[idx];
        boundary_data[idx] = 0; // Initialize boundary to 0
        alloy_data[idx] = 0;    // Initialize alloy to 0 (grain interior)
    }

    __syncthreads(); // Wait for all threads to load shared memory

    // Perform boundary check and precipitate marking if within chunk bounds
    if(x < bx && y < by && z < bz) {
        int current_id = s_poly[s_idx];
        bool is_boundary = false;

        // --- Boundary Check (same logic as mark_boundary_kernel) ---
        if (!is_boundary && tx > 0 && gx_pos > 0) { // Left (-X)
            if (current_id != s_poly[s_idx - 1]) is_boundary = true;
        }
        if (!is_boundary && tx < block_size_x - 1 && x < bx - 1 && gx_pos < gx_total - 1) { // Right (+X)
             if (current_id != s_poly[s_idx + 1]) is_boundary = true;
        }
        if (!is_boundary && ty > 0 && gy_pos > 0) { // Back (-Y)
            if (current_id != s_poly[s_idx - block_size_x]) is_boundary = true;
        }
        if (!is_boundary && ty < block_size_y - 1 && y < by - 1 && gy_pos < gy_total - 1) { // Front (+Y)
            if (current_id != s_poly[s_idx + block_size_x]) is_boundary = true;
        }
        if (!is_boundary && tz > 0 && gz_pos > 0) { // Bottom (-Z)
            if (current_id != s_poly[s_idx - block_size_x * block_size_y]) is_boundary = true;
        }
        if (!is_boundary && tz < block_size_z - 1 && z < bz - 1 && gz_pos < gz_total - 1) { // Top (+Z)
            if (current_id != s_poly[s_idx + block_size_x * block_size_y]) is_boundary = true;
        }
        // --- End Boundary Check ---

        if (is_boundary) {
            boundary_data[idx] = 1; // Mark as boundary

            // --- Precipitate Marking ---
            unsigned int rand_state = seed + idx; // Unique random state per thread
            float rand_val = rand_uniform(&rand_state);

            // If it's a boundary AND random chance occurs, mark as precipitate (2), otherwise mark as boundary (1)
            alloy_data[idx] = (rand_val < precipitate_ratio) ? 2 : 1;
        } else {
            // Not a boundary, so it's grain interior (0) - already initialized
            // alloy_data[idx] = 0; // Redundant, but explicit
        }
    }
}
"""

# --- 编译CUDA代码 ---
# 修改CUDA编译部分的代码
try:
    # 使用之前确定的arch标志
    print("正在编译CUDA代码...")
    # 尝试使用更简单的编译选项，移除可能导致问题的no_extern_c=True
    mod = SourceModule(cuda_code,
                    options=[f"-arch={arch_flag}"], 
                    keep=True) # 保留临时文件以便调试
    
    # 输出可用的CUDA函数名，以便调试
    print("可用的CUDA函数:")
    try:
        functions = mod.get_global_names()
        for name in functions:
            print(f"  - {name}")
    except Exception as e:
        print(f"无法获取函数名列表: {e}")
        
    print("CUDA代码编译成功。")
except cuda.CompileError as e:
    print("CUDA编译错误:")
    print(e.stderr)
    sys.exit(1)
except Exception as e:
    print(f"编译CUDA代码时发生未知错误: {e}")
    sys.exit(1)

# 尝试获取核函数
try:
    get_closest_seed_kernel = mod.get_function("get_closest_seed_kernel_optimized")
    print("成功获取get_closest_seed_kernel_optimized函数")
except Exception as e:
    print(f"获取get_closest_seed_kernel_optimized函数失败: {e}")
    print("尝试使用替代方法获取函数...")
    
    # 查找函数名中包含关键字的函数
    found = False
    for func_name in mod.get_global_names():
        if "get_closest_seed" in func_name:
            print(f"找到替代函数: {func_name}")
            get_closest_seed_kernel = mod.get_function(func_name)
            found = True
            break
    
    if not found:
        print("错误: 无法找到任何与get_closest_seed相关的函数。")
        sys.exit(1)



# 获取核函数引用
get_closest_seed_kernel = mod.get_function("get_closest_seed_kernel_optimized")
mark_boundary_kernel = mod.get_function("mark_boundary_kernel_optimized") # 保留供可能的单独使用
mark_precipitate_kernel = mod.get_function("mark_precipitate_kernel_optimized") # 保留供可能的单独使用
process_complete_chunk_kernel = mod.get_function("process_complete_chunk_kernel")

class OptimizedChunkProcessor:
    """优化版块处理器：使用GPU加速块处理"""

    def __init__(self, grid_size, chunk_size=200, num_workers=None):
        self.grid_size = grid_size
        self.chunk_size = chunk_size

        # 计算处理块的数量
        self.nx_chunks = (grid_size + chunk_size - 1) // chunk_size
        self.ny_chunks = (grid_size + chunk_size - 1) // chunk_size
        self.nz_chunks = (grid_size + chunk_size - 1) // chunk_size

        self.total_chunks = self.nx_chunks * self.ny_chunks * self.nz_chunks
        print(f"网格将被分为 {self.total_chunks} 个块进行处理 ({self.nx_chunks}x{self.ny_chunks}x{self.nz_chunks})")

        # 创建CUDA流，允许异步执行
        self.cuda_stream = cuda.Stream()

        # 创建结果目录
        self.result_dir = Path("./chunks")
        self.result_dir.mkdir(exist_ok=True)
        print(f"块数据将保存在: {self.result_dir.absolute()}")


    def generate_chunk_configs(self):
        """生成所有块的配置"""
        configs = []
        for iz in range(self.nz_chunks):
            for iy in range(self.ny_chunks):
                for ix in range(self.nx_chunks):
                    # 计算当前块的起始坐标 (global grid index)
                    sx = ix * self.chunk_size
                    sy = iy * self.chunk_size
                    sz = iz * self.chunk_size

                    # 计算当前块的实际大小（考虑边界）
                    bx = min(self.chunk_size, self.grid_size - sx)
                    by = min(self.chunk_size, self.grid_size - sy)
                    bz = min(self.chunk_size, self.grid_size - sz)

                    configs.append({
                        'chunk_id': iz * self.ny_chunks * self.nx_chunks + iy * self.nx_chunks + ix,
                        'start': (sx, sy, sz),
                        'size': (bx, by, bz)
                    })
        return configs

import os
import sys
import numpy as np
import random # 确保导入 random 模块
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import gc # 导入垃圾收集器
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# PyCUDA导入
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray # 导入gpuarray

# (之前的CUDA环境设置、验证、print_memory_usage函数、CUDA代码编译、获取核函数等部分保持不变)
# ...
# 确保获取了 process_complete_chunk_kernel:
# get_closest_seed_kernel = mod.get_function("get_closest_seed_kernel_optimized") # for find_closest_grain
# process_complete_chunk_kernel = mod.get_function("process_complete_chunk_kernel") # for process_all
# ...

class OptimizedChunkProcessor:
    """优化版块处理器：使用GPU加速块处理"""

    def __init__(self, grid_size, chunk_size=200, num_workers=None):
        self.grid_size = grid_size
        self.chunk_size = chunk_size

        # 计算处理块的数量
        self.nx_chunks = (grid_size + chunk_size - 1) // chunk_size
        self.ny_chunks = (grid_size + chunk_size - 1) // chunk_size
        self.nz_chunks = (grid_size + chunk_size - 1) // chunk_size

        self.total_chunks = self.nx_chunks * self.ny_chunks * self.nz_chunks
        print(f"网格将被分为 {self.total_chunks} 个块进行处理 ({self.nx_chunks}x{self.ny_chunks}x{self.nz_chunks})")

        # 创建CUDA流，允许异步执行
        self.cuda_stream = cuda.Stream()

        # 创建结果目录
        self.result_dir = Path("./chunks")
        self.result_dir.mkdir(exist_ok=True)
        print(f"块数据将保存在: {self.result_dir.absolute()}")


    def generate_chunk_configs(self):
        """生成所有块的配置"""
        configs = []
        for iz in range(self.nz_chunks):
            for iy in range(self.ny_chunks):
                for ix in range(self.nx_chunks):
                    # 计算当前块的起始坐标 (global grid index)
                    sx = ix * self.chunk_size
                    sy = iy * self.chunk_size
                    sz = iz * self.chunk_size

                    # 计算当前块的实际大小（考虑边界）
                    bx = min(self.chunk_size, self.grid_size - sx)
                    by = min(self.chunk_size, self.grid_size - sy)
                    bz = min(self.chunk_size, self.grid_size - sz)

                    configs.append({
                        'chunk_id': iz * self.ny_chunks * self.nx_chunks + iy * self.nx_chunks + ix,
                        'start': (sx, sy, sz),
                        'size': (bx, by, bz)
                    })
        return configs

    # --- FINAL CORRECTED VERSION of process_chunk_optimized ---
    def process_chunk_optimized(self, config, seeds_gpu, num_seeds_int, size, func_name, precipitate_ratio=0.5):
        """优化版块处理函数 (在主进程中调用，提交任务到GPU)"""
        start_time = time.time() # 总计时

        chunk_id = config['chunk_id']
        sx, sy, sz = config['start']
        bx, by, bz = config['size']

        print(f"[块 {chunk_id}] 开始处理: 位置({sx},{sy},{sz}), 大小({bx}x{by}x{bz})")

        total_points = bx * by * bz
        if total_points == 0: # 跳过空块
            print(f"[块 {chunk_id}] 跳过空块")
            return None if func_name != 'process_all' else (None, None)

        # --------------- Step 3: Find Closest Grain ---------------
        if func_name == 'find_closest_grain':
            alloc_start = time.time()
            try:
                # 分配GPU内存用于结果
                results_gpu = cuda.mem_alloc(total_points * np.dtype(np.int32).itemsize)
            except cuda.MemoryError as e:
                print(f"\n错误: 块 {chunk_id} (find_closest_grain) 分配结果内存失败 ({total_points*4/1024**2:.2f} MB)。GPU内存不足。")
                print_memory_usage()
                raise e

            # 配置和运行优化后的核函数
            block_dim = (16, 8, 8) # 根据GPU调整
            grid_dim = (
                (bx + block_dim[0] - 1) // block_dim[0],
                (by + block_dim[1] - 1) // block_dim[1],
                (bz + block_dim[2] - 1) // block_dim[2]
            )

            kernel_start = time.time()
            # 异步执行核函数 - 直接传递seeds_gpu句柄
            get_closest_seed_kernel(
                seeds_gpu, # 传递GPU数组句柄
                np.int32(sx), np.int32(sy), np.int32(sz),
                np.int32(bx), np.int32(by), np.int32(bz),
                np.int32(self.grid_size), np.float32(size),
                results_gpu, num_seeds_int, # 传递num_seeds_int
                block=block_dim,
                grid=grid_dim,
                stream=self.cuda_stream
            )
            # print(f"[块 {chunk_id}] find_closest_grain 内核启动完成 (耗时: {time.time() - kernel_start:.4f}秒)")

            # 准备主机内存以接收结果
            chunk_result = np.empty(total_points, dtype=np.int32)

            # 异步将结果从GPU复制回主机
            cuda.memcpy_dtoh_async(chunk_result, results_gpu, self.cuda_stream)

            # 确保异步操作完成
            self.cuda_stream.synchronize()

            # 释放GPU结果内存
            results_gpu.free()

            # 重塑为块形状 (Z, Y, X)
            chunk_result = chunk_result.reshape((bz, by, bx))

            total_time = time.time() - start_time
            print(f"[块 {chunk_id}] find_closest_grain 处理完成 (总耗时: {total_time:.4f}秒)")
            return chunk_result # 返回主机numpy数组

        # --------------- Step 4: Process Boundaries and Precipitates ---------------
        elif func_name == 'process_all':
            print(f"[块 {chunk_id}] 开始 process_all 处理 (边界+析出相)")
            # 加载当前块的晶粒ID数据
            load_start = time.time()
            poly_file = self.result_dir / f"poly_chunk_{chunk_id}.npy"
            if not poly_file.exists():
                print(f"警告: 未找到块 {chunk_id} 的晶粒ID数据 ({poly_file})。跳过处理。")
                return None, None

            try:
                # print(f"[块 {chunk_id}] 加载晶粒ID数据: {poly_file}")
                poly_data = np.load(poly_file)
                # print(f"[块 {chunk_id}] 数据加载完成 (耗时: {time.time()-load_start:.4f}秒)")

                # 确保数据类型和连续性
                if not poly_data.flags['C_CONTIGUOUS'] or poly_data.dtype != np.int32:
                    # print(f"[块 {chunk_id}] 转换数据为 C连续 int32")
                    poly_data = np.ascontiguousarray(poly_data, dtype=np.int32)

                # 配置内核执行参数
                block_dim = (8, 8, 8) # 应与上面保持一致或根据此内核优化
                grid_dim = (
                    (bx + block_dim[0] - 1) // block_dim[0],
                    (by + block_dim[1] - 1) // block_dim[1],
                    (bz + block_dim[2] - 1) // block_dim[2]
                )

                # 准备GPU内存
                gpu_alloc_start = time.time()
                # print(f"[块 {chunk_id}] 分配GPU内存: 3 x {poly_data.nbytes/1024**2:.2f} MB")
                poly_gpu = cuda.mem_alloc(poly_data.nbytes)
                boundary_gpu = cuda.mem_alloc(poly_data.nbytes) # For boundary result
                alloy_gpu = cuda.mem_alloc(poly_data.nbytes)    # For final alloy result
                # print(f"[块 {chunk_id}] GPU内存分配完成 (耗时: {time.time()-gpu_alloc_start:.4f}秒)")

                # 将多晶体数据从CPU复制到GPU
                # print(f"[块 {chunk_id}] 将多晶体数据复制到GPU")
                cuda.memcpy_htod_async(poly_gpu, poly_data, self.cuda_stream)

                # --- CORRECTED KERNEL CALL SECTION ---
                # Calculate required shared memory size (bytes) for the combined kernel
                shared_mem_size = block_dim[0] * block_dim[1] * block_dim[2] * np.dtype(np.int32).itemsize

                # Generate a random seed for this chunk's precipitate calculation
                chunk_seed = np.uint32(random.randint(0, 2**32 - 1))

                # print(f"[块 {chunk_id}] 启动组合内核 (边界+析出相) (共享内存: {shared_mem_size} bytes)")
                kernel_start = time.time()
                process_complete_chunk_kernel( # <-- Single, correct kernel call
                    seeds_gpu,              # Pass seeds_gpu handle
                    poly_gpu,               # Input: Grain IDs for the chunk
                    boundary_gpu,           # Output: Boundary markers
                    alloy_gpu,              # Output: Final alloy structure
                    np.int32(sx), np.int32(sy), np.int32(sz),       # Chunk start index
                    np.int32(bx), np.int32(by), np.int32(bz),       # Chunk size
                    np.int32(self.grid_size), np.int32(self.grid_size), np.int32(self.grid_size), # Global grid size
                    chunk_seed,             # Random seed for this chunk
                    np.float32(precipitate_ratio), # Precipitate probability
                    num_seeds_int,          # Total number of seeds
                    block=block_dim,
                    grid=grid_dim,
                    stream=self.cuda_stream,
                    shared=shared_mem_size  # Specify shared memory size
                )
                # print(f"[块 {chunk_id}] 组合内核启动完成 (耗时: {time.time()-kernel_start:.4f}秒)")
                # --- END OF CORRECTED KERNEL CALL SECTION ---

                # 同步确保所有内核操作完成
                # print(f"[块 {chunk_id}] 等待所有CUDA操作完成")
                sync_start = time.time()
                self.cuda_stream.synchronize()
                # print(f"[块 {chunk_id}] CUDA流同步完成 (耗时: {time.time()-sync_start:.4f}秒)")

                # 从GPU复制结果回CPU
                # print(f"[块 {chunk_id}] 将结果从GPU复制回主机")
                result_transfer_start = time.time()
                boundary_data = np.empty_like(poly_data) # Host array for boundary
                alloy_data = np.empty_like(poly_data)    # Host array for alloy
                cuda.memcpy_dtoh_async(boundary_data, boundary_gpu, self.cuda_stream)
                cuda.memcpy_dtoh_async(alloy_data, alloy_gpu, self.cuda_stream)
                self.cuda_stream.synchronize()  # 确保传输完成
                # print(f"[块 {chunk_id}] 结果传输完成 (耗时: {time.time()-result_transfer_start:.4f}秒)")

                # 释放GPU内存
                # print(f"[块 {chunk_id}] 释放GPU内存")
                poly_gpu.free()
                boundary_gpu.free()
                alloy_gpu.free()

                total_time = time.time() - start_time
                print(f"[块 {chunk_id}] process_all 处理完成 (总耗时: {total_time:.4f}秒)")
                return boundary_data, alloy_data # Return both results

            except cuda.MemoryError as e:
                print(f"\n错误: 块 {chunk_id} (process_all) 分配GPU内存失败。GPU内存不足。")
                print_memory_usage()
                # 清理可能部分分配的内存
                if 'poly_gpu' in locals() and poly_gpu and hasattr(poly_gpu, 'free'): poly_gpu.free()
                if 'boundary_gpu' in locals() and boundary_gpu and hasattr(boundary_gpu, 'free'): boundary_gpu.free()
                if 'alloy_gpu' in locals() and alloy_gpu and hasattr(alloy_gpu, 'free'): alloy_gpu.free()
                raise e
            except Exception as e:
                print(f"\n错误: 处理块 {chunk_id} (process_all) 时出错: {e}")
                import traceback
                traceback.print_exc()
                # 清理可能部分分配的内存 (以防万一)
                if 'poly_gpu' in locals() and poly_gpu and hasattr(poly_gpu, 'free'): poly_gpu.free()
                if 'boundary_gpu' in locals() and boundary_gpu and hasattr(boundary_gpu, 'free'): boundary_gpu.free()
                if 'alloy_gpu' in locals() and alloy_gpu and hasattr(alloy_gpu, 'free'): alloy_gpu.free()
                return None, None

    # 修改为接受seeds_gpu和num_seeds_int
    def process_chunk_batch(self, configs, seeds_gpu, num_seeds_int, size, func_name, precipitate_ratio=0.5):
        """(串行地)处理一批块配置, 将任务提交到GPU流"""
        host_results = {} # 暂时存储主机结果

        for config in configs:
            chunk_id = config['chunk_id']
            if func_name == 'find_closest_grain':
                # 这会向流提交核心启动和异步复制
                chunk_result_host = self.process_chunk_optimized(config, seeds_gpu, num_seeds_int, size, func_name)
                if chunk_result_host is not None:
                    host_results[chunk_id] = chunk_result_host # 存储numpy数组

            elif func_name == 'process_all':
                 # 这会提交加载、核心启动和异步复制到流
                boundary_result_host, alloy_result_host = self.process_chunk_optimized(
                    config, seeds_gpu, num_seeds_int, size, func_name, precipitate_ratio
                )
                if boundary_result_host is not None and alloy_result_host is not None:
                    host_results[chunk_id] = (boundary_result_host, alloy_result_host) # 存储元组

        # --- 同步点 ---
        # 等待这批提交到流中的所有操作 (kernel launches, async copies) 完成
        self.cuda_stream.synchronize()

        # --- 保存结果到磁盘 ---
        # 现在数据肯定已经在主机内存 (host_results), 可以安全保存
        save_count = 0
        if func_name == 'find_closest_grain':
            for chunk_id, chunk_result_host in list(host_results.items()):
                 save_path = self.result_dir / f"poly_chunk_{chunk_id}.npy"
                 try:
                     np.save(save_path, chunk_result_host)
                     save_count += 1
                 except Exception as e:
                     print(f"警告: 保存块 {chunk_id} ({func_name}) 失败: {e}")
                 # 明确删除保存后的大型numpy数组
                 del host_results[chunk_id] # 从字典中移除
                 del chunk_result_host      # 删除引用

        elif func_name == 'process_all':
            for chunk_id, results_tuple in list(host_results.items()):
                boundary_result_host, alloy_result_host = results_tuple
                try:
                    np.save(self.result_dir / f"boundary_chunk_{chunk_id}.npy", boundary_result_host)
                    np.save(self.result_dir / f"alloy_chunk_{chunk_id}.npy", alloy_result_host)
                    save_count += 1
                except Exception as e:
                    print(f"警告: 保存块 {chunk_id} ({func_name}) 失败: {e}")
                # 明确删除大型numpy数组
                del host_results[chunk_id]
                del boundary_result_host
                del alloy_result_host

        # 清理内存
        host_results.clear()
        gc.collect() # 更积极地触发垃圾回收

        return True # 表示批处理尝试已完成


    # 修改为接受seeds_gpu和num_seeds_int
    def process_all_chunks_parallel(self, seeds_gpu, num_seeds_int, size, func_name, precipitate_ratio=0.5):
        """
        在主进程中串行处理所有块配置，将GPU任务提交到流中。
        注意：'Parallel' 指的是GPU并行执行，而不是多CPU进程提交。
        """
        configs = self.generate_chunk_configs()

        print(f"将任务提交到GPU进行处理 ({func_name})...")

        # 使用tqdm显示总体进度
        with tqdm(total=len(configs), desc=f"处理块 ({func_name})") as pbar:
            # 将所有配置作为一个大批次处理 (串行提交到流)
            # process_chunk_batch中的内部循环处理单个块提交
            self.process_chunk_batch(configs, seeds_gpu, num_seeds_int, size, func_name, precipitate_ratio)
            pbar.update(len(configs)) # 在整个批次处理完成后更新进度

        print(f"所有 ({func_name}) 块处理和保存完成.")
        print_memory_usage()


    def assemble_sample_slices(self, slice_type, num_slices=3, save_dir="./results"):
        """组装样本切片用于可视化 (使用线程池加速文件IO和绘图)"""
        save_path_dir = Path(save_dir)
        save_path_dir.mkdir(exist_ok=True)
        print(f"\n正在组装 {slice_type} 切片到目录: {save_path_dir.absolute()}")

        # 确定数据文件前缀
        if slice_type == 'poly':
            prefix = "poly_chunk"
            cmap = 'viridis' # 或其他离散cmap，如'tab20'（如果num_grains很小）
        elif slice_type == 'boundary':
            prefix = "boundary_chunk"
            cmap = 'gray'    # 边界的黑白图
        elif slice_type == 'alloy':
            prefix = "alloy_chunk"
            cmap = 'plasma'  # 或其他区分0, 1, 2的映射
        else:
            print(f"错误: 未知的切片类型: {slice_type}")
            return

        # 获取切片索引 - 选择均匀分布的切片
        if self.grid_size <= num_slices:
             slice_indices = list(range(self.grid_size))
        else:
             slice_indices = [int(i * (self.grid_size - 1) / (num_slices - 1)) if num_slices > 1 else self.grid_size // 2
                              for i in range(num_slices)]


        print(f"将生成 Z 索引处的切片: {slice_indices}")

        # 并行处理切片 (IO/组装/绘图)
        tasks = []
        for i, z_idx in enumerate(slice_indices):
            tasks.append((i, z_idx, slice_type, prefix, cmap, save_path_dir))

        # 使用线程池并行处理切片组装和绘图（IO密集型）
        # 限制工作线程以避免IO或内存不足（如果切片很大）
        max_workers = min(num_slices, mp.cpu_count(), 8)
        print(f"使用 {max_workers} 个线程组装和绘制切片...")
        slice_errors = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用submit处理线程内的潜在异常
            futures = [executor.submit(self._process_single_slice, *task) for task in tasks]

            for future in tqdm(futures, total=len(tasks), desc=f"组装 {slice_type} 切片"):
                try:
                    future.result() # 等待线程完成并检查异常
                except Exception as e:
                    print(f"\n处理切片时发生错误: {e}") # 打印线程中的错误
                    slice_errors += 1

        if slice_errors > 0:
             print(f"警告: {slice_errors} 个切片在处理过程中遇到错误。")
        else:
             print(f"所有 {slice_type} 切片已生成。")


    def _process_single_slice(self, slice_num, z_idx, slice_type, prefix, cmap, save_dir):
        """处理单个切片 - 由线程池调用"""
        start_time_slice = time.time()

        # 确定包含该z索引的块行
        chunk_z_row_idx = z_idx // self.chunk_size
        local_z_in_chunk = z_idx % self.chunk_size # Z索引在块内部的位置

        # 创建一个空的切片图像 (确保dtype匹配npy文件, 通常int32)
        slice_img = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        found_any_chunk = False

        # 遍历加载并组装所有x-y平面上的块
        for y_chunk_idx in range(self.ny_chunks):
            y_start_global = y_chunk_idx * self.chunk_size
            y_end_global = min((y_chunk_idx + 1) * self.chunk_size, self.grid_size)
            chunk_height = y_end_global - y_start_global

            for x_chunk_idx in range(self.nx_chunks):
                x_start_global = x_chunk_idx * self.chunk_size
                x_end_global = min((x_chunk_idx + 1) * self.chunk_size, self.grid_size)
                chunk_width = x_end_global - x_start_global

                # 计算块ID
                chunk_id = chunk_z_row_idx * self.ny_chunks * self.nx_chunks + \
                           y_chunk_idx * self.nx_chunks + \
                           x_chunk_idx
                chunk_file = self.result_dir / f"{prefix}_{chunk_id}.npy"

                if chunk_file.exists():
                    try:
                        chunk_data = np.load(chunk_file) # shape: (bz, by, bx)
                        found_any_chunk = True

                        # 检查块的Z维度是否足够大
                        if local_z_in_chunk < chunk_data.shape[0]:
                            # 提取正确的z平面数据，并放入最终的slice_img中
                            slice_img[y_start_global:y_end_global, x_start_global:x_end_global] = \
                                chunk_data[local_z_in_chunk, :chunk_height, :chunk_width]

                    except Exception as e:
                        print(f"\n警告: 加载或处理块文件 {chunk_file} 出错: {e}")

        if not found_any_chunk:
             print(f"警告: 未找到任何用于切片 {slice_num} (z={z_idx}, type={slice_type}) 的块文件。生成的图像将为空。")

        # 保存和可视化切片
        plt.figure(figsize=(10, 10))
        # 确定vmin/vmax以获得更好的颜色映射一致性
        if slice_type == 'alloy':
            vmin, vmax = 0, 2
            im = plt.imshow(slice_img, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
        elif slice_type == 'boundary':
             vmin, vmax = 0, 1
             im = plt.imshow(slice_img, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
        else: # poly
             im = plt.imshow(slice_img, cmap=cmap, interpolation='none')

        plt.colorbar(im, label=f'{slice_type} 值')
        plt.title(f'{slice_type.capitalize()} Slice (Z = {z_idx})')
        plt.xlabel('X Index')
        plt.ylabel('Y Index')
        plt.tight_layout()

        slice_path = save_dir / f"{slice_type}_slice_{slice_num}_z{z_idx}.png"
        try:
            plt.savefig(slice_path, dpi=150) # 使用适中的DPI来平衡速度和大小
        except Exception as e:
            print(f"\n错误: 保存切片图像 {slice_path} 失败: {e}")
        plt.close() # 关闭图形以释放内存


# 修改为一次性分配seeds_gpu
def generate_polycrystal_model_optimized(num_grains, size, grid_size, chunk_size=200,
                                         precipitate_ratio=0.8, visualize=True, num_workers=None):
    """优化版多晶体模型生成函数"""
    print("\n--- 开始多晶体模型生成 (优化版) ---")
    overall_start_time = time.time()

    # 创建优化版处理器
    processor = OptimizedChunkProcessor(grid_size, chunk_size, num_workers)

    # 1. 生成随机晶粒种子点 (Host)
    print("\n[步骤 1/4] 生成随机种子点...")
    step_start_time = time.time()
    seeds_host = np.array([[random.uniform(0, size), random.uniform(0, size), random.uniform(0, size)]
                      for _ in range(num_grains)], dtype=np.float32) # 确保float32
    num_seeds_int = np.int32(num_grains) # 存储为int32用于核函数调用
    print(f"  生成 {num_grains} 个种子点完成 ({time.time() - step_start_time:.2f}s)")
    print_memory_usage()

    # --- 分配种子点到GPU ---
    print("\n[步骤 2/4] 将种子点传输到GPU...")
    step_start_time = time.time()
    try:
        seeds_gpu = gpuarray.to_gpu(seeds_host)
        print(f"  种子点传输到GPU完成 ({time.time() - step_start_time:.2f}s)")
    except cuda.MemoryError as e:
        print(f"\n错误: 分配种子点GPU内存失败 ({seeds_host.nbytes / 1024**2:.2f} MB)。GPU内存不足。")
        print_memory_usage()
        return None # 无法继续
    except Exception as e:
        print(f"\n错误: 传输种子点到GPU时出错: {e}")
        return None
    print_memory_usage()

    # 2. 找到最近的种子点（晶粒分配）- 执行GPU计算并保存结果
    print("\n[步骤 3/4] 分配网格点到最近的晶粒 (GPU)...")
    step_start_time = time.time()
    processor.process_all_chunks_parallel(seeds_gpu, num_seeds_int, size, 'find_closest_grain')
    print(f"  晶粒分配完成 ({time.time() - step_start_time:.2f}s)")
    print_memory_usage()

    # 3. 一次性处理晶界标记和析出相 - 执行GPU计算并保存结果
    print("\n[步骤 4/4] 处理晶界和析出相 (GPU)...")
    step_start_time = time.time()
    processor.process_all_chunks_parallel(seeds_gpu, num_seeds_int, size, 'process_all', precipitate_ratio)
    print(f"  晶界和析出相处理完成 ({time.time() - step_start_time:.2f}s)")
    print_memory_usage()

    # --- 释放种子点GPU内存 ---
    print("\n释放种子点GPU内存...")
    del seeds_gpu # 移除引用
    gc.collect()  # 鼓励垃圾回收
    cuda.Context.synchronize() # 确保GPU在检查内存前处于空闲状态
    print("释放种子点GPU内存后:")
    print_memory_usage()
    # --- END 释放种子点GPU内存 ---

    # 4. 如果需要可视化，生成样本切片 (CPU密集型IO/绘图)
    if visualize:
        print("\n[可视化] 生成样本切片...")
        step_start_time = time.time()
        # 顺序运行切片组装，避免绘图线程过多导致内存/CPU不足
        processor.assemble_sample_slices('poly', num_slices=3)
        processor.assemble_sample_slices('boundary', num_slices=3)
        processor.assemble_sample_slices('alloy', num_slices=3)
        print(f"  可视化切片生成完成 ({time.time() - step_start_time:.2f}s)")
        print_memory_usage()

    total_time = time.time() - overall_start_time
    print(f"\n--- 多晶体模型生成完成 ---")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

    return processor # 返回处理器，以防用户想要进一步交互

def main():
    # 记录程序开始的时间
    start_time = time.time()

    # --- 参数定义 ---
    num_grains = 64000       # 晶粒数目 (例如, 64000, 128000)
    size = 1.0               # 模型物理尺寸 (任意单位)
    grid_size = 4000         # 网格分辨率 (例如, 500, 1000, 2000) - 大幅影响内存和计算时间
    chunk_size = 256         # 块大小 (例如, 128, 256, 400) - 影响块数量和单个块的内存占用
    precipitate_ratio = 0.8  # 边界上形成析出相的比例
    visualize_slices = True  # 是否生成并保存可视化切片
    num_workers = None       # (不再主要用于并行计算提交)
    # --- END 参数定义 ---

    print(f"\n当前参数设置:")
    print(f"网格大小: {grid_size}x{grid_size}x{grid_size} = {grid_size**3 / 1e9:.2f} billion 点")
    print(f"  (预计 Poly/Boundary/Alloy 数据大小 (int32): {grid_size**3 * 4 / 1024**3:.2f} GB each)")
    print(f"晶粒数量: {num_grains}")
    print(f"模型尺寸: {size}")
    print(f"块大小: {chunk_size}x{chunk_size}x{chunk_size}")
    print(f"析出相比例: {precipitate_ratio}")
    print(f"生成可视化: {visualize_slices}")

    print("\n初始内存状态:")
    print_memory_usage()

    # 处理大型多晶体模型
    processor_instance = generate_polycrystal_model_optimized(
        num_grains,
        size,
        grid_size,
        chunk_size,
        precipitate_ratio,
        visualize_slices,
        num_workers
    )

    # 计算总运行时间
    total_time = time.time() - start_time
    print(f"\n脚本总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

    print("\n最终内存状态:")
    print_memory_usage()
    print("\n脚本执行完毕.")

if __name__ == '__main__':
    main()
