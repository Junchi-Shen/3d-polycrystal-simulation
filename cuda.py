import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# 定义 CUDA C++ 风格的内核代码
kernel_code = """
__global__ void add_one(float *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] + 1.0f;
  }
}
"""

# 创建一个 SourceModule 对象来编译 CUDA 代码
mod = SourceModule(kernel_code)

# 获取内核函数
add_one_gpu = mod.get_function("add_one")

# 定义主机上的输入数据
h_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
n = h_in.size

# 在 GPU 上分配内存
d_in = cuda.mem_alloc(h_in.nbytes)
d_out = cuda.mem_alloc(h_in.nbytes)

# 将主机数据复制到设备
cuda.memcpy_htod(d_in, h_in)

# 定义线程块和网格大小
block_size = 5
grid_size = 1

# 调用 GPU 内核函数
add_one_gpu(
    # 输入和输出指针必须作为 numpy.intp 类型传递
    cuda.In(h_in),
    cuda.Out(np.empty_like(h_in)),
    np.int32(n),
    block=(block_size, 1, 1),
    grid=(grid_size, 1, 1),
)

# 将设备上的结果复制回主机
h_out = np.empty_like(h_in)
cuda.memcpy_dtoh(h_out, d_out)

# 打印结果
print("Input array:", h_in)
print("Output array:", h_out)