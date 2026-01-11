# 3D多晶体微结构生成与模拟

一个使用GPU加速的3D多晶体（polycrystal）微结构生成和模拟程序，支持大规模网格计算和可视化。

## 功能特点

- 🚀 **GPU加速计算**：使用PyCUDA进行并行计算，大幅提升处理速度
- 📦 **分块处理**：将大网格分成小块处理，避免GPU内存溢出
- 🔬 **多晶体建模**：基于Voronoi图方法生成多晶体结构
- 🎯 **晶界识别**：自动识别和标记晶粒边界
- 💎 **析出相生成**：在晶界上随机生成析出相
- 📊 **可视化输出**：生成多晶体、晶界和合金结构的切片图像

## 系统要求

- **Python**: 3.8+
- **GPU**: 支持CUDA的NVIDIA显卡
- **内存**: 建议至少32GB RAM（用于大规模计算）
- **CUDA**: 需要安装CUDA Toolkit（11.7+ 或 12.0+）

## 安装依赖

```bash
pip install numpy matplotlib tqdm psutil pycuda
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

直接运行主程序：

```bash
python 3DCA_2.py
```

### 参数配置

在`main()`函数中可以修改以下参数：

```python
num_grains = 64000       # 晶粒数目
size = 1.0               # 模型物理尺寸
grid_size = 4000         # 网格分辨率 (4000x4000x4000)
chunk_size = 256         # 块大小
precipitate_ratio = 0.8  # 边界上形成析出相的比例
visualize_slices = True  # 是否生成可视化切片
```

### 输出文件

程序会在以下目录生成结果：

- `./chunks/` - 分块数据文件（.npy格式）
- `./results/` - 可视化切片图像（.png格式）

## 项目结构

```
.
├── 3DCA_2.py                          # 主程序文件
├── corrosion_frames_strict_rules_v2/  # 腐蚀模拟相关（待开发）
├── corrosion_output_gpu/             # GPU腐蚀输出
├── corrosion_plots/                   # 腐蚀可视化
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖列表
└── .gitignore                         # Git忽略文件
```

## 技术细节

### GPU加速

- 使用PyCUDA进行GPU并行计算
- 实现了多个CUDA内核函数：
  - `get_closest_seed_kernel_optimized`: 查找最近晶粒种子点
  - `mark_boundary_kernel_optimized`: 标记晶界
  - `mark_precipitate_kernel_optimized`: 标记析出相
  - `process_complete_chunk_kernel`: 组合处理内核

### 内存优化

- 分块处理策略，避免一次性加载整个网格
- 使用共享内存优化GPU计算
- 异步内存传输，提高效率
- 及时释放GPU内存

### 计算规模

支持大规模网格计算：
- 默认：4000³ = 640亿网格点
- 每个数据数组约256GB（int32格式）
- 通过分块处理，可以在有限GPU内存下完成计算

## 注意事项

⚠️ **内存要求**：大规模计算需要大量内存，建议至少32GB RAM

⚠️ **CUDA路径**：程序会自动查找CUDA安装路径，如果找不到请手动设置环境变量

⚠️ **运行时间**：大规模计算可能需要较长时间，请耐心等待

## 开发计划

- [ ] 腐蚀模拟功能完善
- [ ] 更多可视化选项
- [ ] 性能进一步优化
- [ ] 支持多GPU并行

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

