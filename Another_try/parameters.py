# parameters.py
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
# from constants import * # 导入常量

from constants import GRAIN_AL, GRAIN_BOUNDARY_AL, PRECIPITATE, METAL_TYPE_3

@dataclass
class SimulationParameters:
    """持有模拟的基本设置"""
    grid_file: str              # 初始金属网格 .npy 文件路径
    steps: int = 10000          # 总模拟步数
    concentration_ratio: float = 0.2 # S 元胞初始浓度
    solution_thickness: int = 10   # 溶液层厚度
    rng_seed: int = 42           # 随机数种子

    # 这些字段将在 __post_init__ 中根据 grid_file 初始化
    bound_grid_cpu: np.ndarray = field(init=False, repr=False) # 不在 repr 中显示大数组
    grid_shape_metal: Tuple[int, int, int] = field(init=False)

    def __post_init__(self):
        """在初始化后加载数据并设置形状"""
        if not os.path.exists(self.grid_file):
            print(f"错误: 找不到初始金属网格文件: {self.grid_file}")
            print("将生成一个简单的 50x50x50 测试网格...")
            n_test, m_test, l_test = 50, 50, 50
            # 使用导入的常量
            self.bound_grid_cpu = np.random.choice(
                [GRAIN_AL, GRAIN_BOUNDARY_AL, PRECIPITATE],
                size=(n_test, m_test, l_test),
                p=[0.8, 0.15, 0.05]
            ).astype(np.int32)
            # 更改 grid_file 属性以反映实际使用的文件
            self.grid_file = "test_bound_grid.npy"
            np.save(self.grid_file, self.bound_grid_cpu)
            print(f"已生成并使用测试文件: {self.grid_file}")
        else:
            print(f"正在加载初始金属网格: {self.grid_file}")
            self.bound_grid_cpu = np.load(self.grid_file).astype(np.int32)

        self.grid_shape_metal = self.bound_grid_cpu.shape
        print(f"加载的金属网格尺寸: {self.grid_shape_metal}")
        
@dataclass
class CorrosionRule:
    """持有腐蚀过程的规则参数"""
    # 基础腐蚀概率
    corrosion_probabilities: Dict[int, float] = field(default_factory=lambda: {
        GRAIN_AL: 0.0002,
        GRAIN_BOUNDARY_AL: 0.6,
        PRECIPITATE: 0.05,
        METAL_TYPE_3: 0.8
    })
    # 腐蚀产物 P 的移动概率
    p_Pmove: float = 0.2

@dataclass
class VisualizationParameters:
    """持有可视化相关的参数"""
    interval: int = 500          # 每隔多少步进行一次可视化
    z_slice: Optional[int] = None # 可视化时显示的 Z 切片索引 (None 为中间)
    show_plots: bool = False     # 是否实时显示可视化结果
    save_plots: bool = True      # 是否保存可视化结果
    save_dir: str = "corrosion_plots_refactored" # 保存图像的目录