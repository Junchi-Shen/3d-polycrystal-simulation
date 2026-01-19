# main.py
import numpy as np
# import cupy as cp # main.py 本身可能不需要直接导入 cupy
import tqdm
import argparse
import os
import sys # 用于添加路径

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 或者 '0' 也可以

# --- 然后再 import cupy, numba, 和你其他的模块 ---
import numpy as np
# import cupy as cp # simulator.py 等文件中会导入
import tqdm
import argparse
import sys
# ... (你其他的 import) ...

# --- 将项目根目录添加到 Python 路径 ---
# 这使得我们可以使用 from simulation import ... 等
# 假设 main.py 在项目的根目录下，或者需要根据实际位置调整
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# ---------------------------------------

from parameters import SimulationParameters, CorrosionRule, VisualizationParameters
from simulation.simulator import CorrosionSimulator
from visualization.plotter import visualize_corr_grid_cpu

def run_simulation(sim_params: SimulationParameters,
                   rule: CorrosionRule,
                   vis_params: VisualizationParameters):
    """运行主模拟循环"""

    # 模拟器初始化现在包含数据加载和 GPU 设置
    try:
        simulator = CorrosionSimulator(sim_params, rule)
    except ValueError as e:
        print(f"初始化模拟器失败: {e}")
        return # 初始化失败则退出

    print(f"--- 开始模拟 {sim_params.steps} 步 ---")
    # 使用 simulator.shape 获取实际形状
    n_total, m, l = simulator.shape
    print(f"模拟网格实际尺寸: ({n_total}, {m}, {l})")

    for step in tqdm.tqdm(range(sim_params.steps)):
        try:
            simulator.step()
        except Exception as e:
            print(f"\n在模拟步骤 {step} 出错: {e}")
            # 可以选择在这里中断或继续
            break

        # 定期可视化
        if vis_params.interval > 0 and (step % vis_params.interval == 0 or step == sim_params.steps - 1):
            try:
                grid_data_cpu = simulator.get_grid_for_visualization()
                # 确保 grid_data_cpu 不是 None 或空的
                if grid_data_cpu is not None and grid_data_cpu.size > 0:
                     visualize_corr_grid_cpu(grid_data_cpu, step, vis_params)
                else:
                     print(f"警告: 步骤 {step} 获取的可视化数据为空。")
            except Exception as e:
                 print(f"\n在可视化步骤 {step} 出错: {e}")
                 # 可视化错误通常不应中断模拟，继续执行

    print("--- 模拟完成 ---")

if __name__ == '__main__':
    # --- 使用 argparse 获取参数 ---
    parser = argparse.ArgumentParser(description="CUDA 加速的 3D 腐蚀模拟 (重构版)")
    parser.add_argument("--grid_file", type=str, required=True, help="包含初始金属网格的 .npy 文件路径")
    parser.add_argument("--steps", type=int, default=10000, help="模拟的总步数 (默认: 10000)")
    parser.add_argument("--thickness", type=int, default=10, help="溶液层厚度 (默认: 10)")
    parser.add_argument("--conc", type=float, default=0.2, help="S元胞初始浓度 (默认: 0.2)")
    parser.add_argument("--p_pmove", type=float, default=0.2, help="腐蚀产物 P 的移动概率 (默认: 0.2)")
    parser.add_argument("--vis_interval", type=int, default=500, help="可视化间隔步数 (0 表示不保存/显示, 默认: 500)")
    parser.add_argument("--vis_slice", type=int, default=None, help="可视化的 Z 切片索引 (默认: 中间)")
    parser.add_argument("--vis_dir", type=str, default="corrosion_plots_refactored", help="可视化图片保存目录 (默认: corrosion_plots_refactored)")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子 (默认: 42)")
    parser.add_argument("--no_save", action='store_true', help="如果设置，则不保存可视化图片")
    # 可以添加更多参数，例如修改腐蚀概率

    args = parser.parse_args()

    # --- 创建参数对象 ---
    # SimulationParameters 会在初始化时加载 grid_file
    sim_parameters = SimulationParameters(
        grid_file=args.grid_file,
        steps=args.steps,
        concentration_ratio=args.conc,
        solution_thickness=args.thickness,
        rng_seed=args.seed
    )

    corrosion_rule = CorrosionRule(
        p_Pmove=args.p_pmove
        # TODO: 可以添加从命令行修改 corrosion_probabilities 的功能
    )

    vis_parameters = VisualizationParameters(
        interval=args.vis_interval,
        z_slice=args.vis_slice,
        show_plots=False, # 命令行通常不方便实时显示
        save_plots=not args.no_save, # 根据命令行参数设置是否保存
        save_dir=args.vis_dir
    )

    # --- 运行模拟 ---
    run_simulation(sim_parameters, corrosion_rule, vis_parameters)

    print("模拟程序执行完毕。")