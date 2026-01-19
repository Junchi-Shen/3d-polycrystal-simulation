# constants.py
import numpy as np # Numba 内核中可能需要 np 类型
# 元胞类型常量
GRAIN_AL = 0          # 晶粒铝
GRAIN_BOUNDARY_AL = 1 # 晶界铝
PRECIPITATE = 2       # 析出相
METAL_TYPE_3 = 3      # 类型3金属 (假设)
NEUTRAL_SOLUTION = 4  # 中性溶液
CORROSIVE_SOLUTION = 5 # 腐蚀性溶液 (S 元胞)
CORROSION_PRODUCT = 6 # 腐蚀产物 (P 元胞)