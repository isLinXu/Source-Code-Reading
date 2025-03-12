import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习

np.random.seed(2)  # 设置 NumPy 随机种子为 2，确保结果可重复

T = 20  # 设置周期 T 为 20
L = 1000  # 设置每个序列的长度 L 为 1000
N = 100  # 设置生成的序列数量 N 为 100

x = np.empty((N, L), 'int64')  # 创建一个空的 NumPy 数组 x，形状为 (N, L)，数据类型为 int64
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)  # 为每个序列生成 x 值，添加随机噪声

data = np.sin(x / 1.0 / T).astype('float64')  # 计算每个 x 值对应的正弦值，并转换为 float64 类型
torch.save(data, open('traindata.pt', 'wb'))  # 将生成的数据保存为 'traindata.pt' 文件