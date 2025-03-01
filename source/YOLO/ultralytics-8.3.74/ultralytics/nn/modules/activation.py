# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Activation modules."""  # 激活模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块

class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""
    # 来自https://github.com/kostas1515/AGLU的统一激活函数模块

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        # 初始化统一激活函数
        super().__init__()  # 调用父类的初始化方法
        self.act = nn.Softplus(beta=-1.0)  # 定义Softplus激活函数，beta设置为-1.0
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda参数
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        # 计算统一激活函数的前向传播
        lam = torch.clamp(self.lambd, min=0.0001)  # 将lambda参数限制在最小值0.0001
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))  # 计算激活值并返回