import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        '''
        背景：LayerNorm的问题（计算均值方差开销大）

        RMSNorm定义：只使用均方根（RMS）进行归一化，不重新中心化（无均值减法）

        公式：给定输入x，RMS(x) = sqrt(mean(x_i^2) + eps)，然后 normalized = x / RMS(x)，再乘以可学习的增益参数g

        与LayerNorm对比：LayerNorm: (x-μ)/σ * g + b；RMSNorm: x / RMS(x) * g，无偏置b

        优点：计算量小（省去了计算均值和方差中的均值部分），效果与LayerNorm相当或更好（据原论文）
        '''
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        # x / x
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True)+self.eps)
    
    def forward(self, x: torch.Tensor):
        return (self.weight * self.norm(x.float())).type_as(x)
