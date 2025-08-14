import torch
import torch.nn.functional as F
from torch import nn

#自定义不带参数的层
#其实所谓的“层”可以理解为设计好的一种矩阵计算方法。（注意不管怎样新定义的层的父类都是nn.Module）
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean() #就是一种计算方法而已
    
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean()

#带参数的层
#用nn.Parameter()实现
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,)) #创建一个一维张量
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data #注意理解这个公式，实际上还是要符合矩阵运算的规则
        return F.relu(linear)
#在 Linear 层中，偏置（bias）的维度是 units 而不是 in_units，这是由线性变换的数学定义决定的：

# 对于线性变换 y = X·W + b：
# 输入 X 的形状是 (batch_size, in_units)
# 权重 W 的形状是 (in_units, units)（实现输入维度到输出维度的映射）
# 矩阵乘法 X·W 的结果形状是 (batch_size, units)
# 偏置 b 需要与这个结果进行相加（通过广播机制），因此必须是 (units,) 形状
    
linear = MyLinear(5, 3)
linear.weight

linear(torch.rand(2, 5))

#使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))

