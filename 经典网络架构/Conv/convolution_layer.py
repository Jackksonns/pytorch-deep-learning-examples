import torch
from torch import nn

# 自定义二维互相关运算函数（不依赖 d2l）
def corr2d(X, K):
    """
    计算二维互相关运算（不含翻转的卷积）
    参数：
        X: 输入张量（二维）
        K: 卷积核（二维）
    返回：
        Y: 互相关输出张量
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 局部区域与核对应元素相乘后求和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# 测试 corr2d 函数的基本功能
X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0],
                  [2.0, 3.0]])
print("corr2d(X, K):\n", corr2d(X, K))

# 自定义的二维卷积层（用 corr2d 实现）
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # 卷积核权重为可训练参数
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 构造输入数据：6行8列，中间四列为0，其余为1
X = torch.ones((6, 8))
X[:, 2:6] = 0
print("X:\n", X)

# 目标卷积核：用于检测左右边缘（1, -1）
K = torch.tensor([[1.0, -1.0]])

# 目标输出 Y：使用 corr2d 得到
Y = corr2d(X, K)
print("Y:\n", Y)

# 查看转置情况下的卷积核检测垂直边缘效果
print("X.t():\n", X.t())
print("corr2d(X.t(), K):\n", corr2d(X.t(), K))

# 使用 nn.Conv2d 训练一个能学出上述卷积核的模型
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)

# PyTorch 的 Conv2d 要求输入为4D： [batch_size, channels, height, width]
X_train = X.reshape((1, 1, 6, 8))  # 输入
Y_train = Y.reshape((1, 1, 6, 7))  # 目标输出

# 设置学习率
lr = 0.03

# 训练模型使其学会目标卷积核
for i in range(10):
    Y_hat = conv2d(X_train)  # 前向传播
    loss = (Y_hat - Y_train) ** 2  # 平方误差
    conv2d.zero_grad()  # 梯度清零
    loss.sum().backward()  # 反向传播
    # 手动更新参数（SGD）
    with torch.no_grad():
        conv2d.weight -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'Epoch {i+1}, Loss {loss.sum():.3f}')

# 查看学到的卷积核权重
print("Learned kernel:", conv2d.weight.data.reshape((1, 2)))
