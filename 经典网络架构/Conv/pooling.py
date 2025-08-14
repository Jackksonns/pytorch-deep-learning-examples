import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    """
    手动实现二维池化操作（max/avg pooling）
    
    参数:
        X: 输入的二维张量 (H, W)
        pool_size: 池化窗口大小 (p_h, p_w)
        mode: 'max' 表示最大池化，'avg' 表示平均池化

    返回:
        Y: 经过池化后的二维张量
    """
    p_h, p_w = pool_size
    # 输出尺寸为 (H - p_h + 1, W - p_w + 1)
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            patch = X[i: i + p_h, j: j + p_w]
            if mode == 'max':
                Y[i, j] = patch.max()
            elif mode == 'avg':
                Y[i, j] = patch.mean()
    return Y

# 示例：手动实现的池化
X = torch.tensor([
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0]
])
print("手动最大池化：\n", pool2d(X, (2, 2)))
print("手动平均池化：\n", pool2d(X, (2, 2), 'avg'))

# 使用 PyTorch 内建 nn.MaxPool2d
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print("原始 X：\n", X)

# 1. 普通 3x3 最大池化，无 padding，默认 stride
pool2d_layer = nn.MaxPool2d(3)
print("内建 3x3 最大池化：\n", pool2d_layer(X))

# 2. 带 padding=1，stride=2 的最大池化
pool2d_layer = nn.MaxPool2d(3, padding=1, stride=2)
print("padding=1, stride=2 最大池化：\n", pool2d_layer(X))

# 3. 非正方形池化窗口 (2,3)，stride=(2,3)，padding=(0,1)
pool2d_layer = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print("非正方形窗口池化：\n", pool2d_layer(X))

# 多通道情况：复制一个通道并做 +1
X = torch.cat((X, X + 1), dim=1)  # 变成2通道
print("多通道输入 X：\n", X)

# 使用 nn.MaxPool2d 自动对多个通道分别池化
pool2d_layer = nn.MaxPool2d(2, stride=2)
print("多通道最大池化：\n", pool2d_layer(X))
