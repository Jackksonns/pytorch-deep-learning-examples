import torch
from torch import nn
#要求：理解转置卷积的原理，会计算，同时会调用高级api即可

# 自定义转置卷积操作（不使用nn.ConvTranspose2d）
# 输入 X：输入特征图
# 输入 K：卷积核（注意不是转置核）
# 返回 Y：进行转置卷积后的输出

def trans_conv(X, K):
    h, w = K.shape
    # 输出尺寸会比输入更大（转置卷积的特性）
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    # 遍历输入的每一个元素，将其乘以 kernel 并“加到”对应位置上（反卷积操作）
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

# 测试 trans_conv 函数
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))


# 以下使用高级API：nn.ConvTranspose2d 实现转置卷积（可学习参数）


# 先将 X 和 K reshape 为 4D 张量，适配 nn.ConvTranspose2d 输入格式
# 输入尺寸：[batch_size, in_channels, height, width]
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)

# 不加 padding 或 stride 的默认转置卷积
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K  # 手动设置权重为我们定义的 K
print(tconv(X))

# 加上 padding=1，输出尺寸会更大
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))

# 设置 stride=2，可以放大图像尺寸（上采样）
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))


# 示例：一个常见的先下采样（Conv2d），再上采样（ConvTranspose2d）恢复原图

X = torch.rand(size=(1, 10, 16, 16))  # 输入通道为10，图像为16x16
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)  # 输出变为20通道
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
# 判断是否能恢复原始尺寸
print(tconv(conv(X)).shape == X.shape)  # 输出：True 表示恢复成功


# 自定义核函数与矩阵展开的等价实现

X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 等价于卷积运算（无 padding、无 stride）
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

Y = corr2d(X, K)
print(Y)


# 将卷积核转换为稀疏矩阵形式，以便用矩阵乘法模拟卷积

def kernel2matrix(K):
    # 目标是构造一个稀疏矩阵 W，使得 W @ X.reshape(-1) = 卷积结果
    k, W = torch.zeros(5), torch.zeros((4, 9))
    # 手动展开 K 的值放入 k 中
    k[:2], k[3:5] = K[0, :], K[1, :]
    # 构造 W 的每一行，分别对应卷积窗口滑动的 4 个位置
    W[0, :5] = k           # 卷积左上角位置
    W[1, 1:6] = k          # 卷积向右滑动一个位置
    W[2, 3:8] = k          # 卷积向下滑动一行，从第3个元素开始
    W[3, 4:] = k           # 最后一行
    return W

W = kernel2matrix(K)
print(W)

# 将图像矩阵展开成向量，乘以 W 模拟卷积（结果等价于 Y）
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))


# 验证反卷积是否等价于 W.T 与 Y 的乘积

Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))
