import time
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 以下用一个简单的网络LeNet来展示多gpu训练

# 手动检测并管理 GPU 设备 (try_gpus)
# 数据集加载与预处理 (load_data_fashion_mnist)
# 简易定时器 (Timer)
# 动态训练曲线绘制 (Animator)
# 参数同步与更新（allreduce、train_batch 中的梯度聚合 + SGD）
# 模型评估 (evaluate_accuracy_gpu)

# 判断并获取GPU设备，默认返回cpu或多个GPU列表

def try_gpus(n):
    """返回0~n-1的GPU设备列表，如果没有GPU则返回['cpu']。"""
    gpus = []
    count = torch.cuda.device_count()
    for i in range(min(n, count)):
        gpus.append(torch.device(f'cuda:{i}'))
    if not gpus:
        gpus = [torch.device('cpu')]
    return gpus

# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
# 这里直接用函数实现LeNet前向，不依赖nn.Module

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss_fn = nn.CrossEntropyLoss(reduction='none')

# 数据同步：向多个设备分发参数并附带梯度

def get_params(params, device):
    """将参数复制到指定device上，并附加梯度信息"""
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

# AllReduce梯度聚合

def allreduce(data):
    """将多个设备上的tensor梯度聚合并同步。"""
    for i in range(1, len(data)):
        data[0] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)

# 将X和y拆分到多个设备上

def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    X_shards = nn.parallel.scatter(X, devices)
    y_shards = nn.parallel.scatter(y, devices)
    return X_shards, y_shards

# 小批量训练步骤

def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算loss
    losses = []
    for Xs, ys, params_i in zip(X_shards, y_shards, device_params):
        y_hat = lenet(Xs, params_i)
        l = loss_fn(y_hat, ys).sum()
        losses.append(l)
    # 反向传播在每个GPU上分别执行
    for l in losses:
        l.backward()
    # 将每个GPU的所有梯度相加，并广播
    with torch.no_grad():
        for i in range(len(device_params[0])):
            grads = [device_params[c][i].grad for c in range(len(devices))]
            allreduce(grads)
    # 更新模型参数
    with torch.no_grad():
        batch_size = X.shape[0]
        for params_i in device_params:
            for p in params_i:
                p -= lr * p.grad / batch_size
                p.grad.zero_()

# 加载Fashion-MNIST数据集

def load_data_fashion_mnist(batch_size):
    """下载Fashion-MNIST数据集并创建DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter

# 简易定时器
class Timer:
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start_time = None
    def start(self):
        self.start_time = time.time()
    def stop(self):
        if self.start_time:
            self.times.append(time.time() - self.start_time)
    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0

# 模型评估函数

def evaluate_accuracy_gpu(net, data_iter, device):
    """使用net在data_iter上评估准确率"""
    net_params = net.__defaults__[1]  # 获取参数列表的第二个默认参数列表
    net_device = device
    net_eval = lambda X: lenet(X, net_params)
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net_eval(X)
            preds = y_hat.argmax(dim=1)
            acc_sum += (preds == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 简易绘图工具
class Animator:
    """在训练过程中绘制测试精度曲线"""
    def __init__(self, xlabel, ylabel, xlim, ylim=None):
        self.xlabel, self.ylabel, self.xlim, self.ylim = xlabel, ylabel, xlim, ylim
        self.X, self.Y = [], []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(*xlim)
        if ylim: self.ax.set_ylim(*ylim)
    def add(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        self.ax.clear()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.plot(self.X, self.Y, marker='o')
        plt.pause(0.01)
    def close(self):
        plt.ioff()
        plt.show()

# 定义多gpu训练的训练函数

def train(num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = try_gpus(num_gpus)
    # 将模型参数复制到num_gpus个设备
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            for d in devices:
                if d.type != 'cpu': torch.cuda.synchronize(d)
        timer.stop()
        # 在设备0上评估模型
        test_acc = evaluate_accuracy_gpu(lenet, test_iter, devices[0])
        animator.add(epoch + 1, test_acc)
    animator.close()
    print(f'测试精度：{test_acc:.2f}，{timer.avg():.1f}秒/轮，在 {devices} 上训练')

# 启动训练，指定GPU数量、批量大小和学习率
train(num_gpus=1, batch_size=256, lr=0.2)
