import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F

# 以下用 ResNet-18 展示多 GPU 训练(重点看下面的训练函数即可)

# 判断并获取 GPU 设备列表

def try_gpus(n):
    """返回 0~n-1 的 GPU 设备列表，如果没有 GPU 则返回 ['cpu']。"""
    gpus = []
    count = torch.cuda.device_count()
    for i in range(min(n, count)):
        gpus.append(torch.device(f'cuda:{i}'))
    if not gpus:
        gpus = [torch.device('cpu')]
    return gpus

# 定义 Residual 模块（与 d2l.Residual 相同）
class Residual(nn.Module):  #@save
    """残差块，包括可选的 1x1 卷积实现通道和尺寸匹配"""
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

# 定义 ResNet-18 模型

def resnet18(num_classes, in_channels=1):
    """稍加修改的 ResNet-18,输入通道可定制"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    net = nn.Sequential(
        # 初始卷积层：3x3, stride=1, padding=1
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    # 四个阶段，每个阶段包含若干残差块
    net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
    net.add_module('resnet_block2', resnet_block(64, 128, 2))
    net.add_module('resnet_block3', resnet_block(128, 256, 2))
    net.add_module('resnet_block4', resnet_block(256, 512, 2))
    # 全局平均池化 + 全连接层
    net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1,1)))
    net.add_module('fc', nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, num_classes)
    ))
    return net

# 加载 Fashion-MNIST 数据集

def load_data_fashion_mnist(batch_size):
    """下载 Fashion-MNIST 并创建 DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    mnist_train = datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
    mnist_test = datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
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

# 评估准确率（支持 DataParallel）
def evaluate_accuracy_gpu(net, data_iter, device):
    """在 data_iter 上评估模型 net 的准确率"""
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            preds = net(X).argmax(dim=1)
            acc_sum += (preds == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 训练函数，使用 DataParallel

def train(net, num_gpus, batch_size, lr, num_epochs=10):
    """使用 multiple GPUs via nn.DataParallel 训练模型 net"""
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    #这里是定义devices
    devices = try_gpus(num_gpus)
    # 初始化权重
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    # 多 GPU 并行
    # 下面这个api很关键，给定一个实例化的网络和设备，返回一个多GPU的实例化网络
    net = nn.DataParallel(net, device_ids=devices)
    net.to(devices[0])

    # 优化器与损失函数
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    timer = Timer()
    # 绘图
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch')
    ax.set_ylabel('test acc')
    ax.set_xlim(1, num_epochs)
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            X, y = X.to(devices[0]), y.to(devices[0])
            trainer.zero_grad()
            l = loss_fn(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        # 同步 CUDA
        for d in devices:
            if d.type != 'cpu': torch.cuda.synchronize(d)
        # 测试
        net.eval()
        test_acc = evaluate_accuracy_gpu(net, test_iter, devices[0])
        # 绘制曲线
        ax.scatter(epoch+1, test_acc)
        plt.pause(0.01)
    plt.ioff()
    plt.show()
    print(f'测试精度：{test_acc:.2f},{timer.avg():.1f} 秒/轮，在 {devices} 上训练')

# 启动训练
net = resnet18(num_classes=10)
train(net, num_gpus=2, batch_size=256, lr=0.1, num_epochs=10)
