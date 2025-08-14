import time
import os
import torch
import torchvision
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image

# 设置图像显示大小
plt.rcParams['figure.figsize'] = (6, 6)

# 打开并显示图像（假设有这个图片
img = Image.open('../img/cat1.jpg')
plt.imshow(img)
plt.axis('off')

# 自定义展示多个图像的函数

def show_images(imgs, num_rows, num_cols, scale=1.5):
    """在网格中展示图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.axis('off')
    plt.show()


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)

# 翻转和裁剪

# 左右随机翻转
apply(img, transforms.RandomHorizontalFlip())

# 上下随机翻转
apply(img, transforms.RandomVerticalFlip())

# 随机裁剪
shape_aug = transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 改变颜色
# 随机改变颜色的亮度
apply(img, transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))

# 随机更改图像的色调
apply(img, transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))

# 我们还可以创建一个RandomColorJitter实例，并设置如何同时
# [随机更改图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）]
color_aug = transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

augs = transforms.Compose([
    transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

# 使用图像增强进行训练
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

# 便于读取图像和应用图像增强的辅助函数：（注意这里已经默认在函数内读取了指定的数据

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    workers = min(4, os.cpu_count() if os.cpu_count() else 1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=workers)
    return dataloader

# 用于记录时间的计时器
class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.t0 = time.time()

    def stop(self):
        self.times.append(time.time() - self.t0)
        return self.times[-1]

    def sum(self):
        return sum(self.times)

# 简易累加器
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]

# 计算准确率

def accuracy(y_hat, y):
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.sum())

# 评估模型在GPU上的准确率
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

# 简易绘图器
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.X, self.Y = [], []

    def add(self, x, ys):
        self.X.append(x)
        if not self.Y:
            self.Y = [[y] for y in ys]
        else:
            for y_seq, y in zip(self.Y, ys):
                y_seq.append(y)
        self._draw()

    def _draw(self):
        plt.clf()
        for y_seq in self.Y:
            plt.plot(self.X, y_seq)
        if self.xlabel: plt.xlabel(self.xlabel)
        if self.ylabel: plt.ylabel(self.ylabel)
        if self.xlim: plt.xlim(self.xlim)
        if self.ylim: plt.ylim(self.ylim)
        if self.legend: plt.legend(self.legend)
        plt.pause(0.001)

# 用多GPU进行小批量训练
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

# 用多GPU进行模型训练，本身并不直接执行数据增强，它只是一个训练过程的调度函数
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=None):
    """用多GPU进行模型训练"""
    devices = devices or [torch.device('cuda' if torch.cuda.is_available() else 'cpu')] * torch.cuda.device_count()
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    if len(devices) > 1:
        net = nn.DataParallel(net, device_ids=[d.index for d in devices]).to(devices[0])
    else:
        net = net.to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = evaluate_accuracy_gpu(net, test_iter, devices[0])
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{devices}')

batch_size = 256
# 查找可用设备
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] or [torch.device('cpu')]
# 使用torchvision提供的ResNet18
net = models.resnet18(pretrained=False)
net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.fc = nn.Linear(net.fc.in_features, 10)

# 初始化权重

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

# 使用图像增强来训练的训练函数
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    #这里就运行了上面的原始训练函数
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

train_augs = transforms.Compose([
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])

test_augs = transforms.Compose([
     transforms.ToTensor()])

train_with_data_aug(train_augs, test_augs, net)
