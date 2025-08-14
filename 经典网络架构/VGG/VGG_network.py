import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
import time

# 定义 VGG 块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# 定义卷积结构
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 构建 VGG 网络
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

# 构建简化版 VGG 网络（通道数除以4）
ratio = 4
small_conv_arch = [(c[0], c[1] // ratio) for c in conv_arch]
net = vgg(small_conv_arch)

# 打印每层输出形状
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

# 数据加载与预处理
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(trans)

    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2))

# 精度评估
def evaluate_accuracy(net, data_iter, device=None):
    if device is None:
        device = next(net.parameters()).device
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            pred = net(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

# 训练函数
def train(net, train_iter, test_iter, num_epochs, lr, device):
    print(f'Training on {device}')
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        start = time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

        train_acc = total_correct / total_samples
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(f'Epoch {epoch + 1}: '
              f'train loss {total_loss / total_samples:.4f}, '
              f'train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}, '
              f'time {time.time() - start:.1f} sec')

# 设置参数并训练
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(net, train_iter, test_iter, num_epochs, lr, device)
