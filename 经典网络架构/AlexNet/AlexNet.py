import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

# 定义 AlexNet 网络结构
net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10)
)

# 打印每一层的输出形状
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 数据预处理与加载（把28×28强行resize到224×224以适配AlexNet）
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size)

# 训练函数定义
def evaluate_accuracy(net, data_iter, device=None):
    if device is None:
        device = next(net.parameters()).device
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(net, train_iter, test_iter, num_epochs, lr, device):
    print(f'training on {device}')
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.size(0)
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.size(0)
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(f'epoch {epoch + 1}, loss {train_loss_sum / n:.4f}, '
              f'train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}, '
              f'time {time.time() - start:.1f} sec')

# 训练 AlexNet
lr, num_epochs = 0.01, 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(net, train_iter, test_iter, num_epochs, lr, device)
