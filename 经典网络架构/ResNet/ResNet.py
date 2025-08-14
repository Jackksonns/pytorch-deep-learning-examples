import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def try_gpu():  #@save
    """如果存在GPU，则使用GPU，否则使用CPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """
    下载Fashion-MNIST数据集，然后将图像大小调整为resize，并将数据加载到DataLoader。
    返回训练集和测试集迭代器。
    """
    # 定义转换操作
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    
    # 下载数据集
    mnist_train = datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
    mnist_test = datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
    
    # 构造DataLoader
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):  #@save
    """
    定义训练函数，使用交叉熵损失和SGD优化器。
    每个epoch输出训练损失、训练准确率和测试准确率。
    """
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # 训练模型
        net.train()
        train_loss, train_acc, n_samples = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * y.shape[0]
            n_samples += y.shape[0]
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
        
        # 评估模型
        test_acc = evaluate_accuracy(net, test_iter, device)
        
        print(f"epoch {epoch + 1}, loss {train_loss / n_samples:.4f}, "
              f"train acc {train_acc / n_samples:.3f}, "
              f"test acc {test_acc:.3f}")


def evaluate_accuracy(net, data_iter, device):  #@save
    """
    在给定数据集上评估模型net的准确率。
    """
    net.eval()
    acc_sum, n_samples = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n_samples += y.shape[0]
    return acc_sum / n_samples


class Residual(nn.Module):  #@save
    """
    从零实现残差块。
    当use_1x1conv为True时，通过1x1卷积将输入通道变换到num_channels。
    strides用于控制卷积步幅以调整特征图高宽。
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))  # 两层卷积 + BN + ReLU
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)  # 通过1x1卷积变换输入
        Y += X  # 残差连接
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """
    构造由num_residuals个Residual组成的序列。
    除第一个stage外，第一个残差块使用1x1卷积改变通道数并减半高宽。
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)


# 构建ResNet-18
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 后续阶段
b2 = resnet_block(64, 64, 2, first_block=True)
b3 = resnet_block(64, 128, 2)
b4 = resnet_block(128, 256, 2)
b5 = resnet_block(256, 512, 2)
# 整合网络
net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10)
)

# 检查各层输出形状
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:', X.shape)

# 训练模型
lr, num_epochs, batch_size = 0.05, 10, 256
device = try_gpu()
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
