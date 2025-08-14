import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 自定义 Inception 块，添加 BatchNorm
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层 + BN
        self.p1_1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1)
        )
        # 线路2，1x1卷积层后接3x3卷积层 + BN
        self.p2_1 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0])
        )
        self.p2_2 = nn.Sequential(
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c2[1])
        )
        # 线路3，1x1卷积层后接5x5卷积层 + BN
        self.p3_1 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0])
        )
        self.p3_2 = nn.Sequential(
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(c3[1])
        )
        # 线路4，3x3最大汇聚层后接1x1卷积层 + BN
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Sequential(
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4)
        )

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

# 辅助分类器模块
class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super(AuxClassifier, self).__init__(**kwargs)
        # 平均池化 + 1x1卷积 + 全连接
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # 缩小尺寸
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 完整 GoogLeNet 网络，包含两个辅助分类器
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        # 主干网络
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # b3 中两个 Inception
        self.b3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.b3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # b4 中五个 Inception
        self.b4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.b4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.b4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.b4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.b4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # b5 两个 Inception + 全局池化
        self.b5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.b5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # 主分类器
        self.fc = nn.Linear(1024, num_classes)
        # 两个辅助分类器，注意第一个输入通道改为480
        self.aux1 = AuxClassifier(480, num_classes)
        self.aux2 = AuxClassifier(512, num_classes)

    def forward(self, x):
        # 主干前两层
        x = self.b1(x)
        x = self.b2(x)
        # b3
        x = self.b3_1(x)
        x = self.b3_2(x)
        # 辅助分类器1输出
        aux1_out = self.aux1(x)
        x = self.maxpool3(x)
        # b4
        x = self.b4_1(x)
        x = self.b4_2(x)
        # 辅助分类器2输出
        aux2_out = self.aux2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)
        x = self.maxpool4(x)
        # b5
        x = self.b5_1(x)
        x = self.b5_2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        # 主分类器输出
        main_out = self.fc(x)
        return main_out, aux1_out, aux2_out

# 查看各个层的输出形状
net = GoogLeNet(num_classes=10)
X = torch.rand(size=(1, 1, 96, 96))
main, a1, a2 = net(X)
print('主分类器输出形状:', main.shape)
print('辅助分类器1输出形状:', a1.shape)
print('辅助分类器2输出形状:', a2.shape)

# 训练相关参数
lr, num_epochs, batch_size = 0.01, 20, 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理与加载
transform = transforms.Compose([
    transforms.Resize(96),         # 调整图像大小以匹配 GoogLeNet 输入要求
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 数据标准化
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 模型迁移到设备
net.to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# 评估准确率

def evaluate_accuracy(data_iter, net, device=None):
    if device is None:
        device = next(net.parameters()).device
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            main_out, _, _ = net(X)
            acc_sum += (main_out.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 模型训练函数，包含辅助损失

def train(net, train_iter, test_iter, loss_fn, num_epochs, optimizer, device):
    print("training on", device)
    net.to(device)
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        start = time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            main_out, aux1_out, aux2_out = net(X)
            # 主损失 + 0.3*aux1 + 0.3*aux2
            l1 = loss_fn(main_out, y)
            l2 = loss_fn(aux1_out, y)
            l3 = loss_fn(aux2_out, y)
            loss = l1 + 0.3 * l2 + 0.3 * l3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += l1.item()
            train_acc_sum += (main_out.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device)
        print(f'epoch {epoch + 1}, loss {train_loss_sum / batch_count:.4f}, '
              f'train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}, '
              f'time {time.time() - start:.1f} sec')

# 启动训练
train(net, train_loader, test_loader, criterion, num_epochs, optimizer, device)
