# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# # 定义NiN模块

# def nin_block(in_channels, out_channels, kernel_size, strides, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
#     )

# # 构建NiN网络
# net = nn.Sequential(
#     nin_block(1, 96, kernel_size=11, strides=4, padding=0),  # 对大图有效，小图通常会导致信息损失
#     nn.MaxPool2d(3, stride=2),
#     nin_block(96, 256, kernel_size=5, strides=1, padding=2),
#     nn.MaxPool2d(3, stride=2),
#     nin_block(256, 384, kernel_size=3, strides=1, padding=1),
#     nn.MaxPool2d(3, stride=2),
#     nn.Dropout(0.5),
#     # 标签类别数是10
#     nin_block(384, 10, kernel_size=3, strides=1, padding=1),
#     nn.AdaptiveAvgPool2d((1, 1)),
#     # 将四维的输出转成二维的输出，其形状为(批量大小,10)
#     nn.Flatten()
# )

# # 打印每一层输出形状，便于调试
# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)

# # 超参数设置
# # 降低学习率，添加动量
# lr, momentum, num_epochs, batch_size = 0.01, 0.9, 10, 128

# # 数据预处理和加载
# # 增加归一化，提高训练稳定性
# transform = transforms.Compose([
#     transforms.Resize(224),  # 若图像尺寸过大，可改为 Resize(64)
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # 训练集与测试集使用不同的数据集
# train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
# train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
# test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # 选择设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')
# net.to(device)

# # 定义损失和优化器
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# # 训练与评估循环
# for epoch in range(num_epochs):
#     net.train()
#     train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
#     for X, y in train_iter:
#         X, y = X.to(device), y.to(device)
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         optimizer.zero_grad()
#         l.backward()
#         optimizer.step()
#         train_loss_sum += l.item() * y.size(0)
#         train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
#         n += y.size(0)

#     net.eval()
#     with torch.no_grad():
#         test_acc = 0.0
#         m = 0
#         for X, y in test_iter:
#             X, y = X.to(device), y.to(device)
#             y_hat = net(X)
#             test_acc += (y_hat.argmax(dim=1) == y).sum().item()
#             m += y.size(0)
#     print(f'Epoch {epoch + 1}/{num_epochs}, ' 
#           f'loss {train_loss_sum / n:.4f}, ' 
#           f'train acc {train_acc_sum / n:.3f}, ' 
#           f'test acc {test_acc / m:.3f}')


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义NiN模块，添加 BatchNorm 以稳定训练
# 保留中文注释
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

# 针对 Fashion-MNIST 校准的网络，使用较小卷积核
net = nn.Sequential(
    # 小图适用较小核与步幅1
    nin_block(1, 64, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(2, stride=2),
    nin_block(64, 128, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(2, stride=2),
    nin_block(128, 256, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(2, stride=2),
    nn.Dropout(0.5),
    # 输出类别
    nin_block(256, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

# 打印每层输出形状，确认尺寸
X = torch.rand(size=(1, 1, 32, 32))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 超参数设置
lr, momentum, num_epochs, batch_size = 0.01, 0.9, 10, 128

# 数据预处理：保持原 28x28，并Resize到32，加入归一化
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
net.to(device)

# 损失和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# 训练评估
for epoch in range(num_epochs):
    net.train()
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss_sum += l.item() * y.size(0)
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.size(0)

    net.eval()
    with torch.no_grad():
        test_acc, m = 0.0, 0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            test_acc += (y_hat.argmax(dim=1) == y).sum().item()
            m += y.size(0)
    print(f'Epoch {epoch+1}/{num_epochs}, '  
          f'loss {train_loss_sum/n:.4f}, '  
          f'train acc {train_acc_sum/n:.3f}, '  
          f'test acc {test_acc/m:.3f}')
