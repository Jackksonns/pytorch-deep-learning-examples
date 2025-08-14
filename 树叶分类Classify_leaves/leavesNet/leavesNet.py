import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from dataset import CSVImageDataset
from torch.utils.data import DataLoader

#leavesNet是本人自命名网络，用于树叶分类竞赛（架构可能包含VGG NiN ResNet Inception）
class Residual(nn.Module):
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

# （googleNet的一部分）自定义 Inception 块，添加 BatchNorm
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





