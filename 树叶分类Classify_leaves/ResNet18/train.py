import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import CSVImageDataset
from ResNet_leaves import *  
#从训练集拆分出验证集（含标签）进行模型评估，打印 val acc

#对原始测试集（无标签）进行预测，并将结果保存到 submission.csv（格式：img_name,label）

def try_gpu():  # @save
    """如果存在GPU，则使用GPU，否则使用CPU。"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(batch_size, resize=None, val_ratio=0.15):
    """
    加载训练集并拆分出验证集，测试集无标签仅用于预测。
    - batch_size: 批量大小
    - resize: 如果提供，则调整图像大小为 (resize, resize)
    - val_ratio: 验证集比例
    返回: train_iter, val_iter, test_iter
    """
    # 图像转换操作
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # 完整带标签的训练数据
    full_dataset = CSVImageDataset(
        csv_file='../classify-leaves/train.csv',
        image_root='../classify-leaves/',
        transform=transform,
        has_labels=True
    )
    # 按比例拆分为训练集和验证集
    n_val = int(len(full_dataset) * val_ratio)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # 无标签测试集，用于最终预测，不计算准确率
    test_dataset = CSVImageDataset(
        csv_file='../classify-leaves/test.csv',
        image_root='../classify-leaves/',
        transform=transform,
        has_labels=False
    )

    # DataLoader
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, val_iter, test_iter


def evaluate_accuracy(net, data_iter, device):  # @save
    """
    在有标签的数据集上评估模型 net 的准确率。
    """
    net.eval()
    acc_sum, n_samples = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n_samples += y.size(0)
    return acc_sum / n_samples if n_samples > 0 else float('nan')


from torch.optim.lr_scheduler import OneCycleLR

def train_improved(net, train_iter, val_iter, num_epochs, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3,
                           steps_per_epoch=len(train_iter),
                           epochs=num_epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc, epochs_no_improve = 0, 0
    for epoch in range(num_epochs):
        # 训练
        net.train()
        total_loss, total_acc, n = 0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * y.size(0)
            total_acc  += (y_hat.argmax(1)==y).sum().item()
            n += y.size(0)
        train_loss, train_acc = total_loss/n, total_acc/n

        # 验证
        val_acc = evaluate_accuracy(net, val_iter, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc, epochs_no_improve = val_acc, 0
            torch.save(net.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                print("验证集10轮无提升，提前结束训练。")
                break



if __name__ == '__main__':
    # 超参数
    lr, num_epochs, batch_size = 0.001, 100, 256
    device = try_gpu()

    # 加载数据: 返回 train/val/test
    train_iter, val_iter, test_iter = load_data(batch_size, resize=96)

    # 构建网络: ResNet
    b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = resnet_block(64, 64, 2, first_block=True)
    b3 = resnet_block(64, 128, 2)
    b4 = resnet_block(128, 256, 2)
    b5 = resnet_block(256, 512, 2)

    net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 176)
)
    net = net.to(device)

    # 训练与预测
    train_improved(net, train_iter, val_iter, num_epochs, device)
