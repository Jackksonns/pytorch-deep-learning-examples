import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# 如果使用完整的Kaggle竞赛的数据集，设置demo为False
demo = True

# 下载并解压数据集
if demo:
    import zipfile
    import requests

    def download_extract(url, root='data'):
        os.makedirs(root, exist_ok=True)
        fname = os.path.join(root, url.split('/')[-1])
        if not os.path.exists(fname):
            print("Downloading...")
            r = requests.get(url)
            with open(fname, 'wb') as f:
                f.write(r.content)
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(root)
        return os.path.join(root, 'kaggle_cifar10_tiny')

    data_dir = download_extract(
        'https://d2l-data.s3.amazonaws.com/kaggle_cifar10_tiny.zip')
else:
    data_dir = '../data/cifar-10/'

# 读取标签
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))

def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label_count.get(label, 0) < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

# 预处理与数据加载
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)

transform_train = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])])

train_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test/train'), transform=transform_train)
train_valid_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test/train_valid'), transform=transform_train)
valid_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test/valid'), transform=transform_test)
test_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test/test'), transform=transform_test)

train_iter = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
train_valid_iter = DataLoader(train_valid_ds, batch_size, shuffle=True, drop_last=True)
valid_iter = DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = DataLoader(test_ds, batch_size, shuffle=False)

# 模型定义
def get_net():
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net

def evaluate_accuracy_gpu(net, data_iter, device):
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train_batch(net, X, y, loss_fn, optimizer, device):
    net.train()
    optimizer.zero_grad()
    X, y = X.to(device), y.to(device)
    y_hat = net(X)
    loss = loss_fn(y_hat, y).mean()
    loss.backward()
    optimizer.step()
    acc = (y_hat.argmax(dim=1) == y).float().mean().item()
    return loss.item(), acc

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

    for epoch in range(num_epochs):
        net.train()
        metric = [0.0, 0.0, 0]
        for X, y in train_iter:
            l, acc = train_batch(net, X, y, loss, optimizer, devices[0])
            metric[0] += l * y.size(0)
            metric[1] += acc * y.size(0)
            metric[2] += y.size(0)

        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        valid_acc = evaluate_accuracy_gpu(net, valid_iter, devices[0]) if valid_iter else None
        print(f"epoch {epoch + 1}, train loss {train_loss:.3f}, train acc {train_acc:.3f}" +
              (f", valid acc {valid_acc:.3f}" if valid_acc is not None else ""))
        scheduler.step()

# 训练模型
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] or [torch.device('cpu')]
num_epochs, lr, wd = 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 重新训练用于提交
net = get_net()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 预测
net = net.to(devices[0])
net.eval()
preds = []
with torch.no_grad():
    for X, _ in test_iter:
        y_hat = net(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
