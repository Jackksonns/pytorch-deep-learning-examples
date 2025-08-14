import os
import csv
import shutil
import random
import time
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# 下载和解压数据（来自d2l的dog_tiny）
def download_extract_dog_tiny():
    url = 'https://d2l-data.s3.amazonaws.com/kaggle_dog_tiny.zip'
    zip_name = 'kaggle_dog_tiny.zip'
    target_dir = 'dog_tiny'
    if not os.path.exists(zip_name):
        os.system(f"wget {url}")
    if not os.path.exists(target_dir):
        shutil.unpack_archive(zip_name, target_dir)
    return target_dir

# 读取 CSV 标签文件
def read_csv_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.strip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

# 创建训练集和验证集目录
def reorg_train_valid(data_dir, labels, valid_ratio):
    train_dir = os.path.join(data_dir, 'train_valid_test/train')
    valid_dir = os.path.join(data_dir, 'train_valid_test/valid')
    train_valid_dir = os.path.join(data_dir, 'train_valid_test/train_valid')

    for output_dir in [train_dir, valid_dir, train_valid_dir]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    label_count = {}
    for filename, label in labels.items():
        label_count.setdefault(label, [])
        label_count[label].append(filename)

    for label, filenames in label_count.items():
        random.shuffle(filenames)
        n_valid = int(len(filenames) * valid_ratio)
        for i, filename in enumerate(filenames):
            for category in ['train_valid', 'valid' if i < n_valid else 'train']:
                dir_path = os.path.join(data_dir, 'train_valid_test', category, label)
                os.makedirs(dir_path, exist_ok=True)
                shutil.copy(os.path.join(data_dir, 'train', filename + '.jpg'),
                            os.path.join(dir_path, filename + '.jpg'))

# 组织测试集（无标签）
def reorg_test(data_dir):
    test_dir = os.path.join(data_dir, 'train_valid_test/test/unknown')
    os.makedirs(test_dir, exist_ok=True)
    for filename in os.listdir(os.path.join(data_dir, 'test')):
        shutil.copy(os.path.join(data_dir, 'test', filename),
                    os.path.join(test_dir, filename))

# 组织数据集
def reorg_dog_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

# 下载并整理数据
demo = True
data_dir = download_extract_dog_tiny() if demo else os.path.join('..', 'data', 'dog-breed-identification')
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# 图像增强与归一化
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_ds, train_valid_ds = [ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

# 构建微调模型
def get_net(devices):
    net = nn.Sequential()
    net.features = torchvision.models.resnet34(pretrained=True)
    net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 120))
    net = net.to(devices[0])
    for param in net.features.parameters():
        param.requires_grad = False
    return net

loss = nn.CrossEntropyLoss(reduction='none')

# 评估验证损失
def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(devices[0]), y.to(devices[0])
            out = net.output_new(net.features(X))
            l = loss(out, y)
            l_sum += l.sum().item()
            n += y.numel()
    return l_sum / n

# 动画绘图工具（模拟 d2l.Animator）
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None):
        self.fig, self.ax = plt.subplots()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.x_data, self.y_data = [], [[] for _ in legend]
        self.xlim = xlim
        self.lines = [self.ax.plot([], [], label=legend[i])[0]
                      for i in range(len(legend))]
        self.ax.set_xlabel(xlabel)
        self.ax.set_xlim(*xlim)
        self.ax.legend()

    def add(self, x, ys):
        self.x_data.append(x)
        for i, y in enumerate(ys):
            if y is not None:
                self.y_data[i].append(y)
        for i, line in enumerate(self.lines):
            line.set_data(self.x_data, self.y_data[i])
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

# 训练函数
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    num_batches = len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'valid loss'])

    for epoch in range(num_epochs):
        net.train()
        metric = [0.0, 0]
        start = time.time()
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(devices[0]), y.to(devices[0])
            optimizer.zero_grad()
            y_hat = net.output_new(net.features(X))
            l = loss(y_hat, y).sum()
            l.backward()
            optimizer.step()
            metric[0] += l.item()
            metric[1] += y.numel()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter:
            val_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, val_loss))
        scheduler.step()
        print(f'epoch {epoch+1}, train loss {metric[0]/metric[1]:.4f}, '
              f'time {(time.time() - start):.2f}s')

# 自动获取GPU
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] or [torch.device('cpu')]
num_epochs, lr, wd = 10, 1e-4, 1e-4
lr_period, lr_decay = 2, 0.9
net = get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 使用训练+验证集重新训练模型
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 输出预测结果
preds = []
net.eval()
with torch.no_grad():
    for X, _ in test_iter:
        output = nn.functional.softmax(net.output_new(net.features(X.to(devices[0]))), dim=1)
        #这里是对输出进行softmax处理，不是输出一个类答案而是输出其概率——也就是某一个样本对每个类的概率是多少。这是其输出与之前的输出不同的
        preds.extend(output.cpu().numpy())

ids = sorted(os.listdir(os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for img_id, output in zip(ids, preds):
        f.write(img_id.split('.')[0] + ',' + ','.join(map(str, output)) + '\n')

