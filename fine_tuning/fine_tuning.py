import os
import hashlib
import requests
import zipfile
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# 数据集下载与解压
#@save
data_hub = {
    'hotdog': (
        'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip',
        'fba480ffa8aa7e0febbb511d181409f899b9baa5'
    )
}

def download_extract(name, folder='data'):
    """下载并解压指定数据集"""
    url, sha1 = data_hub[name]
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, f'{name}.zip')
    # 下载
    if not os.path.exists(zip_path):
        print(f'Downloading {name} dataset...')
        r = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
    # 校验
    sha1_hash = hashlib.sha1()
    with open(zip_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha1_hash.update(chunk)
    assert sha1_hash.hexdigest() == sha1, 'SHA1 mismatch, corrupted download.'
    # 解压
    extract_path = os.path.join(folder, name)
    if not os.path.exists(extract_path):
        print(f'Extracting {name} dataset...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(folder)
    return extract_path

# 显示图像网格
#@save
def show_images(imgs, num_rows, num_cols, scale=1.5):
    """在网格中显示图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        img = imgs[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 获取可用 GPU 设备列表
#@save
def try_all_gpus():
    """返回可用的 GPU 设备列表，如果没有 GPU，则返回 CPU"""
    return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] or [torch.device('cpu')]

# 多 GPU/单 GPU 通用训练函数
#@save
def train_ch13(net, train_iter, test_iter, loss_fn, trainer, num_epochs, devices=None):
    """训练并评估模型，支持多GPU"""
    devices = devices or [torch.device('cpu')]
    # 如果不止一个 GPU，则使用 DataParallel
    if len(devices) > 1:
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    else:
        net = net.to(devices[0])

    for epoch in range(num_epochs):
        # 训练
        net.train()
        metric = {'loss_sum': 0.0, 'num_samples': 0, 'num_correct': 0}
        for X, y in train_iter:
            # 多 GPU 时，DataParallel 已自动分发
            X = X.to(devices[0])
            y = y.to(devices[0])
            y_hat = net(X)
            l = loss_fn(y_hat, y).mean()
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric['loss_sum'] += l.item() * y.numel()
            metric['num_samples'] += y.numel()
            metric['num_correct'] += (y_hat.argmax(dim=1) == y).sum().item()
        train_loss = metric['loss_sum'] / metric['num_samples']
        train_acc = metric['num_correct'] / metric['num_samples']
        # 评估
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(devices[0])
                y = y.to(devices[0])
                y_hat = net(X)
                correct += (y_hat.argmax(dim=1) == y).sum().item()
                total += y.numel()
        test_acc = correct / total
        print(f'epoch {epoch + 1}, loss {train_loss:.4f}, ' \
              f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')


# 主脚本执行逻辑
if __name__ == '__main__':
    # 获取数据集
    data_dir = download_extract('hotdog', folder='data')

    # 创建训练集和测试集
    train_imgs = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'))
    test_imgs = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'))

    # 随机挑选前后共 16 张图片进行展示
    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

    # 图像增强与标准化
    normalize = torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize])

    # 加载预训练模型并替换输出层
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    finetune_net = torchvision.models.resnet18(pretrained=True)
    #只对最后一层的weight做权重初始化
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)

    # 微调模型
    def train_fine_tuning(net, learning_rate, batch_size=128,
                          num_epochs=5, param_group=True):
        train_iter = DataLoader(
            torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'train'), transform=train_augs),
            batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(
            torchvision.datasets.ImageFolder(
                os.path.join(data_dir, 'test'), transform=test_augs),
            batch_size=batch_size)
        devices = try_all_gpus()
        loss = nn.CrossEntropyLoss()
        if param_group:
            params_1x = [param for name, param in net.named_parameters()
                         if name not in ['fc.weight', 'fc.bias']]
            trainer = torch.optim.SGD([
                #这里是别的层由于已经调好了，用较小的学习率；而最后一层就用较大的学习率
                {'params': params_1x},
                {'params': net.fc.parameters(), 'lr': learning_rate * 10}
            ], lr=learning_rate, weight_decay=0.001)
        else:
            trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                      weight_decay=0.001)
        train_ch13(net, train_iter, test_iter, loss, trainer,
                   num_epochs, devices)

    # 使用较小的学习率微调
    train_fine_tuning(finetune_net, 5e-5)

    # 从头训练模型
    scratch_net = torchvision.models.resnet18(pretrained=False)
    #从头训练模型同样需要把后面的层改成合适输出类型的线性层，只不过没引入预训练模型（参数）而已
    scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
    train_fine_tuning(scratch_net, 5e-4, param_group=False)
