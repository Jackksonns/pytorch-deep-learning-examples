import os
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
import tarfile
import urllib.request
import ssl

def download_voc2012_from_mirror(target_dir='VOCdevkit'):
    voc_root = os.path.join(target_dir, 'VOC2012')
    if os.path.isdir(voc_root):
        print(f"[✔] 已存在：{voc_root}")
        return

    url = 'http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar'
    filename = 'VOCtrainval_11-May-2012.tar'
    download_path = os.path.join(target_dir, filename)
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(download_path):
        print(f"[↓] 正在下载：{url}")
        urllib.request.urlretrieve(url, download_path)
        print(f"[✔] 下载到：{download_path}")
    else:
        print(f"[!] 压缩包已存在：{download_path}")

    print(f"[📦] 解压到 {target_dir} …")
    with tarfile.open(download_path) as tar:
        tar.extractall(path=target_dir)
    print("[✔] 解压完成！")


# 全卷积网络（全连接卷积神经网络-FCN）
# 搭建模型
# 使用预训练的ResNet18去掉最后的全连接层和平均池化层
#使用在ImageNet数据集上预训练的ResNet-18作为源模型。——提取图像特征
pretrained_net = models.resnet18(pretrained=True)
# 查看原网络最后3层结构
print(list(pretrained_net.children())[-3:])

# 创建一个全卷积网络net，保留到倒数第3层之前的所有层
net = nn.Sequential(*list(pretrained_net.children())[:-2])

# 测试输出形状
X = torch.rand(size=(1, 3, 320, 480))
print('feature map shape:', net(X).shape)  # (1,512,10,15)

# 使用1x1卷积将输出通道数转换为VOC数据集的类别数
num_classes = 21  # VOC中包括背景共21类
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
# 转置卷积用于上采样，将特征图放大32倍
net.add_module('transpose_conv', nn.ConvTranspose2d(
    num_classes, num_classes, kernel_size=64, padding=16, stride=32))

# 初始化转置卷积层为双线性插值权重
# 双线性插值的上采样可通过转置卷积层实现

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og_x = torch.arange(kernel_size).reshape(-1, 1)
    og_y = torch.arange(kernel_size).reshape(1, -1)
    filt = (1 - torch.abs(og_x - center) / factor) * \
           (1 - torch.abs(og_y - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    # 将filt赋值到对应的对角位置
    for i in range(min(in_channels, out_channels)):
        weight[i, i, :, :] = filt
    return weight

# 用于测试双线性权重的转置卷积，stride=2就是高宽放大两倍
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

# 加载并测试上采样效果
# 读取图像并转换为Tensor
img = transforms.ToTensor()(Image.open('./catdog.jpg'))  # 保留原注释路径
X = img.unsqueeze(0)  # 增加批次维度
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()

# 显示输入和输出图像
print('input image shape:', img.permute(1, 2, 0).shape)
plt.imshow(img.permute(1, 2, 0))
plt.show()
plt.savefig('image1.png')
print('output image shape:', out_img.shape)
plt.imshow(out_img)
plt.show()
plt.savefig('image2.png')

# 用双线性插值初始化FCN中的transpose_conv
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 数据集准备
# 使用torchvision的VOCSegmentation来加载VOC2012分割数据集
# 定义图像和标签的预处理
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

def get_voc_dataloader(batch_size, crop_size, root='VOCdevkit'):
    # 训练集变换：随机裁剪、转Tensor、归一化
    train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    # 验证/测试集变换：中心裁剪、转Tensor、归一化
    val_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    # 标签使用最近邻插值保持整数类别
    target_transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.PILToTensor()
    ])
    train_ds = datasets.VOCSegmentation(root, year='2012', image_set='train', download=False,
                                       transform=train_transform,
                                       target_transform=target_transform)
    val_ds = datasets.VOCSegmentation(root, year='2012', image_set='val', download=False,
                                     transform=val_transform,
                                     target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)
    return train_loader, val_loader

download_voc2012_from_mirror('VOCdevkit')

batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = get_voc_dataloader(batch_size, crop_size)

#训练与评估
# 定义损失函数
# reduction='none'返回每像素损失，后续对每张图像取平均

def loss_fn(inputs, targets):
    # 去掉通道维度确保targets形状[N,H,W]
    targets = targets.squeeze(1).long()
    # 交叉熵损失默认对channel维度做softmax
    return F.cross_entropy(inputs, targets, reduction='mean')

# 设备配置
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] or [torch.device('cpu')]
print('training on:', devices)

# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-3)

# 准确率计算

def evaluate_accuracy(net, data_loader, device):
    net.eval()
    metric = {'loss': 0.0, 'num': 0}
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            l = loss_fn(outputs, y)
            metric['loss'] += l.item() * X.shape[0]
            metric['num'] += X.shape[0]
    return metric['loss'] / metric['num']

# 训练函数

def train(net, train_loader, val_loader, loss_fn, trainer, num_epochs, devices):
    for epoch in range(num_epochs):
        net.train()
        total_loss, total_num = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(devices[0]), y.to(devices[0])
            trainer.zero_grad()
            outputs = net(X)
            l = loss_fn(outputs, y)
            l.backward()
            trainer.step()
            total_loss += l.item() * X.shape[0]
            total_num += X.shape[0]
        train_loss = total_loss / total_num
        val_loss = evaluate_accuracy(net, val_loader, devices[0])
        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss:.4f}, "
              f"Val loss: {val_loss:.4f}")

# 开始训练
num_epochs = 5
train(net, train_iter, test_iter, loss_fn, trainer, num_epochs, devices)

#预测与可视化

def predict(img, net, device):
    # 图像归一化并增加batch维度
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    X = transform(img).unsqueeze(0).to(device)
    pred = net(X).argmax(dim=1)
    #在通道维度做argmax
    return pred[0].cpu()

# VOC颜色映射表
VOC_COLORMAP = [[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128],
                [128,0,128], [0,128,128], [128,128,128], [64,0,0], [192,0,0],
                [64,128,0], [192,128,0], [64,0,128], [192,0,128], [64,128,128],
                [192,128,128], [0,64,0], [128,64,0], [0,192,0], [128,192,0],
                [0,64,128]]

# 将预测标签映射为RGB图像

def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP)
    return colormap[pred.numpy()]

# 从VOC数据集中读取测试图像并展示预测结果
voc_root = 'VOCdevkit'
if not os.path.isdir(voc_root):
    raise FileNotFoundError(f"目录 {voc_root} 未找到，请确保已下载VOC2012数据集。")

# 随机选取n张展示
n = 4
fig, axes = plt.subplots(3, n, figsize=(n*3, 9))
for i in range(n):
    img, label = test_iter.dataset[i]
    pil_img = transforms.ToPILImage()(img)
    # 预测结果
    pred = predict(pil_img, net, devices[0])
    # 可视化原图、预测和真实标签
    axes[0, i].imshow(pil_img)
    axes[0, i].set_title('原图')
    axes[1, i].imshow(label2image(pred))
    axes[1, i].set_title('预测')
    axes[2, i].imshow(transforms.ToPILImage()(label.squeeze(0)))
    axes[2, i].set_title('真实')
    for ax in axes[:, i]:
        ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('image3.png')
