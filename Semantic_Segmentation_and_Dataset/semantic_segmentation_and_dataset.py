import os
import urllib.request
import tarfile
import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

# 定义数据集下载和解压的URL和存储Hub
# Pascal VOC2012 数据集下载链接
DATA_URL = 'http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar'
DATA_HASH = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'


import os
import urllib.request

def download_extract(name, root):
    url = 'http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar'
    fname = os.path.join(root, name + '.tar')  # 构造路径：VOCdevkit/VOC2012.tar

    # ✅ 关键点：确保目录存在
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # 下载
    if not os.path.exists(fname):
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(url, fname)
        print('Download finished.')

    # 解压（你可以使用 tarfile 或其他方式）
    extracted_dir = os.path.join(root, 'VOC2012')
    if not os.path.exists(extracted_dir):
        import tarfile
        with tarfile.open(fname) as tar:
            tar.extractall(path=root)
        print('Extraction finished.')

    return extracted_dir


# 获取可用的 DataLoader 工作线程数
def get_dataloader_workers():
    """返回用于 DataLoader 的 num_workers 数量。"""
    return min(8, os.cpu_count() or 1)

# 读取VOC图像及对应分割标签
# Pascal VOC2012 语义分割数据集

#语义分割的label，需要对每一个像素都有标签。对于同一张图片，每个pixel都有一个唯一的标签。
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像及其标签，返回图像张量列表和标签张量列表。"""
    split = 'train.txt' if is_train else 'val.txt'
    txt_path = os.path.join(voc_dir, 'ImageSets', 'Segmentation', split)
    # 读取文件名列表
    with open(txt_path, 'r') as f:
        image_ids = [line.strip() for line in f]
    features, labels = [], []
    for img_id in image_ids:
        img_path = os.path.join(voc_dir, 'JPEGImages', f'{img_id}.jpg')
        # 读取RGB图像
        feature = read_image(img_path)
        features.append(feature)
        # 读取分割标签图像（保留原始通道以便后续映射）
        lbl_path = os.path.join(voc_dir, 'SegmentationClass', f'{img_id}.png')
        label = read_image(lbl_path, mode=ImageReadMode.RGB)
        labels.append(label)
    return features, labels

# 在画布上展示多张图像
def show_images(imgs, num_rows, num_cols, scale=2):
    """在 matplotlib 画布上并排显示多张图像。"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        img = imgs[i]
        # 如果是张量，转换为 numpy 并调整通道顺序
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).numpy().astype('uint8')
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 定义颜色映射和类别列表
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 构建从RGB三通道映射到类别索引的查找表
def voc_colormap2label():
    """构建一个大小为256^3的张量，用于将RGB封装的索引映射到类别标签。"""
    cm2lbl = torch.zeros(256**3, dtype=torch.long)
    for idx, color in enumerate(VOC_COLORMAP):
        key = (color[0] * 256 + color[1]) * 256 + color[2]
        cm2lbl[key] = idx
    return cm2lbl

# 将RGB标签图映射到类别索引
def voc_label_indices(colormap, cm2lbl):
    """将标签图中的每个像素RGB值映射为类别索引矩阵。"""
    # 调整通道顺序为HWC，并转为 numpy
    arr = colormap.permute(1, 2, 0).numpy().astype('int32')
    # 计算每个像素对应的唯一键值
    idx = (arr[:, :, 0] * 256 + arr[:, :, 1]) * 256 + arr[:, :, 2]
    return cm2lbl[idx]

# 随机裁剪特征图和标签图
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征图和标签图，保持空间对齐。"""
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feat_crop = TF.crop(feature, i, j, h, w)
    label_crop = TF.crop(label, i, j, h, w)
    return feat_crop, label_crop

# 自定义VOC分割数据集
class VOCSegDataset(torch.utils.data.Dataset):
    """一个基于PyTorch的VOC语义分割自定义数据集类。"""
    def __init__(self, is_train, crop_size, voc_dir):
        self.crop_size = crop_size
        # 归一化设置，使用ImageNet预训练模型的均值和标准差
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 读取所有图像及标签
        features, labels = read_voc_images(voc_dir, is_train)
        # 过滤掉尺寸小于裁剪大小的图像
        valid_pairs = [(f, l) for f, l in zip(features, labels)
                       if f.shape[1] >= crop_size[0] and f.shape[2] >= crop_size[1]]
        self.features = [self.normalize(f.float() / 255) for f, _ in valid_pairs]
        self.labels = [l for _, l in valid_pairs]
        # 构建映射表
        self.cm2lbl = voc_colormap2label()
        print(f'read {len(self.features)} examples')

    def __getitem__(self, idx):
        f, lbl = self.features[idx], self.labels[idx]
        # 随机裁剪并映射标签
        f_crop, lbl_crop = voc_rand_crop(f, lbl, *self.crop_size)
        return f_crop, voc_label_indices(lbl_crop, self.cm2lbl)

    def __len__(self):
        return len(self.features)
    
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集，返回训练和测试迭代器"""
    # voc_dir = download_extract('VOCdevkit/VOC2012', root='VOCdevkit')
    voc_dir= 'VOCdevkit/VOC2012'
    num_workers = get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


# 加载数据并创建迭代器
if __name__ == '__main__':
    # 下载并解压VOC2012
    voc_root = download_extract('VOC2012', root='VOCdevkit')
    # 查看前5张图像及对应标签
    feats, labs = read_voc_images(voc_root, True)
    imgs = feats[:5] + labs[:5]
    show_images(imgs, 2, 5)

    # 测试随机裁剪
    crop_imgs = []
    for _ in range(5):
        fc, lc = voc_rand_crop(feats[0], labs[0], 200, 300)
        crop_imgs += [fc, lc]
    show_images(crop_imgs, 2, 5)

    # 创建数据集及DataLoader
    crop_size = (320, 480)
    batch_size = 64
    train_ds = VOCSegDataset(True, crop_size, voc_root)
    test_ds = VOCSegDataset(False, crop_size, voc_root)
    train_loader, test_loader = load_data_voc(batch_size, crop_size)
    for X, Y in train_loader:
        print(X.shape, Y.shape)
        #其实这里的train_loader相当于train_iter
        break
