import os
import zipfile
import requests
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
#注意：这段代码没有调用任何深度学习模型
#这段代码并不是在识别香蕉，而是在展示已标注的真实标签（ground truth）bounding box
#（图中标出的框是数据集中原本就带的标注标签，并非模型预测出来的结果）
# 下载并解压数据集
import urllib.request
import ssl

def download_extract_banana_detection():
    url = 'https://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip'
    fname = 'banana-detection.zip'
    target_dir = './banana-detection'

    if not os.path.exists(fname):
        print('Downloading...', fname)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as response, open(fname, 'wb') as out_file:
            out_file.write(response.read())

    if not os.path.exists(target_dir):
        print('Extracting...', fname)
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall('.')

    return target_dir


# 读取数据集——这里是把所有的图片读到内存里面（比较傻，但是由于这是小数据集，没有关系）
# 好的方法是要使用时再读取。
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = download_extract_banana_detection()
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256  #这里的处理是把读到的数据除以256，变成0-1之间的数

# 一个用于加载香蕉检测数据集的自定义数据集
class BananasDataset(Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

# 为训练集合测试集返回数据加载器实例
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = DataLoader(BananasDataset(is_train=True),
                            batch_size, shuffle=True)
    val_iter = DataLoader(BananasDataset(is_train=False),
                          batch_size)
    return train_iter, val_iter

# 展示图像
def show_images(imgs, num_rows, num_cols, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    return axes

# 显示边界框
def show_bboxes(axes, bboxes, colors):
    """显示所有边界框"""
    for ax, bbox_list in zip(axes, bboxes):
        for bbox in bbox_list:
            color = colors[0] if isinstance(colors, list) else colors
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0],
                bbox[3] - bbox[1], fill=False, edgecolor=color,
                linewidth=2)
            ax.add_patch(rect)

# 示例运行
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)  # 图像和标签形状

# 展示10幅带有真实边界框的图像。
# 我们可以看到在所有这些图像中香蕉的旋转角度、大小和位置都有所不同。
# 当然，这只是一个简单的人工数据集，实践中真实世界的数据集通常要复杂得多。
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = show_images(imgs, 2, 5, scale=2)
show_bboxes(
    axes,
    [[(label[0][1:5] * edge_size).tolist()] for label in batch[1][0:10]], #这里edge_size为256，因为之前除了，这里要乘上。
    colors='w'
)
plt.show()
