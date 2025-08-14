from torch.utils.data import Dataset
from PIL import Image
import os

#注意：第一类dataset写法，此时文件夹名称就是label图片的标签（由于图像分类任务）
#而如果是图像恢复任务，则文件夹名称也可以是标签名称（就是一一对应的图片，一个文件夹放破损的图片，一个文件夹放修复后的图片）这样喂给模型训练用。
#而如果是图像恢复数据集的测试集dataset类的话，同样gt和lq都要有，只不过在dataloader循环里面，拿出lq喂给模型，然后gt单独拿出来作为评估指标的数据输入。
class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    
root_train = r'dataset\\train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyDataset(root_train, ants_label_dir)
bees_dataset = MyDataset(root_train, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
#这就是把两个数据集拼在一起组合索引（当然数据集的定义里面就已经区分了不同数据集，所以可以粗暴的+）

#dataset的另外一种写法：
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader

# # 读取数据并划分训练集、测试集
# datas = pd.read_csv('data/data.csv', sep=',', header=None)
# train_data = datas[:int(len(datas) * 0.9)]
# test_data = datas[int(len(datas) * 0.9):]

# # 自定义数据集类
# class Mydataset(Dataset):
#     def __init__(self, train_data):
#         self.x_data = train_data[:, :-1]  # 特征数据（除最后一列）
#         self.y_data = train_data[:, [-1]]  # 标签数据（最后一列）
    
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]  # 按索引返回样本
    
#     def __len__(self):
#         return len(self.x_data)  # 数据集长度

# # 构建数据集和数据加载器
# trainset = Mydataset(train_data)
# train_loader = DataLoader(trainset, batch_size=4, shuffle=False)

# # 测试打印第一个batch
# for batch in train_loader:
#     print(batch)
#     break

print(len(train_dataset))
print(train_dataset[0])

img, label = train_dataset[0]
img.show()

img2, label2 = train_dataset[123]
img2.show()

img3, label3 = train_dataset[124]
img3.show()

#接下来要转化成tensor的形式以适应pytorch
import torchvision
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(degrees=15),
                            
])

#使用torchvision下载并查看数据集
import torchvision

train_set=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=dataset_transform)
test_set=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=dataset_transform)

print(train_set[0])
print(test_set[0])

img1, label = train_set[0]
img2, traget = test_set[0]
print(test_set.classes[traget])


