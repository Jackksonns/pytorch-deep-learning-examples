import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CSVImageDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None, has_labels=True):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.has_labels = has_labels

        if self.has_labels:
            # 只有有标签时才构造映射
            self.label2idx = {label: idx for idx, label in enumerate(sorted(self.data.iloc[:, 1].unique()))}
            self.idx2label = {v: k for k, v in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_root, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label_str = self.data.iloc[idx, 1]
            label = self.label2idx[label_str]
            return image, label
        else:
            return image
