# #斯坦福自然语言推断（SNLI）数据集
# #[**斯坦福自然语言推断语料库（Stanford Natural Language Inference，SNLI）**]是由500000多个带标签的英语句子对组成的集合 :cite:`Bowman.Angeli.Potts.ea.2015`。我们在路径`../data/snli_1.0`中下载并存储提取的SNLI数据集。
import os
import re
import requests
import zipfile
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

SNLI_URL = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'


def download_and_extract_snli(root_dir: str) -> str:
    """
    下载并解压SNLI数据集到指定目录，返回解压后的数据路径。
    """
    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, 'snli_1.0.zip')
    data_dir = os.path.join(root_dir, 'snli_1.0')
    if not os.path.isdir(data_dir):
        # 下载
        print(f"Downloading SNLI dataset to {zip_path}...")
        response = requests.get(SNLI_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        # 解压
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            # z.extractall(root_dir)
            for member in z.namelist():
                # 清理非法字符（尤其是 '\r'）
                clean_name = member.replace('\r', '')
                target_path = os.path.join(root_dir, clean_name)

                # 如果是目录则创建
                if member.endswith('/'):
                    os.makedirs(target_path, exist_ok=True)
                else:
                    # 确保父目录存在
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with z.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
    else:
        print(f"SNLI data already extracted in {data_dir}")
    return data_dir


def extract_text(s: str) -> str:
    """
    去除多余符号并规范化空格。
    """
    s = re.sub(r"\(", '', s)
    s = re.sub(r"\)", '', s)
    s = re.sub(r"\s{2,}", ' ', s)
    return s.strip()


def read_snli(data_dir: str, split: str = 'train'):
    """
    读取SNLI数据，返回前提、假设、标签列表。
    split: 'train' 或 'test'
    """
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    filename = f"snli_1.0_{split}.txt"
    path = os.path.join(data_dir, filename)
    premises, hypotheses, labels = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.split('\t')
            label = parts[0]
            if label not in label_map:
                continue
            premise = extract_text(parts[1])
            hypothesis = extract_text(parts[2])
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(label_map[label])
    return premises, hypotheses, labels


def tokenize(lines):
    """
    基于空格简单分词。
    返回列表的标记列表。
    """
    return [line.lower().split() for line in lines]


class Vocab:
    """
    根据给定的token列表构建词表。
    支持保留token（如<pad>）并过滤低频词。
    """
    def __init__(self, tokens, min_freq=1, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频
        counter = Counter(tok for line in tokens for tok in line)
        # 排序：频率从高到低，字典序作为次序
        self.idx_to_token = list(reserved_tokens)
        self.token_to_idx = {tok: idx for idx, tok in enumerate(reserved_tokens)}
        # 添加高频token
        for tok, freq in counter.most_common():
            if freq < min_freq:
                break
            if tok not in self.token_to_idx:
                self.idx_to_token.append(tok)
                self.token_to_idx[tok] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        支持单个token或token列表的索引查询。
        """
        if isinstance(tokens, list):
            return [self.__getitem__(tok) for tok in tokens]
        return self.token_to_idx.get(tokens, self.token_to_idx.get('<unk>', 0))


def truncate_pad(line, max_len, pad_idx):
    """
    将序列截断或填充到指定长度。
    """
    if len(line) > max_len:
        return line[:max_len]
    return line + [pad_idx] * (max_len - len(line))


class SNLIDataset(Dataset):
    """
    用于加载SNLI数据的PyTorch数据集。
    """
    def __init__(self, premises, hypotheses, labels, vocab=None, num_steps=50, min_freq=5):
        self.premises_tokens = tokenize(premises)
        self.hypotheses_tokens = tokenize(hypotheses)
        if vocab is None:
            # 构建词表
            self.vocab = Vocab(self.premises_tokens + self.hypotheses_tokens,
                               min_freq=min_freq,
                               reserved_tokens=['<pad>', '<unk>'])
        else:
            self.vocab = vocab
        self.num_steps = num_steps
        self.pad_idx = self.vocab.token_to_idx['<pad>']
        self.premises = torch.tensor([
            truncate_pad(self.vocab[line], self.num_steps, self.pad_idx)
            for line in self.premises_tokens
        ])
        self.hypotheses = torch.tensor([
            truncate_pad(self.vocab[line], self.num_steps, self.pad_idx)
            for line in self.hypotheses_tokens
        ])
        self.labels = torch.tensor(labels)
        print(f"Read {len(self.premises)} examples")

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, root_dir='./bert/data/snli', num_steps=50, num_workers=None):
    """
    下载SNLI数据并返回训练/测试迭代器和词表。
    """
    # 获取工作线程数
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    data_dir = download_and_extract_snli(root_dir)
    train_p, train_h, train_y = read_snli(data_dir, 'train')
    test_p, test_h, test_y = read_snli(data_dir, 'test')
    train_set = SNLIDataset(train_p, train_h, train_y, num_steps=num_steps)
    test_set = SNLIDataset(test_p, test_h, test_y,
                           vocab=train_set.vocab,
                           num_steps=num_steps) #代码已经成功封装了处理好的 SNLI 数据集，但它并不会“保存成文件”，而是封装在内存对象里，由 PyTorch 的 DataLoader 和 Dataset 控制。
    train_iter = DataLoader(train_set, batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    test_iter = DataLoader(test_set, batch_size,
                           shuffle=False,
                           num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab


if __name__ == '__main__':
    # 简单测试
    train_iter, test_iter, vocab = load_data_snli(batch_size=128, num_steps=50)
    print(f"vocab size: {len(vocab)}")
    for (X0, X1), Y in train_iter:
        print(X0.shape, X1.shape, Y.shape)
        break

#运行完该脚本后，现在已经有了封装完毕的 train_iter、test_iter 和 vocab，直接在当前脚本调用即可or其他文件import这个文件。