# 纯 PyTorch 实现的 BERT 预训练
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math
import re
import time
import matplotlib.pyplot as plt
from datetime import datetime

# 工具类定义
class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def sum(self):
        return sum(self.times)

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, save_path=None):
        self.fig, self.ax = plt.subplots()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.save_path = save_path  # 新增保存路径
        self.X, self.Y = [], [[] for _ in legend]

    def add(self, x, ys):
        self.X.append(x)
        for i, y in enumerate(ys):
            self.Y[i].append(y)
        self.ax.clear()
        for y, label in zip(self.Y, self.legend):
            self.ax.plot(self.X, y, label=label)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.xlim:
            self.ax.set_xlim(*self.xlim)
        self.ax.legend()
        plt.pause(0.001)

    def save(self, filename=None):
        if filename is None:
            filename = self.save_path or "training_plot.png"
        self.fig.savefig(filename)
        print(f"🎉 图像已保存为：{filename}")


# Tokenizer 和 Vocab 构建（简化）
def tokenize(lines):
    return [re.findall(r'\b\w+\b', line.lower()) for line in lines]

def build_vocab(lines, min_freq=5):
    tokens = [tk for line in tokenize(lines) for tk in line]
    counter = Counter(tokens)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    idx_to_token = ['<unk>', '<pad>', '<cls>', '<sep>', '<mask>']
    idx_to_token += [token for token, freq in token_freqs if freq >= min_freq]
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    return idx_to_token, token_to_idx

# 数据集构建（以预处理好的MLM+NSP样本为例）
class BERTDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return tuple([torch.tensor(x) for x in self.data[idx]])
    def __len__(self):
        return len(self.data)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# BERT 模型定义
class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 key_size, query_size, value_size, hid_in_features,
                 mlm_in_features, nsp_in_features):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_hiddens,
                                                   nhead=num_heads,
                                                   dim_feedforward=ffn_num_hiddens,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLM 预测头
        self.mlm = nn.Sequential(
            nn.Linear(mlm_in_features, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(norm_shape),
            nn.Linear(num_hiddens, vocab_size)
        )

        # NSP 分类头
        self.nsp = nn.Sequential(
            nn.Linear(nsp_in_features, num_hiddens),
            nn.Tanh(),
            nn.Linear(num_hiddens, 2)
        )

    def forward(self, tokens, segments, valid_lens, pred_positions=None):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = self.pos_encoding(X)
        encoded_X = self.encoder(X)

        if pred_positions is not None:
            batch_size = encoded_X.shape[0]
            mlm_input = torch.stack([encoded_X[i, pos] for i, pos in enumerate(pred_positions)], 0)
            mlm_Y_hat = self.mlm(mlm_input)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(encoded_X[:, 0, :])
        return encoded_X, mlm_Y_hat, nsp_Y_hat

# 损失函数计算逻辑
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X,
                         valid_lens_x, pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x, pred_positions_X)
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * \
            mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    nsp_l = loss(nsp_Y_hat, nsp_y)
    return mlm_l, nsp_l, mlm_l + nsp_l

# 训练函数
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, device_ids=devices)
    net.to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, Timer()
    animator = Animator(xlabel='step', ylabel='loss',
                        xlim=[1, num_steps], legend=['mlm', 'nsp'])
    metric = Accumulator(4)
    num_steps_reached = False

    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, \
            mlm_weights_X, mlm_Y, nsp_y in train_iter:

            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size,
                                                   tokens_X, segments_X, valid_lens_x,
                                                   pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            timer.stop()
            metric.add(mlm_l.item(), nsp_l.item(), tokens_X.shape[0], 1)
            animator.add(step + 1, (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on {str(devices)}')
    animator.save(f"bert_training_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


# 编码接口
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

def get_bert_encoding(net, tokens_a, vocab, device, tokens_b=None):
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor([[vocab[token] for token in tokens]], device=device)
    segment_ids = torch.tensor([segments], device=device)
    valid_len = torch.tensor([len(tokens)], device=device)
    encoded_X, _, _ = net(token_ids, segment_ids, valid_len)
    return encoded_X


#以下是测试这个脚本的代码：
import bert
# ======== 构建 toy vocab 和词表映射 ========
idx_to_token = ['<unk>', '<pad>', '<cls>', '<sep>', '<mask>',
                'a', 'crane', 'is', 'flying', 'driver', 'he', 'just', 'left', 'came']
vocab = {token: idx for idx, token in enumerate(idx_to_token)}
vocab_size = len(vocab)

# ======== 构造 toy 数据集，模拟 MLM+NSP ========
# 输入token序列（索引表示）
tokens_X = torch.tensor([
    [2, 5, 6, 7, 8, 3, 0, 0],  # <cls> a crane is flying <sep> <pad> <pad>
    [2, 5, 6, 9, 13, 3, 10, 11]  # <cls> a crane driver came <sep> he just
])
segments_X = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1]
])
valid_lens_x = torch.tensor([5, 8])
pred_positions_X = torch.tensor([
    [2, 3],  # crane, is
    [2, 4]   # crane, came
])
mlm_weights_X = torch.tensor([
    [1, 1],
    [1, 1]
])
mlm_Y = torch.tensor([
    [6, 7],  # crane, is
    [6, 13]  # crane, came
])
nsp_y = torch.tensor([0, 1])  # NSP 标签：第一对为连续句子，第二对不是

dataset = BERTDataset(list(zip(tokens_X, segments_X, valid_lens_x,
                               pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)))
train_iter = DataLoader(dataset, batch_size=2, shuffle=True)

# ======== 模型构建 ========
net = BERTModel(
    vocab_size=vocab_size, num_hiddens=32, norm_shape=[32],
    ffn_num_input=32, ffn_num_hiddens=64, num_heads=2,
    num_layers=2, dropout=0.1,
    key_size=32, query_size=32, value_size=32,
    hid_in_features=32, mlm_in_features=32, nsp_in_features=32
)
devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
loss = nn.CrossEntropyLoss()

# ======== 开始训练 ========
train_bert(train_iter, net, loss, vocab_size, devices, num_steps=10)

# ======== 测试编码调用 ========
tokens_a = ['a', 'crane', 'is', 'flying']
encoded = get_bert_encoding(net, tokens_a, vocab, device=devices[0])
print(f"\n[CLS] embedding: {encoded[:, 0, :5]}")  # 仅打印前5维