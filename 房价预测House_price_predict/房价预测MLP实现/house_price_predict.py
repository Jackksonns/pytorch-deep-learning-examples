import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#注意：在这段代码里没有在真正的 Kaggle “测试集”（kaggle_house_test.csv）上计算过任何指标，因为那份数据是没有标签 (SalePrice) 的——它只能用来做最终提交，无法直接评估误差。你在训练过程中看到的 “test acc” 或 “验证 log rmse” 其实是代码在交叉验证（K‑折）或显式划分训练集时从训练数据里抽出来的一部分“验证集”（validation set）上计算的，而不是那份真正的无标签测试集。

# 下载和缓存数据集
#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

#这里的下载代码不知道为啥总是会下载到当前目录的上一级目录下
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名
    """
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件
    """
    for name in DATA_HUB:
        download(name)

# 访问和阅读数据集
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data.shape)
print(test_data.shape)

# 看看[前四个和最后两个特征，以及相应标签]（房价）。
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 我们可以看到，(在每个样本中，第一个特征是ID，)
# 这有助于模型识别每个训练样本。
# 虽然这很方便，但它不携带任何用于预测的信息。
# 因此，在将数据提供给模型之前，(我们将其从数据集中删除)。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数据预处理
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 保存原始数据用于新模型
original_train_data = train_data.iloc[:, 1:-1].copy()  # 不包括ID和标签
original_test_data = test_data.iloc[:, 1:].copy()      # 不包括ID

# 为类别特征准备数据
categorical_features = original_train_data.dtypes[original_train_data.dtypes == 'object'].index
# 确保测试集有相同的列
original_all_features = pd.concat([original_train_data, original_test_data], axis=0, ignore_index=True)

# 获取类别特征的词汇表大小
cat_vocab_sizes = []
# 创建类别编码映射
cat_encoders = {}
for col in categorical_features:
    # 统一处理NaN值并转换为字符串
    original_all_features[col] = original_all_features[col].fillna('missing').astype(str)
    # 创建类别到索引的映射
    unique_vals = original_all_features[col].astype('category').cat.categories
    encoder = {val: i for i, val in enumerate(unique_vals)}
    cat_encoders[col] = encoder
    vocab_size = len(unique_vals)
    cat_vocab_sizes.append(vocab_size if vocab_size > 0 else 1)

# 处理离散值，用独点编码替换
all_features = pd.get_dummies(all_features, dummy_na=True)
# 确保所有列都是 float32
all_features = all_features.astype(np.float32)

# 将其转换为张量表示用于训练。
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 新的HousePriceMLP模型
class HousePriceMLP(nn.Module):
    def __init__(self, num_numeric, cat_vocab_sizes, emb_dim=4):
        super().__init__()
        # 减少嵌入维度以降低复杂性
        self.embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cat_vocab_sizes])
        input_dim = num_numeric + emb_dim * len(cat_vocab_sizes)
        # 简化网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat([x_num, *embs], dim=1)
        output = self.net(x)
        # 确保输出是二维的，形状为(batch_size, 1)
        if output.dim() == 1:
            output = output.unsqueeze(1)
        # 确保输出为正数
        output = torch.clamp(output, min=1e-3)
        return output

# 训练
loss = nn.MSELoss()
in_features = train_features.shape[1]

# get_net函数里面创建了网络，构建了要训练的模型。要改模型在这里改
def get_net():
    # 使用新的HousePriceMLP模型
    num_numeric = len(numeric_features)
    return HousePriceMLP(num_numeric, cat_vocab_sizes)

def prepare_features_for_mlp(data, numeric_features, categorical_features, cat_encoders, is_train=True):
    """为MLP模型准备特征"""
    # 数值特征
    numeric_data = data[numeric_features].copy()
    # 填充数值特征中的NaN值
    numeric_data = numeric_data.fillna(numeric_data.mean())
    # 标准化数值特征
    numeric_data = (numeric_data - numeric_data.mean()) / (numeric_data.std() + 1e-8)
    x_num = torch.tensor(numeric_data.values, dtype=torch.float32)
    
    # 类别特征
    categorical_data = data[categorical_features].copy()
    # 使用预定义的编码器转换为类别编码
    for col in categorical_features:
        # 处理NaN值并转换为字符串
        categorical_data[col] = categorical_data[col].fillna('missing').astype(str)
        # 应用预定义的编码映射
        categorical_data[col] = categorical_data[col].map(cat_encoders[col])
        # 处理可能的NaN映射（如果测试集中有训练集中没有的类别）
        categorical_data[col] = categorical_data[col].fillna(0).astype(int)
    
    x_cat = torch.tensor(categorical_data.values.astype(np.int64), dtype=torch.long)
    
    return x_num, x_cat

# 修改log_rmse函数以适配新模型
def log_rmse(net, features, labels):
    # 判断是否是新模型（HousePriceMLP）
    if isinstance(net, HousePriceMLP):
        # 直接使用已准备好的数据进行预测
        preds = net(features[0], features[1])
        # 确保预测值和标签具有相同的维度
        if preds.dim() == 1:
            preds = preds.reshape(-1, 1)
        if labels.dim() == 1:
            labels = labels.reshape(-1, 1)
        # 更严格的裁剪以避免log(0)或log(negative)
        clipped_preds = torch.clamp(preds, 1e-6, float('inf'))
    else:
        preds = net(features)
        # 确保预测值和标签具有相同的维度
        if preds.dim() == 1:
            preds = preds.reshape(-1, 1)
        if labels.dim() == 1:
            labels = labels.reshape(-1, 1)
        # 更严格的裁剪以避免log(0)或log(negative)
        clipped_preds = torch.clamp(preds, 1e-6, float('inf'))
    # 确保标签值也是正数
    clipped_labels = torch.clamp(labels, 1e-6, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(clipped_labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 判断是否是新模型（HousePriceMLP）
    if isinstance(net, HousePriceMLP):
        # 新模型情况：train_features是(x_num, x_cat)的元组
        x_num, x_cat = train_features
        dataset = TensorDataset(x_num, x_cat, train_labels)
        train_iter = DataLoader(dataset, batch_size, shuffle=True)
        # 使用更小的学习率
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=weight_decay)
        
        for epoch in range(num_epochs):
            net.train()  # 设置为训练模式
            for x_n, x_c, y in train_iter:
                optimizer.zero_grad()
                output = net(x_n, x_c)
                # 确保输出和目标具有相同的维度
                if output.dim() == 1:
                    output = output.reshape(-1, 1)
                if y.dim() == 1:
                    y = y.reshape(-1, 1)
                # 使用较小的裁剪范围
                output = torch.clamp(output, 1e-3, 1e6)
                y_clipped = torch.clamp(y, 1e-3, 1e6)
                l = loss(output, y_clipped)
                if not torch.isnan(l) and not torch.isinf(l):
                    l.backward()
                    # 更严格的梯度裁剪
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                    optimizer.step()
            
            # 评估模式计算损失
            net.eval()
            with torch.no_grad():
                train_rmse = log_rmse(net, (x_num, x_cat), train_labels)
                train_ls.append(train_rmse)
                
                if test_labels is not None:
                    if isinstance(test_features, tuple):
                        test_rmse = log_rmse(net, test_features, test_labels)
                        test_ls.append(test_rmse)
                    else:
                        test_rmse = log_rmse(net, test_features, test_labels)
                        test_ls.append(test_rmse)
                    
                    # 检查是否有异常值
                    if np.isnan(train_ls[-1]) or np.isinf(train_ls[-1]) or \
                       (test_labels is not None and (np.isnan(test_ls[-1]) or np.isinf(test_ls[-1]))):
                        print(f"检测到异常值，第{epoch+1}轮训练后停止")
                        break
                        
            # 训练模式继续训练
            net.train()
            
    else:
        # 原始模型情况
        dataset = TensorDataset(train_features, train_labels)
        train_iter = DataLoader(dataset, batch_size, shuffle=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            net.train()  # 设置为训练模式
            for X, y in train_iter:
                optimizer.zero_grad()
                output = net(X)
                # 确保输出和目标具有相同的维度
                if output.dim() == 1:
                    output = output.reshape(-1, 1)
                if y.dim() == 1:
                    y = y.reshape(-1, 1)
                # 使用较小的裁剪范围
                output = torch.clamp(output, 1e-3, 1e6)
                y_clipped = torch.clamp(y, 1e-3, 1e6)
                l = loss(output, y_clipped)
                if not torch.isnan(l) and not torch.isinf(l):
                    l.backward()
                    optimizer.step()
            
            # 评估模式计算损失
            net.eval()
            with torch.no_grad():
                train_rmse = log_rmse(net, train_features, train_labels)
                train_ls.append(train_rmse)
                
                if test_labels is not None:
                    test_rmse = log_rmse(net, test_features, test_labels)
                    test_ls.append(test_rmse)
                    
                    # 检查是否有异常值
                    if np.isnan(train_ls[-1]) or np.isinf(train_ls[-1]) or \
                       (test_labels is not None and (np.isnan(test_ls[-1]) or np.isinf(test_ls[-1]))):
                        print(f"检测到异常值，第{epoch+1}轮训练后停止")
                        break
            
            # 训练模式继续训练
            net.train()
            
    return train_ls, test_ls

#K折交叉验证。适用于小样本数据集、回归任务和分类任务等。（房价预测是回归任务）
def get_k_fold_data(k, i, X, y):
    # 判断X是否是元组（新模型的情况）
    if isinstance(X, tuple):
        # 新模型情况：X是(x_num, x_cat)的元组
        x_num, x_cat = X
        assert k > 1
        fold_size = x_num.shape[0] // k
        x_num_train, x_cat_train, y_train = None, None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            x_num_part, x_cat_part, y_part = x_num[idx, :], x_cat[idx, :], y[idx]
            if j == i:
                x_num_valid, x_cat_valid, y_valid = x_num_part, x_cat_part, y_part
            elif x_num_train is None:
                x_num_train, x_cat_train, y_train = x_num_part, x_cat_part, y_part
            else:
                x_num_train = torch.cat([x_num_train, x_num_part], 0)
                x_cat_train = torch.cat([x_cat_train, x_cat_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return (x_num_train, x_cat_train), y_train, (x_num_valid, x_cat_valid), y_valid
    else:
        # 原始模型情况
        assert k > 1
        fold_size = X.shape[0] // k
        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            # 只有当有足够的数据点时才绘图
            if len(train_ls) > 1:
                plt.figure()
                plt.plot(list(range(1, len(train_ls) + 1)), train_ls, label='train')
                plt.plot(list(range(1, len(valid_ls) + 1)), valid_ls, label='valid')
                plt.xlabel('epoch')
                plt.ylabel('rmse')
                plt.yscale('log')
                plt.legend()
                plt.grid(True)
                plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
# 为新模型准备数据
x_num_train, x_cat_train = prepare_features_for_mlp(original_train_data, numeric_features, categorical_features, cat_encoders)
x_num_test, x_cat_test = prepare_features_for_mlp(original_test_data, numeric_features, categorical_features, cat_encoders, is_train=False)

# 打印调试信息
print("训练数据形状:")
print("数值特征:", x_num_train.shape)
print("类别特征:", x_cat_train.shape)
print("标签:", train_labels.shape)
print("数值特征范围:", x_num_train.min().item(), "到", x_num_train.max().item())
print("类别特征范围:", x_cat_train.min().item(), "到", x_cat_train.max().item())
print("标签范围:", train_labels.min().item(), "到", train_labels.max().item())

# 检查是否有NaN或inf值
print("数值特征中NaN的数量:", torch.isnan(x_num_train).sum().item())
print("数值特征中inf的数量:", torch.isinf(x_num_train).sum().item())
print("类别特征中NaN的数量:", torch.isnan(x_cat_train).sum().item())
print("标签中NaN的数量:", torch.isnan(train_labels).sum().item())

# 降低学习率
train_l, valid_l = k_fold(k, (x_num_train, x_cat_train), train_labels, num_epochs, lr*0.001, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')

# 提交kaggle预测

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    plt.figure()
    plt.plot(np.arange(1, num_epochs + 1), train_ls)
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    
    # 根据模型类型进行预测
    if isinstance(net, HousePriceMLP):
        preds = net(test_features[0], test_features[1]).detach().numpy()
    else:
        preds = net(test_features).detach().numpy()
    
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission1.csv', index=False)

    
    # 保存模型参数
    torch.save(net.state_dict(), 'kaggle_house_price_model2.pth')
    print("模型已保存为 kaggle_house_price_model2.pth")

# 自动运行训练和预测
train_and_pred((x_num_train, x_cat_train), (x_num_test, x_cat_test), train_labels, test_data, num_epochs, lr*0.001, weight_decay, batch_size)
