import os
import random
import zipfile
import requests
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# 注意：该代码仅处理了sohu_data，并没有验证集或测试集的处理

# # 1. 下载和解压数据
# def download_wikitext2(data_dir='sohu_data'):
#     url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
#     filename = 'sohu_data.zip'
#     file_path = os.path.join(data_dir, filename)
#
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#
#     if not os.path.exists(file_path):
#         print("Downloading Sohu_data...")
#         response = requests.get(url, stream=True)
#         total_size = int(response.headers.get('content-length', 0))
#         # tqdm 显示下载进度条
#         with open(file_path, 'wb') as f, tqdm(
#             desc=filename, total=total_size, unit='B', unit_scale=True
#         ) as bar:
#             for data in response.iter_content(chunk_size=1024):
#                 f.write(data)
#                 bar.update(len(data))
#
#     # 解压 zip 文件
#     with zipfile.ZipFile(file_path, 'r') as zip_ref:
#         zip_ref.extractall(data_dir)
#     return os.path.join(data_dir, 'sohu_data')

# 2. 分句和分词
def read_wiki(data_dir):
    file_path = os.path.join(data_dir, 'sohu_data.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 按行读取，每行用 ' . ' 分割成句子列表，只保留长度≥2的行，去除首尾空白，并转为小写
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.strip().split(' . ')) >= 2]
    # 打乱段落顺序，保证训练数据随机性
    random.shuffle(paragraphs)
    return paragraphs

def tokenize(text):
    # 简单英文分词（每个字符串句子中提取单词，去除非字母数字）
    # 返回每个句子对应的词列表，text 是句子列表
    return [re.findall(r"\b\w+\b", sentence.lower()) for sentence in text]

# 3. 构建词表
class Vocab:
    def __init__(self, tokens, min_freq=5, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计所有词频
        counter = Counter([tk for line in tokens for tk in line])
        # 先把保留词加入词表
        self.idx_to_token = list(reserved_tokens)
        self.token_to_idx = {tk: i for i, tk in enumerate(self.idx_to_token)}
        # 词频≥min_freq且不在保留词中的词加入词表
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.token_to_idx)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token_or_tokens):
        # 支持输入单词或单词列表，返回对应索引
        if isinstance(token_or_tokens, list):
            return [self[token] for token in token_or_tokens]
        # 不在词表的用 <unk> 索引替代
        return self.token_to_idx.get(token_or_tokens, self.token_to_idx['<unk>'])

    def to_tokens(self, indices):
        # 索引转词
        if isinstance(indices, list):
            return [self.idx_to_token[i] for i in indices]
        return self.idx_to_token[indices]

# 4. 生成NSP和MLM数据
def get_next_sentence(sent_a, sent_b, paragraphs):
    # 50%概率为正确的下一句，50%随机选择别的句子作为负样本
    if random.random() < 0.5:
        return sent_a, sent_b, True
    else:
        # 随机从随机段落里随机选一句作为负样本
        sent_b = random.choice(random.choice(paragraphs))
        return sent_a, sent_b, False

def get_tokens_and_segments(tokens_a, tokens_b):
    # 拼接输入token序列并添加特殊符号：CLS开头，SEP分隔
    tokens = ['<cls>'] + tokens_a + ['<sep>'] + tokens_b + ['<sep>']
    # segments标签：tokens_a部分为0，tokens_b部分为1，用于区分句子
    segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    return tokens, segments

def replace_mlm_tokens(tokens, candidate_pos, num_preds, vocab):
    input_tokens = tokens[:]  # 复制tokens
    pred_labels = []
    # 打乱候选预测词的位置顺序
    random.shuffle(candidate_pos)
    # 只遮蔽num_preds个词
    for pos in candidate_pos[:num_preds]:
        original = tokens[pos]
        # 80%概率替换为 <mask>
        if random.random() < 0.8:
            input_tokens[pos] = '<mask>'
        # 10%概率保持原词
        elif random.random() < 0.5:
            input_tokens[pos] = original
        # 10%概率替换为随机词
        else:
            input_tokens[pos] = random.choice(vocab.idx_to_token)
        # 记录预测位置和对应真实词，用于计算损失
        pred_labels.append((pos, original))
    # 按预测位置排序
    pred_labels.sort(key=lambda x: x[0])
    pred_positions = [p for p, _ in pred_labels]
    pred_tokens = [t for _, t in pred_labels]
    # 返回输入tokens的id，预测位置，和真实预测词id
    return vocab[input_tokens], pred_positions, vocab[pred_tokens]

# 5. 组织Dataset
class WikiTextDataset(Dataset):
    def __init__(self, paragraphs, max_len):
        # 对每个段落，先将每个句子 tokenize，保留「段落→句子→词」三层结构
        tokenized = [tokenize(p) for p in paragraphs]
        # 把所有段落里的句子拼成一个大词列表，用于构建词表
        sentences = [word for para in tokenized for sent in para for word in sent]
        # 构建词表，包含特殊token和出现频率>=5的词
        self.vocab = Vocab([sentences], min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>', '<unk>'])

        examples = []
        for paragraph in tokenized:
            # 对每个段落中相邻句子构造NSP样本
            for i in range(len(paragraph) - 1):
                a, b, is_next = get_next_sentence(paragraph[i], paragraph[i + 1], tokenized)
                # 超长样本过滤
                if len(a) + len(b) + 3 > max_len:
                    continue
                # 生成输入tokens和segment标签
                tokens, segments = get_tokens_and_segments(a, b)
                # 生成MLM输入和标签
                input_ids, pred_pos, mlm_labels = replace_mlm_tokens(tokens,
                    [j for j, tk in enumerate(tokens) if tk not in ['<cls>', '<sep>']],
                    max(1, round(len(tokens) * 0.15)), self.vocab)
                # 汇总样本（MLM输入，预测位置，MLM标签，segment标签，NSP标签）
                examples.append((input_ids, pred_pos, self.vocab[mlm_labels], segments, is_next))

        self.max_len = max_len
        # 最大MLM预测个数，通常为最大长度15%
        self.max_preds = round(max_len * 0.15)
        self.pad_id = self.vocab['<pad>']

        # 统一padding，形成训练输入张量
        self.inputs = self._pad_bert_inputs(examples)

    def _pad_bert_inputs(self, examples):
        all_token_ids, all_segments, valid_lens = [], [], []
        all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels = [], [], [], []

        for (input_ids, pred_pos, mlm_labels, segments, is_next) in examples:
            # 对token id做padding，补齐max_len长度
            padding = [self.pad_id] * (self.max_len - len(input_ids))
            all_token_ids.append(torch.tensor(input_ids + padding))
            # segment标签padding为0
            all_segments.append(torch.tensor(segments + [0] * len(padding)))
            # valid_len是非pad部分长度
            valid_lens.append(torch.tensor(len(input_ids)))

            # 对预测位置做padding，补齐最大预测数
            pred_pad = [0] * (self.max_preds - len(pred_pos))
            all_pred_positions.append(torch.tensor(pred_pos + pred_pad))
            # MLM权重，预测位置为1，padding位置为0，loss时用来mask
            mlm_w = [1.0] * len(mlm_labels) + [0.0] * len(pred_pad)
            all_mlm_weights.append(torch.tensor(mlm_w))
            # MLM标签padding为0
            all_mlm_labels.append(torch.tensor(mlm_labels + [0] * len(pred_pad)))
            # NSP标签，0/1二分类
            nsp_labels.append(torch.tensor(int(is_next)))

        # 返回多任务训练所需所有张量的元组列表
        return list(zip(all_token_ids, all_segments, valid_lens,
                        all_pred_positions, all_mlm_weights,
                        all_mlm_labels, nsp_labels))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 按索引返回所有训练样本张量
        return self.inputs[idx]

# 6. 加载数据
def load_data_wiki(batch_size, max_len):
    # data_dir = download_wikitext2()
    data_dir = "./sohu_data"
    paragraphs = read_wiki(data_dir)
    dataset = WikiTextDataset(paragraphs, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.vocab

# 7. 测试
if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
         mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
              pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
              nsp_y.shape)
        break

    print("词表大小：", len(vocab))
