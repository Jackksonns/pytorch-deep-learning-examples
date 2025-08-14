import math
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu

#注意：如果要复用Transformer块，如何复用？
#1. 把Transformer结构里面的类的实现，全部复制到新py文件；
#2. 对特定的下游任务实现自己的Embedding 层；
#3. 数据读入、训练／评估循环也根据特定下游任务撰写即可。

# Configuration
DATA_DIR = 'data/train'  # 目录下应有 train.en 和 train.fr
SRC_FILE = os.path.join(DATA_DIR, 'train.en')
TGT_FILE = os.path.join(DATA_DIR, 'train.fr')
SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '<unk>']

#Tokenizer & Vocab
def tokenize(text):
    return text.lower().strip().split()

class Vocab:
    def __init__(self, tokens, min_freq=1):
        counter = Counter(tokens)
        self.itos = SPECIAL_TOKENS + [tok for tok, freq in counter.items() if freq >= min_freq and tok not in SPECIAL_TOKENS]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi.get(tok, self.stoi['<unk>']) for tok in tokens]

    def __len__(self):
        return len(self.itos)

#Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_vocab, tgt_vocab, max_len=50):
        self.src_sents = open(src_path, encoding='utf-8').read().splitlines()
        self.tgt_sents = open(tgt_path, encoding='utf-8').read().splitlines()
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src_tokens = ['<bos>'] + tokenize(self.src_sents[idx]) + ['<eos>']
        tgt_tokens = ['<bos>'] + tokenize(self.tgt_sents[idx]) + ['<eos>']
        src_ids = self.src_vocab.encode(src_tokens)[:self.max_len]
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)[:self.max_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab.stoi['<pad>'])
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab.stoi['<pad>'])
    return src_padded.to(device), tgt_padded.to(device)

#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, num_hiddens, 2)
        div_term = torch.exp(i * -math.log(10000.0) / num_hiddens)
        pe = torch.zeros((1, max_len, num_hiddens))
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pos_encoding', pe)

    def forward(self, X):
        X = X + self.pos_encoding[:, :X.size(1), :]
        return self.dropout(X)

#Attention & FFN & Norm
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_size,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)

    def forward(self, Q, K, V, key_padding_mask=None, attn_mask=None):
        out, _ = self.attn(Q, K, V,
                          key_padding_mask=key_padding_mask,
                          attn_mask=attn_mask)
        return out

class PositionWiseFFN(nn.Module):
    def __init__(self, dim_model, dim_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model)
        )

    def forward(self, X): return self.net(X)

class AddNorm(nn.Module):
    def __init__(self, dim_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, X, Y): return self.norm(X + self.dropout(Y))

#Encoder & Decoder Blocks
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dim_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.addnorm1 = AddNorm(embed_size, dropout)
        self.ffn = PositionWiseFFN(embed_size, dim_ff)
        self.addnorm2 = AddNorm(embed_size, dropout)

    def forward(self, X, src_mask):
        out = self.attn(X, X, X, key_padding_mask=src_mask)
        Y = self.addnorm1(X, out)
        return self.addnorm2(Y, self.ffn(Y))

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.addnorm1 = AddNorm(embed_size, dropout)
        self.enc_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.addnorm2 = AddNorm(embed_size, dropout)
        self.ffn = PositionWiseFFN(embed_size, dim_ff)
        self.addnorm3 = AddNorm(embed_size, dropout)

    def forward(self, X, enc_out, src_mask, tgt_mask):
        out1 = self.self_attn(X, X, X, attn_mask=tgt_mask)
        Y1 = self.addnorm1(X, out1)
        out2 = self.enc_attn(Y1, enc_out, enc_out, key_padding_mask=src_mask)
        Y2 = self.addnorm2(Y1, out2)
        return self.addnorm3(Y2, self.ffn(Y2))

#Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_size, num_heads, dim_ff, num_layers, dropout, device):
        super().__init__()
        self.device = device
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(embed_size, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_size, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)

    def make_src_mask(self, src):
        return (src == src_vocab.stoi['<pad>'])

    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device), diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc = self.pos_enc(self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim))
        for layer in self.encoder_layers:
            enc = layer(enc, src_mask)
        dec = self.pos_enc(self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim))
        for layer in self.decoder_layers:
            dec = layer(dec, enc, src_mask, tgt_mask)
        return self.fc_out(dec)

#Training, Evaluation, Translation
def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_labels.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def translate(model, sentence, max_len=50):
    model.eval()
    tokens = ['<bos>'] + tokenize(sentence) + ['<eos>']
    ids = [src_vocab.stoi.get(tok, src_vocab.stoi['<unk>']) for tok in tokens]
    src_tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    outputs = [tgt_vocab.stoi['<bos>']]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)
        preds = model(src_tensor, tgt_tensor)
        next_id = preds.argmax(-1)[0, -1].item()
        outputs.append(next_id)
        if next_id == tgt_vocab.stoi['<eos>']:
            break
    return [tgt_vocab.itos[i] for i in outputs]

#主函数
def main():
    nltk.download('punkt')
    global device, src_vocab, tgt_vocab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据构建词表
    src_tokens = []
    tgt_tokens = []
    with open(SRC_FILE, encoding='utf-8') as f_src, open(TGT_FILE, encoding='utf-8') as f_tgt:
        for s, t in zip(f_src, f_tgt):
            src_tokens += tokenize(s)
            tgt_tokens += tokenize(t)
    src_vocab = Vocab(src_tokens)
    tgt_vocab = Vocab(tgt_tokens)

    # 数据加载
    dataset = TranslationDataset(SRC_FILE, TGT_FILE, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # 模型和训练设置
    model = Transformer(len(src_vocab), len(tgt_vocab), embed_size=512,
                        num_heads=8, dim_ff=2048, num_layers=6,
                        dropout=0.1, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi['<pad>'])
    CLIP = 1

    # 训练
    for epoch in range(10):
        loss = train_epoch(model, dataloader, optimizer, criterion, CLIP)
        print(f"Epoch {epoch+1}, Loss: {loss:.3f}")

    # 测试示例
    for sent in ["go .", "i lost .", "he's calm ."]:
        print(sent, "=>", ' '.join(translate(model, sent)))

if __name__ == '__main__':
    main()

#注意，一般训练的数据要准备几千到几万为宜，我这里准备的数据就五行，所以训练出来的结果
#输出不佳，只能输出最常见的符号'。'