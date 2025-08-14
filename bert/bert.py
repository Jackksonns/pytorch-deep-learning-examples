import torch
from torch import nn
import math

# Multi-Head Attention & Encoder Block
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout=0.0, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=num_hiddens, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        attn_mask = None
        if valid_lens is not None:
            attn_mask = torch.zeros_like(queries[:, :, 0]) == 1
            for i, length in enumerate(valid_lens):
                attn_mask[i, length:] = True

        output, _ = self.attention(queries, keys, values, key_padding_mask=attn_mask)
        return self.W_o(output)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(ffn_num_input, ffn_num_hiddens),
            nn.ReLU(),
            nn.Linear(ffn_num_hiddens, ffn_num_input)
        )

    def forward(self, X):
        return self.ffn(X)

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens=None):
        Y = self.attention(X, X, X, valid_lens)
        X = self.addnorm1(X, Y)
        Z = self.ffn(X)
        return self.addnorm2(X, Z)

# BERT Encoder
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        #创建了一个可学习的位置嵌入矩阵，形状为 (1, max_len, num_hiddens)。
        #1: batch 维度（占位，不是必须的）
        #max_len: 序列最大长度（如512），代表最多能编码这么多位置
        #num_hiddens: 每个位置的向量维度，等于 embedding dim，例如 768
        self.blks = nn.Sequential(*[
            EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                         ffn_num_input, ffn_num_hiddens, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tokens, segments, valid_lens=None):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

# Masked Language Model
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1] #获取总共需要预测的 token 数
        pred_positions = pred_positions.reshape(-1) # 展平 pred_positions 为一维向量
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size).repeat_interleave(num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        return self.mlp(masked_X) #最后送入 MLP 分类器

# Next Sentence Prediction
class NextSentencePred(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)

# BERT Model
class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads,
                                   num_layers, dropout, max_len,
                                   key_size, query_size, value_size)
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens),
            nn.Tanh()
        )
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        mlm_Y_hat = self.mlm(encoded_X, pred_positions) if pred_positions is not None else None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

# Helper Function
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

# Testing Usage Example
if __name__ == '__main__':
    # 参数定义
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2

    # 初始化 BERT 模型
    model = BERTModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)

    # 构造模拟输入数据（两个样本，每个句子长度为8）
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0,0,0,0,1,1,1,1],[0,0,0,1,1,1,1,1]])
    pred_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])  # 预测掩码位置
    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])  # MLM 标签.表示两个样本中被 mask 的 token 的真实词汇索引;第一个样本中，mask 了 [第1位, 第5位, 第2位]，真实词是 vocab 中编号为 [7, 8, 9]
    nsp_Y = torch.tensor([0, 1])  # NSP 标签

    # 模型前向传播
    encoded_X, mlm_Y_hat, nsp_Y_hat = model(tokens, segments, pred_positions=pred_positions)

    # 计算 Masked Language Model 的损失
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print("MLM loss shape:", mlm_l.shape)

    # 计算 Next Sentence Prediction 的损失
    nsp_l = loss(nsp_Y_hat, nsp_Y)
    print("NSP loss shape:", nsp_l.shape)

    # 总损失（可加权平均）
    total_loss = (mlm_l.mean() + nsp_l.mean()) / 2
    print("Total loss:", total_loss.item())
