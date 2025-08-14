import torch
import torch.nn as nn
import torch.nn.functional as F

#这段代码通过多个注意力头并行处理输入序列的信息，捕捉不同子空间的特征关系，最后融合成一个更丰富的表示，返回与输入 queries 同样形状但语义更丰富的输出。
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries: (B, q_len, d)
        # keys: (B, k_len, d)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / d**0.5  # (B, q_len, k_len)

        if valid_lens is not None:
            # valid_lens: (B,) or (B, q_len)
            shape = scores.shape
            if valid_lens.dim() == 1:
                valid_lens = valid_lens.repeat_interleave(shape[1], dim=0)
            else:
                valid_lens = valid_lens.reshape(-1)

            # 用 -1e6 来 mask 超出有效长度的 key
            mask = torch.arange(shape[2], device=scores.device)[None, :] < valid_lens[:, None]
            scores = scores.reshape(-1, shape[2])
            scores[~mask] = -1e6
            scores = scores.reshape(shape)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.bmm(attention_weights, values)


def transpose_qkv(X, num_heads):
    """为了多头注意力的并行计算而变换形状"""
    # X: (batch_size, seq_len, num_hiddens)
    batch_size, seq_len, num_hiddens = X.shape
    X = X.reshape(batch_size, seq_len, num_heads, -1)
    X = X.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
    return X.reshape(-1, seq_len, X.shape[-1])  # (batch_size*num_heads, seq_len, head_dim)


def transpose_output(X, num_heads):
    """将多头的输出还原回原始形状"""
    batch_head, seq_len, head_dim = X.shape
    batch_size = batch_head // num_heads
    X = X.reshape(batch_size, num_heads, seq_len, head_dim)
    X = X.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
    return X.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, num_hiddens)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout=0.0, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, dropout=0.5)
attention.eval()

batch_size, num_queries = 2, 4
num_kvpairs = 6
valid_lens = torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X, Y, Y, valid_lens).shape)  # 应输出: torch.Size([2, 4, 100])
