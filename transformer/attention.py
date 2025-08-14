import torch
from torch import nn
import math
import matplotlib.pyplot as plt

# 1. Multi-Head Attention（简化版）
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=num_hiddens, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, queries, keys, values, valid_lens=None):
        # valid_lens: [batch_size] or [batch_size, seq_len]
        if valid_lens is not None:
            # 构造 attention mask
            # mask 为 True 表示该位置为“无效”
            mask = torch.arange(keys.size(1), device=keys.device)[None, :] >= valid_lens[:, None]
        else:
            mask = None
        output, _ = self.attention(queries, keys, values, key_padding_mask=mask)
        return output

# 2. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        P = torch.zeros((1, max_len, num_hiddens))
        position = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.pow(10000, torch.arange(0, num_hiddens, 2).float() / num_hiddens)
        P[0, :, 0::2] = torch.sin(position / div_term)
        P[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer('P', P)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)

# 测试 MultiHeadAttention
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, dropout=0.5)
attention.eval()

batch_size, num_queries = 2, 4
valid_lens = torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))  # [B, T, D]
out = attention(X, X, X, valid_lens)
print("Multi-Head Attention Output Shape:", out.shape)

# 测试 PositionalEncoding + 可视化
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :num_steps, :]

# 绘图
plt.figure(figsize=(6, 2.5))
for i in range(6, 10):
    plt.plot(torch.arange(num_steps), P[0, :, i].detach().numpy(), label=f"dim {i}")
plt.xlabel("Position")
plt.title("Positional Encoding")
plt.legend()
plt.tight_layout()
plt.show()
