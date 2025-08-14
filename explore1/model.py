import math
import torch
from torch import nn
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

class PositionalEncoding(nn.Module):
    """
    位置编码模块，可以添加到输入嵌入中
    """
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


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    """
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
    """
    位置逐点前馈网络
    """
    def __init__(self, dim_model, dim_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model)
        )

    def forward(self, X): 
        return self.net(X)


class AddNorm(nn.Module):
    """
    残差连接和层归一化
    """
    def __init__(self, dim_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, X, Y): 
        return self.norm(X + self.dropout(Y))


class TransformerEncoderBlock(nn.Module):
    """
    Transformer编码器块
    """
    def __init__(self, embed_size, num_heads, dim_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.addnorm1 = AddNorm(embed_size, dropout)
        self.ffn = PositionWiseFFN(embed_size, dim_ff)
        self.addnorm2 = AddNorm(embed_size, dropout)

    def forward(self, X, src_mask=None):
        out = self.attn(X, X, X, key_padding_mask=src_mask)
        Y = self.addnorm1(X, out)
        return self.addnorm2(Y, self.ffn(Y))


class TransformerDecoderBlock(nn.Module):
    """
    Transformer解码器块
    """
    def __init__(self, embed_size, num_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.addnorm1 = AddNorm(embed_size, dropout)
        self.enc_attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.addnorm2 = AddNorm(embed_size, dropout)
        self.ffn = PositionWiseFFN(embed_size, dim_ff)
        self.addnorm3 = AddNorm(embed_size, dropout)

    def forward(self, X, enc_out, src_mask=None, tgt_mask=None):
        out1 = self.self_attn(X, X, X, attn_mask=tgt_mask)
        Y1 = self.addnorm1(X, out1)
        out2 = self.enc_attn(Y1, enc_out, enc_out, key_padding_mask=src_mask)
        Y2 = self.addnorm2(Y1, out2)
        return self.addnorm3(Y2, self.ffn(Y2))


class TransformerEncoder(nn.Module):
    """
    Transformer编码器，可作为即插即用模块
    """
    def __init__(self, vocab_size, embed_size=512, num_heads=8, 
                 dim_ff=2048, num_layers=6, dropout=0.1, max_len=1000):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, dropout, max_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_size, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_size)

    def make_src_mask(self, src, pad_token_id=0):
        return (src == pad_token_id)

    def forward(self, src):
        """
        前向传播
        
        Args:
            src: 输入序列，形状为(batch_size, seq_len)
            
        Returns:
            编码后的序列表示，形状为(batch_size, seq_len, embed_size)
        """
        src_mask = self.make_src_mask(src)
        enc = self.pos_enc(self.embedding(src) * math.sqrt(self.embed_size))
        
        for layer in self.encoder_layers:
            enc = layer(enc, src_mask)
            
        return self.norm(enc)


class TransformerDecoder(nn.Module):
    """
    Transformer解码器，可作为即插即用模块
    """
    def __init__(self, vocab_size, embed_size=512, num_heads=8, 
                 dim_ff=2048, num_layers=6, dropout=0.1, max_len=1000):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, dropout, max_len)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_size, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.norm = nn.LayerNorm(embed_size)

    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        mask = torch.triu(torch.ones((seq_len, seq_len), device=tgt.device), diagonal=1).bool()
        return mask

    def forward(self, tgt, enc_out, src_mask=None):
        """
        前向传播
        
        Args:
            tgt: 目标序列，形状为(batch_size, tgt_seq_len)
            enc_out: 编码器输出，形状为(batch_size, src_seq_len, embed_size)
            src_mask: 源序列掩码，形状为(batch_size, src_seq_len)
            
        Returns:
            解码后的输出 logits，形状为(batch_size, tgt_seq_len, vocab_size)
        """
        tgt_mask = self.make_tgt_mask(tgt)
        dec = self.pos_enc(self.embedding(tgt) * math.sqrt(self.embed_size))
        
        for layer in self.decoder_layers:
            dec = layer(dec, enc_out, src_mask, tgt_mask)
            
        dec = self.norm(dec)
        return self.fc_out(dec)


class SequenceClassifierHead(nn.Module):
    """
    序列分类头，可用于将Transformer编码器的输出转换为分类结果
    """
    def __init__(self, embed_size, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, encoder_output):
        """
        前向传播
        
        Args:
            encoder_output: 编码器输出，形状为(batch_size, seq_len, embed_size)
            
        Returns:
            分类结果，形状为(batch_size, num_classes)
        """
        # 使用序列的第一个token（通常是<bos>）的表示进行分类
        # 或者使用全局平均池化
        pooled_output = encoder_output.mean(dim=1)  # 全局平均池化
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# 自定义 Inception 块，添加 BatchNorm
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层 + BN
        self.p1_1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1)
        )
        # 线路2，1x1卷积层后接3x3卷积层 + BN
        self.p2_1 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0])
        )
        self.p2_2 = nn.Sequential(
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c2[1])
        )
        # 线路3，1x1卷积层后接5x5卷积层 + BN
        self.p3_1 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0])
        )
        self.p3_2 = nn.Sequential(
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(c3[1])
        )
        # 线路4，3x3最大汇聚层后接1x1卷积层 + BN
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Sequential(
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4)
        )

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class InceptionTransformerEncoder(nn.Module):
    """
    结合Inception模块和Transformer的编码器
    """
    def __init__(self, vocab_size, embed_size=512, num_heads=8, 
                 dim_ff=2048, num_layers=6, dropout=0.1, max_len=1000,
                 inception_channels=3):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, dropout, max_len)
        
        # 添加Inception模块用于图像特征提取
        self.inception_conv = nn.Sequential(
            nn.Conv2d(inception_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception模块
        self.inception_block1 = Inception(64, 64, (96, 128), (16, 32), 32)
        self.inception_block2 = Inception(256, 128, (128, 192), (32, 96), 64)
        
        # 池化和降维
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.inception_fc = nn.Linear(480, embed_size)  # 480是Inception块输出的通道数
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_size, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_size)

    def make_src_mask(self, src, pad_token_id=0):
        return (src == pad_token_id)

    def forward(self, src):
        """
        前向传播
        
        Args:
            src: 输入可以是文本序列(batch_size, seq_len)或图像(batch_size, channels, height, width)
            
        Returns:
            编码后的序列表示，形状为(batch_size, seq_len, embed_size)
        """
        if src.dim() == 4:  # 图像输入 (batch_size, channels, height, width)
            # 使用Inception网络处理图像
            x = self.inception_conv(src)
            x = self.inception_block1(x)
            x = self.inception_block2(x)
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            enc = self.inception_fc(x).unsqueeze(1)  # 添加序列维度
            src_mask = None
        else:  # 文本输入 (batch_size, seq_len)
            src_mask = self.make_src_mask(src)
            enc = self.pos_enc(self.embedding(src) * math.sqrt(self.embed_size))
        
        for layer in self.encoder_layers:
            enc = layer(enc, src_mask)
            
        return self.norm(enc)


class InceptionTransformerDecoder(nn.Module):
    """
    结合Inception模块的Transformer解码器
    """
    def __init__(self, vocab_size, embed_size=512, num_heads=8, 
                 dim_ff=2048, num_layers=6, dropout=0.1, max_len=1000,
                 inception_channels=3):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, dropout, max_len)
        
        # 添加Inception模块用于图像特征提取
        self.inception_conv = nn.Sequential(
            nn.Conv2d(inception_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception模块
        self.inception_block1 = Inception(64, 64, (96, 128), (16, 32), 32)
        self.inception_block2 = Inception(256, 128, (128, 192), (32, 96), 64)
        
        # 池化和降维
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.inception_fc = nn.Linear(480, embed_size)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_size, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.norm = nn.LayerNorm(embed_size)

    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        mask = torch.triu(torch.ones((seq_len, seq_len), device=tgt.device), diagonal=1).bool()
        return mask

    def forward(self, tgt, enc_out, src_mask=None):
        """
        前向传播
        
        Args:
            tgt: 目标序列，形状为(batch_size, tgt_seq_len)
            enc_out: 编码器输出，形状为(batch_size, src_seq_len, embed_size)
            src_mask: 源序列掩码，形状为(batch_size, src_seq_len)
            
        Returns:
            解码后的输出 logits，形状为(batch_size, tgt_seq_len, vocab_size)
        """
        tgt_mask = self.make_tgt_mask(tgt)
        dec = self.pos_enc(self.embedding(tgt) * math.sqrt(self.embed_size))
        
        for layer in self.decoder_layers:
            dec = layer(dec, enc_out, src_mask, tgt_mask)
            
        dec = self.norm(dec)
        return self.fc_out(dec)
