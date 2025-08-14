import torch
import torch.nn as nn
from model import InceptionTransformerEncoder, InceptionTransformerDecoder

def image_captioning_example():
    """
    图像描述生成任务示例
    编码器处理图像，解码器生成描述文本
    """
    print("=== 图像描述生成任务 ===")
    
    # 模型参数
    vocab_size = 10000  # 词汇表大小
    embed_size = 512    # 嵌入维度
    num_heads = 8       # 注意力头数
    dim_ff = 2048       # 前馈网络维度
    num_layers = 6      # 层数
    dropout = 0.1       # dropout率
    max_len = 1000      # 最大序列长度
    inception_channels = 3  # 图像通道数(RGB)
    
    # 创建编码器(处理图像)和解码器(生成文本)
    encoder = InceptionTransformerEncoder(
        vocab_size, embed_size, num_heads, dim_ff,
        num_layers, dropout, max_len, inception_channels
    )
    
    decoder = InceptionTransformerDecoder(
        vocab_size, embed_size, num_heads, dim_ff,
        num_layers, dropout, max_len, inception_channels
    )
    
    # 模拟输入
    batch_size = 4
    # 图像输入 (batch_size, channels, height, width)
    images = torch.randn(batch_size, 3, 224, 224)
    # 文本目标序列 (batch_size, sequence_length)
    captions = torch.randint(0, vocab_size, (batch_size, 20))
    
    print(f"输入图像形状: {images.shape}")
    print(f"目标文本序列形状: {captions.shape}")
    
    # 编码器处理图像
    encoder_output = encoder(images)
    print(f"编码器输出形状: {encoder_output.shape}")
    
    # 解码器生成文本描述
    # 解码器输入通常是目标序列的移位版本(去掉最后一个词)
    decoder_input = captions[:, :-1]
    decoder_output = decoder(decoder_input, encoder_output)
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"解码器输出形状: {decoder_output.shape}")


def machine_translation_example():
    """
    机器翻译任务示例
    编码器处理源语言文本，解码器生成目标语言文本
    """
    print("\n=== 机器翻译任务 ===")
    
    # 模型参数
    src_vocab_size = 8000   # 源语言词汇表大小
    tgt_vocab_size = 10000  # 目标语言词汇表大小
    embed_size = 512
    num_heads = 8
    dim_ff = 2048
    num_layers = 6
    dropout = 0.1
    max_len = 1000
    inception_channels = 3
    
    # 为源语言和目标语言创建不同的编码器和解码器
    # 注意：这里我们简化处理，实际上应该为源语言文本创建不包含Inception模块的编码器
    encoder = InceptionTransformerEncoder(
        src_vocab_size, embed_size, num_heads, dim_ff,
        num_layers, dropout, max_len, inception_channels
    )
    
    decoder = InceptionTransformerDecoder(
        tgt_vocab_size, embed_size, num_heads, dim_ff,
        num_layers, dropout, max_len, inception_channels
    )
    
    # 模拟输入
    batch_size = 4
    src_seq_len = 15  # 源序列长度
    tgt_seq_len = 18  # 目标序列长度
    
    # 源语言文本 (batch_size, sequence_length)
    src_sentences = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    # 目标语言文本 (batch_size, sequence_length)
    tgt_sentences = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"源语言句子形状: {src_sentences.shape}")
    print(f"目标语言句子形状: {tgt_sentences.shape}")
    
    # 编码器处理源语言
    encoder_output = encoder(src_sentences)
    print(f"编码器输出形状: {encoder_output.shape}")
    
    # 解码器生成目标语言
    decoder_input = tgt_sentences[:, :-1]  # 移位目标序列
    decoder_output = decoder(decoder_input, encoder_output)
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"解码器输出形状: {decoder_output.shape}")


def question_answering_example():
    """
    基于图像的问答任务示例
    编码器处理图像和问题，解码器生成答案
    """
    print("\n=== 基于图像的问答任务 ===")
    
    # 模型参数
    vocab_size = 5000
    embed_size = 512
    num_heads = 8
    dim_ff = 2048
    num_layers = 6
    dropout = 0.1
    max_len = 1000
    inception_channels = 3
    
    # 编码器处理图像和问题，解码器生成答案
    encoder = InceptionTransformerEncoder(
        vocab_size, embed_size, num_heads, dim_ff,
        num_layers, dropout, max_len, inception_channels
    )
    
    decoder = InceptionTransformerDecoder(
        vocab_size, embed_size, num_heads, dim_ff,
        num_layers, dropout, max_len, inception_channels
    )
    
    # 模拟输入
    batch_size = 4
    # 图像输入
    images = torch.randn(batch_size, 3, 224, 224)
    # 问题文本
    questions = torch.randint(0, vocab_size, (batch_size, 10))
    # 答案文本
    answers = torch.randint(0, vocab_size, (batch_size, 15))
    
    print(f"输入图像形状: {images.shape}")
    print(f"问题文本形状: {questions.shape}")
    print(f"答案文本形状: {answers.shape}")
    
    # 对于这种任务，我们需要修改编码器以同时处理图像和文本
    # 这里我们简化处理，只使用图像作为输入
    encoder_output = encoder(images)
    print(f"编码器输出形状: {encoder_output.shape}")
    
    # 解码器生成答案
    decoder_input = answers[:, :-1]
    decoder_output = decoder(decoder_input, encoder_output)
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"解码器输出形状: {decoder_output.shape}")


def correct_model_usage():
    """
    展示如何正确使用编码器-解码器架构
    """
    print("\n=== 正确的模型使用方式 ===")
    
    class EncoderDecoderModel(nn.Module):
        """
        正确的编码器-解码器模型封装
        """
        def __init__(self, vocab_size, embed_size, num_heads, dim_ff,
                     num_layers, dropout, max_len, inception_channels):
            super().__init__()
            self.encoder = InceptionTransformerEncoder(
                vocab_size, embed_size, num_heads, dim_ff,
                num_layers, dropout, max_len, inception_channels
            )
            self.decoder = InceptionTransformerDecoder(
                vocab_size, embed_size, num_heads, dim_ff,
                num_layers, dropout, max_len, inception_channels
            )
        
        def forward(self, src, tgt):
            """
            前向传播
            
            Args:
                src: 源输入(图像或文本)
                tgt: 目标序列
                
            Returns:
                解码器输出
            """
            encoder_output = self.encoder(src)
            decoder_output = self.decoder(tgt, encoder_output)
            return decoder_output
    
    # 创建模型实例
    model = EncoderDecoderModel(
        vocab_size=1000,
        embed_size=512,
        num_heads=8,
        dim_ff=2048,
        num_layers=6,
        dropout=0.1,
        max_len=1000,
        inception_channels=3
    )
    
    # 示例输入
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, 1000, (batch_size, 20))
    
    # 模型推理
    encoder_output = model.encoder(images)
    decoder_input = captions[:, :-1]
    output = model(images, decoder_input)
    
    print(f"模型输入图像形状: {images.shape}")
    print(f"模型解码器输入形状: {decoder_input.shape}")
    print(f"模型输出形状: {output.shape}")


if __name__ == "__main__":
    image_captioning_example()
    machine_translation_example()
    question_answering_example()
    correct_model_usage()