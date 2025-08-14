import os
import random
import json

def make_sentence(vocab, min_len=5, max_len=15):
    """从 vocab 中随机抽词，拼成一句话（用空格分隔），末尾不带标点。"""
    length = random.randint(min_len, max_len)
    return " ".join(random.choices(vocab, k=length))

def generate_fake_json(path, num_paragraphs=1000, sent_per_para=(2,5)):
    """
    每行一个“段落”，段落内有随机数目的句子，用 ' . ' 分隔。
    最终写入 path。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 取英语常用词表，或自定义一小段词表
    vocab = ["the", "a", "and", "to", "of", "I", "you", "he", "she", "it",
             "is", "was", "in", "on", "for", "with", "that", "this", "we",
             "they", "run", "walk", "talk", "play", "read", "write", "code"]
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(num_paragraphs):
            n_sents = random.randint(*sent_per_para)
            sents = [make_sentence(vocab) for _ in range(n_sents)]
            # 用 “ . ” 隔开，和你的 read_wiki 逻辑保持一致
            line = " . ".join(sents)
            f.write(line + "\n")
    print(f"已生成 {num_paragraphs} 行伪数据到 {path}")

if __name__ == "__main__":
    generate_fake_json("./sohu_data/sohu_data.json",
                       num_paragraphs=2000,
                       sent_per_para=(2, 6))
