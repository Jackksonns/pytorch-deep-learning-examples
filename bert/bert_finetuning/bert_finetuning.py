# import json
# import multiprocessing
# import os
# import torch
# from torch import nn
# from d2l import torch as d2l

# d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
#                              '225d66f04cae318b841a13d32af3acc165f253ac')
# d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
#                               'c72329e68a732bef0452e4b96a1c341c8910f81f')

# def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
#                           num_heads, num_layers, dropout, max_len, devices):
#     data_dir = d2l.download_extract(pretrained_model)
#     # 定义空词表以加载预定义词表
#     vocab = d2l.Vocab()
#     vocab.idx_to_token = json.load(open(os.path.join(data_dir,
#         'vocab.json')))
#     vocab.token_to_idx = {token: idx for idx, token in enumerate(
#         vocab.idx_to_token)}
#     bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
#                          ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
#                          num_heads=4, num_layers=2, dropout=0.2,
#                          max_len=max_len, key_size=256, query_size=256,
#                          value_size=256, hid_in_features=256,
#                          mlm_in_features=256, nsp_in_features=256)
#     # 加载预训练BERT参数
#     bert.load_state_dict(torch.load(os.path.join(data_dir,
#                                                  'pretrained.params')))
#     return bert, vocab

# devices = d2l.try_all_gpus()
# bert, vocab = load_pretrained_model(
#     'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
#     num_layers=2, dropout=0.1, max_len=512, devices=devices)

# class SNLIBERTDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, max_len, vocab=None):
#         all_premise_hypothesis_tokens = [[
#             p_tokens, h_tokens] for p_tokens, h_tokens in zip(
#             *[d2l.tokenize([s.lower() for s in sentences])
#               for sentences in dataset[:2]])]

#         self.labels = torch.tensor(dataset[2])
#         self.vocab = vocab
#         self.max_len = max_len
#         (self.all_token_ids, self.all_segments,
#          self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
#         print('read ' + str(len(self.all_token_ids)) + ' examples')

#     def _preprocess(self, all_premise_hypothesis_tokens):
#         pool = multiprocessing.Pool(4)  # 使用4个进程
#         out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
#         all_token_ids = [
#             token_ids for token_ids, segments, valid_len in out]
#         all_segments = [segments for token_ids, segments, valid_len in out]
#         valid_lens = [valid_len for token_ids, segments, valid_len in out]
#         return (torch.tensor(all_token_ids, dtype=torch.long),
#                 torch.tensor(all_segments, dtype=torch.long),
#                 torch.tensor(valid_lens))

#     def _mp_worker(self, premise_hypothesis_tokens):
#         p_tokens, h_tokens = premise_hypothesis_tokens
#         self._truncate_pair_of_tokens(p_tokens, h_tokens)
#         tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
#         token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
#                              * (self.max_len - len(tokens))
#         segments = segments + [0] * (self.max_len - len(segments))
#         valid_len = len(tokens)
#         return token_ids, segments, valid_len

#     def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
#         # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
#         while len(p_tokens) + len(h_tokens) > self.max_len - 3:
#             if len(p_tokens) > len(h_tokens):
#                 p_tokens.pop()
#             else:
#                 h_tokens.pop()

#     def __getitem__(self, idx):
#         return (self.all_token_ids[idx], self.all_segments[idx],
#                 self.valid_lens[idx]), self.labels[idx]

#     def __len__(self):
#         return len(self.all_token_ids)
    
# # 如果出现显存不足错误，请减少“batch_size”。在原始的BERT模型中，max_len=512
# batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
# data_dir = d2l.download_extract('SNLI')
# train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
# test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
# train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
#                                    num_workers=num_workers)
# test_iter = torch.utils.data.DataLoader(test_set, batch_size,
#                                   num_workers=num_workers)

# class BERTClassifier(nn.Module):
#     def __init__(self, bert):
#         super(BERTClassifier, self).__init__()
#         self.encoder = bert.encoder
#         self.hidden = bert.hidden
#         self.output = nn.Linear(256, 3)

#     def forward(self, inputs):
#         tokens_X, segments_X, valid_lens_x = inputs
#         encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
#         return self.output(self.hidden(encoded_X[:, 0, :]))
    
# net = BERTClassifier(bert)

# lr, num_epochs = 1e-4, 5
# trainer = torch.optim.Adam(net.parameters(), lr=lr)
# loss = nn.CrossEntropyLoss(reduction='none')
# d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
#     devices)

import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW

from datasets import load_dataset

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SNLIDataset(Dataset):
    def __init__(self, split, tokenizer, max_len=128):
        # Load SNLI from HuggingFace datasets
        data = load_dataset('snli', split=split)
        # Filter out examples without label
        data = data.filter(lambda x: x['label'] != -1)
        self.sentences1 = data['premise']
        self.sentences2 = data['hypothesis']
        self.labels = data['label']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        s1 = str(self.sentences1[idx])
        s2 = str(self.sentences2[idx])
        # Tokenize pair
        encoding = self.tokenizer(
            s1,
            s2,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # cls token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

if __name__ == '__main__':
    # Settings
    model_name = 'bert-base-uncased'
    batch_size = 16  #本来是32
    max_len = 64  #本来是128，我这里在cpu上所以改了
    lr = 2e-5
    epochs = 1  #本来是3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # train_dataset = SNLIDataset('train', tokenizer, max_len) 这是原本的
    train_dataset = SNLIDataset('train[:100]', tokenizer, max_len)
    val_dataset = SNLIDataset('validation', tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = BERTClassifier(model_name=model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} '  \
              f'| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save the fine-tuned model
    output_dir = './bert_snli_finetuned'
    os.makedirs(output_dir, exist_ok=True)
    model.bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')
