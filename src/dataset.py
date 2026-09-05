import os

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class NERDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_seq_len=128, label_list=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 先把文本和标签按句子读取进来，然后再建立label到id的映射
        self.sentences, self.labels = self._read_bio_file(data_path)

        if label_list is None:
            all_tags = set()
            for tag_seq in self.labels:
                all_tags.update(tag_seq)
            self.label_list = sorted(list(all_tags))
        else:
            self.label_list = label_list

        self.label2id = {tag: idx for idx, tag in enumerate(self.label_list)}
        self.id2label = {idx: tag for idx, tag in enumerate(self.label_list)}
        self.num_labels = len(self.label_list)

    # 读取BIO文件：每一行是"字符 标签"，空行表示一句结束
    def _read_bio_file(self, file_path):
        sentences = []
        labels = []
        curr_sent = []
        curr_label = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if curr_sent:
                        sentences.append(curr_sent)
                        labels.append(curr_label)
                        curr_sent = []
                        curr_label = []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    char = parts[0]
                    tag = parts[-1]
                    curr_sent.append(char)
                    curr_label.append(tag)

            if curr_sent:
                sentences.append(curr_sent)
                labels.append(curr_label)

        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        chars = self.sentences[index]
        tags = self.labels[index]

        tokens = []
        tag_ids = []

        # 一个汉字可能拆成多个BERT子词，所以要把标签也按对应位置补齐
        for char, tag in zip(chars, tags):
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:
                continue

            tokens.extend(sub_tokens)
            tag_ids.append(self.label2id[tag])
            for _ in range(len(sub_tokens) - 1):
                tag_ids.append(-100)

        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
            tag_ids = tag_ids[:self.max_seq_len - 2]

        # BERT要求前后加[CLS]和[SEP]，并把无效位置用-100忽略
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tag_ids = [-100] + tag_ids + [-100]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_seq_len - len(input_ids)
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len
        tag_ids += [-100] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(tag_ids, dtype=torch.long)
        }


# 根据数据集名称返回训练、验证、测试文件路径，并生成对应的标签列表
def get_dataset_paths(dataset_name, data_dir='./data'):
    if dataset_name == 'MSRA':
        paths = {
            'train': os.path.join(data_dir, 'MSRA', 'train_5k.txt'),
            'dev': os.path.join(data_dir, 'MSRA', 'dev_1k.txt'),
            'test': os.path.join(data_dir, 'MSRA', 'test_1k.txt')
        }
        label_file = None
    elif dataset_name == 'weibo':
        paths = {
            'train': os.path.join(data_dir, 'weibo', 'train.txt'),
            'dev': os.path.join(data_dir, 'weibo', 'dev.txt'),
            'test': os.path.join(data_dir, 'weibo', 'test.txt')
        }
        label_file = os.path.join(data_dir, 'weibo', 'class.txt')
    else:
        raise ValueError(f'不支持的数据集: {dataset_name}')

    label_list = None
    if label_file and os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            types = [line.strip() for line in f if line.strip()]
        label_list = ['O'] + [f'B-{t}' for t in types] + [f'I-{t}' for t in types]

    return paths, label_list
