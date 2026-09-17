import os
import re

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


# BERT自身的BasicTokenizer规则：连续的英文字母/数字视为一个"词"，其余字符（汉字、标点）各自独立
_WORD_CHAR_RE = re.compile(r'[A-Za-z0-9０-９Ａ-Ｚａ-ｚ]')


# 把字符序列按上述规则合并成"词"序列
# 与整句调用tokenizer.tokenize()得到的切分一致
# 纯中文场景下等价于逐字切分（一个汉字就是一个词），因此不会改变纯中文数据的结果
def split_text_into_words(chars):
    words = []
    buf = []

    for char in chars:
        if _WORD_CHAR_RE.fullmatch(char):
            buf.append(char)
        else:
            if buf:
                words.append(''.join(buf))
                buf = []
            words.append(char)

    if buf:
        words.append(''.join(buf))

    return words


# 与split_text_into_words完全相同的切分规则，但同时返回每个"词"由几个原始 token 组成
# tags是按token给出的，而weibo数据里存在emoji、多字符token，它们只对应一个标签，但len可能大于1，
# 若按字符长度推进pos会导致tags[pos]越界或标签错位
def split_text_into_words_with_counts(chars):
    words = []  # [(word_str, token_count), ...]
    buf = []

    for char in chars:
        if _WORD_CHAR_RE.fullmatch(char):
            buf.append(char)
        else:
            if buf:
                words.append((''.join(buf), len(buf)))
                buf = []
            # 非英文数字的字符（含emoji等多字符token）整体作为一个词，只占1个token
            words.append((char, 1))

    if buf:
        words.append((''.join(buf), len(buf)))

    return words


def collate_fn(batch):
    if not batch:
        return {
            'input_ids': torch.empty((0, 0), dtype=torch.long),
            'attention_mask': torch.empty((0, 0), dtype=torch.long),
            'labels': torch.empty((0, 0), dtype=torch.long),
        }

    max_len = max(len(item['input_ids']) for item in batch)
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids_list.append(item['input_ids'] + [0] * pad_len)
        attention_mask_list.append(item['attention_mask'] + [0] * pad_len)
        labels_list.append(item['labels'] + [-100] * pad_len)

    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
        'labels': torch.tensor(labels_list, dtype=torch.long),
    }


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

        # 先按BERT的规则合并成"词"，再对每个词整体做wordpiece
        # 一个词可能被切成多个子词，
        # 标签只挂在第一个子词上，其余子词用-100忽略，避免把一个实体重复计数
        words = split_text_into_words_with_counts(chars)
        pos = 0
        for word, n_tokens in words:
            tag = tags[pos]  # 词的标签取它第一个token的标签
            pos += n_tokens  # 按token数推进，而不是字符长度

            sub_tokens = self.tokenizer.tokenize(word)
            if not sub_tokens:
                # 极少数字符可能被tokenizer丢弃，用[UNK]占位，避免丢字造成标签错位
                sub_tokens = ['[UNK]']

            tokens.extend(sub_tokens)
            tag_ids.append(self.label2id[tag])
            tag_ids.extend([-100] * (len(sub_tokens) - 1))

        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
            tag_ids = tag_ids[:self.max_seq_len - 2]

        # BERT要求前后加[CLS]和[SEP]，并把无效位置用-100忽略
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tag_ids = [-100] + tag_ids + [-100]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': tag_ids
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
