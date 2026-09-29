import os

import torch
from torch.utils.data import Dataset


# 原始token序列转变为BERT输入的统一编码器
# 训练(NERDataset)与推理(predict)共用同一个类，保证词边界和子词对齐方式完全一致
# word_ids()为None的位置是[CLS]/[SEP]等特殊token，不参与训练
# 同一个原始token被切成多个子词时，只有首个子词带标签，其余子词用-100忽略
class BertInputEncoder:

    def __init__(self, tokenizer, max_seq_len=128):
        if not getattr(tokenizer, 'is_fast', False):
            raise ValueError(
                'BertInputEncoder 依赖 fast tokenizer 的 word_ids()，请使用 BertTokenizerFast'
            )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # 词边界由调用方给出（训练时是BIO文件的逐token切分，推理时是逐字符）
    # 再用word_ids()把每个原始token对齐到它在子词序列中的位置
    # 返回 (input_ids, attention_mask, word_to_first_token)
    def encode_words(self, words):
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        )

        word_ids = encoding.word_ids()

        # word_to_first_token[i]表示第i个原始token的首个子词在序列中的位置
        # 被tokenizer完全丢弃的原始token（如空白符）没有对应位置，保持None
        word_to_first_token = [None] * len(words)
        for token_index, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_to_first_token[word_id] is None:
                word_to_first_token[word_id] = token_index

        return encoding['input_ids'], encoding['attention_mask'], word_to_first_token

    # 把整句文本编码成可直接喂给模型的张量，并保留逐字符的对齐关系
    def encode_text(self, text):
        chars = list(text)
        input_ids, attention_mask, word_to_first_token = self.encode_words(chars)

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
            'word_to_first_token': word_to_first_token,
            'chars': chars
        }


class NERDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_seq_len=128, label_list=None):
        # 编码逻辑交给编码器类，数据集只负责读取数据、维护标签映射
        self.encoder = BertInputEncoder(tokenizer, max_seq_len)

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
        words = self.sentences[index]
        tags = self.labels[index]

        # 每个原始token独立送入tokenizer，标签挂在它的首个子词上，其余子词用-100忽略，避免把一个实体重复计数
        input_ids, attention_mask, word_to_first_token = self.encoder.encode_words(words)

        tag_ids = [-100] * len(input_ids)
        for word_index, token_index in enumerate(word_to_first_token):
            if token_index is None:
                # 该原始token没有对应输入位置（被tokenizer丢弃或超出最大长度），只能跳过其标签
                continue
            tag_ids[token_index] = self.label2id[tags[word_index]]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': tag_ids
        }

    # 把一个batch里长度不一的样本补齐到batch内最大长度
    # 补齐位的attention_mask置0，标签置-100（评测时会被跳过）
    @staticmethod
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

    # 根据数据集名称返回训练、验证、测试文件路径，并生成对应的标签列表
    @staticmethod
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
