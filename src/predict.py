import argparse
import json
import os

import torch
from transformers import BertTokenizer

try:
    from .config import Config
    from .model import BertWithDropout
except ImportError:
    from config import Config
    from model import BertWithDropout


# 读取已训练好的模型路径和标签表
def load_model_and_labels(config):
    save_dir = os.path.join(config.save_dir, f'{os.path.basename(config.model_name)}_{config.dataset}')
    label_path = os.path.join(save_dir, 'label_list.json')
    model_path = os.path.join(save_dir, 'best_model.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'未找到最佳模型权重: {model_path}。请先运行训练生成模型，再进行预测。'
        )
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f'未找到标签映射文件: {label_path}。请先运行训练生成 label_list.json。'
        )

    with open(label_path, 'r', encoding='utf-8') as f:
        label_list = json.load(f)

    return save_dir, model_path, label_list


# 把输入文本转换成token ids，记录每个字符对应的第一个token索引
def build_bert_inputs(text, tokenizer, max_seq_len=128):
    tokens = []
    char_first_token_index = []
    chars = []

    for ch in text:
        sub_tokens = tokenizer.tokenize(ch)
        if not sub_tokens:
            # 该字符（如空格）会被tokenizer丢弃，同步跳过，避免标签错位
            continue

        if len(tokens) + len(sub_tokens) > max_seq_len - 2:
            break

        start = len(tokens)
        tokens.extend(sub_tokens)
        char_first_token_index.append(start + 1)
        chars.append(ch)

    input_tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1] * len(input_ids)

    pad_len = max_seq_len - len(input_ids)
    if pad_len > 0:
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long),
        'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
        'char_first_token_index': char_first_token_index,
        'chars': chars
    }


# 将BIO标签序列重新组装成实体
def extract_entities(chars, pred_tags):
    entities = []
    current_type = None
    current_chars = []

    def flush():
        if current_type is not None and current_chars:
            entities.append((current_type, ''.join(current_chars)))

    for ch, tag in zip(chars, pred_tags):
        if tag == 'O':
            flush()
            current_type = None
            current_chars = []
        elif tag.startswith('B-'):
            flush()
            current_type = tag[2:]
            current_chars = [ch]
        elif tag.startswith('I-'):
            if current_type == tag[2:]:
                current_chars.append(ch)
            else:
                flush()
                current_type = tag[2:]
                current_chars = [ch]
        else:
            flush()
            current_type = None
            current_chars = []

    flush()
    return entities


# 识别返回逐字标签和识别出的实体
def predict_text(model, tokenizer, text, label_list, device, max_seq_len):
    model.eval()
    encoded = build_bert_inputs(text, tokenizer, max_seq_len=max_seq_len)

    with torch.no_grad():
        logits = model(
            encoded['input_ids'].to(device),
            encoded['attention_mask'].to(device)
        ).logits

    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    pred_tags = []

    for char_index, first_token_index in enumerate(encoded['char_first_token_index']):
        pred_id = pred_ids[first_token_index]
        pred_tags.append(label_list[pred_id])

    entities = extract_entities(encoded['chars'], pred_tags)
    return pred_tags, entities


def main():
    parser = argparse.ArgumentParser(description='中文命名实体识别预测脚本')
    parser.add_argument('--config', type=str, default='configs/01_msra_bert_base.json', help='训练时使用的配置文件路径')
    parser.add_argument('--text', type=str, default=None, help='待识别的文本，若不传则从命令行交互输入')
    parser.add_argument('--model_path', type=str, default=None, help='可选：直接指定模型权重路径')
    args = parser.parse_args()

    config = Config(config_path=args.config)
    device = torch.device(config.device)

    save_dir, model_path, label_list = load_model_and_labels(config)
    if args.model_path:
        model_path = args.model_path

    tokenizer = BertTokenizer.from_pretrained(config.model_name, local_files_only=True)
    model = BertWithDropout(
        model_name=config.model_name,
        num_labels=len(label_list),
        dropout_rate=config.dropout_rate
    )
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if args.text is not None:
        text = args.text
    else:
        text = input('请输入待识别文本：').strip()

    if not text:
        print('输入为空，退出。')
        return

    pred_tags, entities = predict_text(model, tokenizer, text, label_list, device, config.max_seq_len)

    print('\n输入文本：')
    print(text)
    print('\n逐字预测标签：')
    print(list(zip(list(text), pred_tags)))
    print('\n识别结果：')
    if not entities:
        print('未识别到实体')
    else:
        for entity_type, entity_text in entities:
            print(f'{entity_type}: {entity_text}')


if __name__ == '__main__':
    main()
