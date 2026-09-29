import argparse
import json
import os

import torch
from transformers import BertTokenizerFast

try:
    from .config import Config
    from .model import BertWithDropout
    from .dataset import encode_words
except ImportError:
    from config import Config
    from model import BertWithDropout
    from dataset import encode_words
    


# 读取已训练好的模型路径和标签表
def load_model_and_labels(config, model_path_override=None):
    save_dir = config.get_experiment_dir()
    label_path = os.path.join(save_dir, 'label_list.json')

    # 如果用户通过--model_path指定了模型，就优先检查它；否则检查默认的best_model.pt
    model_path = model_path_override or os.path.join(save_dir, 'best_model.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'未找到模型权重: {model_path}。请先运行训练生成模型，再进行预测。'
        )
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f'未找到标签映射文件: {label_path}。请先运行训练生成 label_list.json。'
        )

    with open(label_path, 'r', encoding='utf-8') as f:
        label_list = json.load(f)

    return save_dir, model_path, label_list


# 把输入文本转换成token ids，并用word_ids()记录每个字符对应的首个子词位置
# 与训练保持一致：同样由外部给定词边界，每个字符独立作为一个词送入tokenizer
def build_bert_inputs(text, tokenizer, max_seq_len=128):
    chars = list(text)

    input_ids, attention_mask, word_to_first_token = encode_words(
        tokenizer, chars, max_seq_len
    )

    pad_len = max_seq_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long),
        'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
        'word_to_first_token': word_to_first_token,
        'chars': chars
    }


# 将BIO标签序列重新组装成实体
# 与评测口径保持一致：B- 才能开启实体，I- 只能延续同类型实体
# 裸I-（前面是O或类型不同）按O处理，不能算成实体
def extract_entities(chars, pred_tags):
    entities = []
    current_type = None
    current_chars = []

    def flush():
        if current_type is not None and current_chars:
            entities.append((current_type, ''.join(current_chars)))

    for ch, tag in zip(chars, pred_tags):
        prefix, _, tag_type = tag.partition('-')

        if prefix == 'B':
            flush()
            current_type, current_chars = tag_type, [ch]
        elif prefix == 'I' and current_type == tag_type and current_chars:
            # 正常延续：和当前实体同类型
            current_chars.append(ch)
        else:
            # O，或非法的裸I-：都表示当前实体到此结束
            flush()
            current_type, current_chars = None, []

    flush()
    return entities


# 识别返回逐字标签和识别出的实体（同时返回实际参与推理的 chars，避免原文含空格时错位）
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

    # 一个字符若被切成多个子词，只有首子词在训练时带标签，所以预测也只取首子词
    for token_index in encoded['word_to_first_token']:
        if token_index is None:
            # 该字符没有对应子词（空白符、或被最大长度截断），一律按O处理
            pred_tags.append('O')
            continue
        pred_tags.append(label_list[pred_ids[token_index]])

    entities = extract_entities(encoded['chars'], pred_tags)
    return encoded['chars'], pred_tags, entities


def main():
    parser = argparse.ArgumentParser(description='中文命名实体识别预测脚本')
    parser.add_argument('--config', type=str, default='configs/01_msra_bert_base.json', help='训练时使用的配置文件路径')
    parser.add_argument('--text', type=str, default=None, help='待识别的文本，若不传则从命令行交互输入')
    parser.add_argument('--model_path', type=str, default=None, help='可选：直接指定模型权重路径')
    args = parser.parse_args()

    config = Config(config_path=args.config)
    device = torch.device(config.device)

    save_dir, model_path, label_list = load_model_and_labels(config, args.model_path)

    tokenizer = BertTokenizerFast.from_pretrained(config.model_name, local_files_only=True)
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

    chars, pred_tags, entities = predict_text(model, tokenizer, text, label_list, device, config.max_seq_len)

    print('\n输入文本：')
    print(text)
    print('\n逐字预测标签：')
    print(list(zip(chars, pred_tags)))
    print('\n识别结果：')
    if not entities:
        print('未识别到实体')
    else:
        for entity_type, entity_text in entities:
            print(f'{entity_type}: {entity_text}')


if __name__ == '__main__':
    main()
