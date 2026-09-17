import argparse
import json
import os

import torch
from transformers import BertTokenizer

try:
    from .config import Config
    from .model import BertWithDropout
    from .dataset import split_text_into_words
except ImportError:
    from config import Config
    from model import BertWithDropout
    from dataset import split_text_into_words
    


# 读取已训练好的模型路径和标签表
def load_model_and_labels(config, model_path_override=None):
    save_dir = config.get_experiment_dir()
    label_path = os.path.join(save_dir, 'label_list.json')

    # 如果用户通过 --model_path 指定了模型，就优先检查它；否则检查默认的 best_model.pt
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


# 把输入文本转换成token ids，记录每个字符的标签来源token索引
# 与训练完全一致：先按BERT规则合并成词，再对每个词整体做wordpiece
def build_bert_inputs(text, tokenizer, max_seq_len=128):
    tokens = []
    word_infos = []  # [(该词包含的字符列表, 该词首个token在序列中的位置)]

    for word in split_text_into_words(text):
        sub_tokens = tokenizer.tokenize(word)
        if not sub_tokens:
            # 该字符会被tokenizer丢弃，用[UNK]占位，避免丢字造成标签错位
            sub_tokens = ['[UNK]']

        if len(tokens) + len(sub_tokens) > max_seq_len - 2:
            break

        first_token_index = len(tokens) + 1  # +1 跳过开头的[CLS]
        tokens.extend(sub_tokens)
        word_infos.append((list(word), first_token_index))

    # 展开成逐字符形式
    # 一个词若被切成多个子词，训练时只有首子词带标签，其余子词是-100、输出没有意义，
    # 所以词内所有字符都取首子词的预测，保证与训练一致，实体也不会被切碎
    chars = []
    char_first_token_index = []
    for word_chars, first_token_index in word_infos:
        for ch in word_chars:
            chars.append(ch)
            char_first_token_index.append(first_token_index)

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

    for char_index, first_token_index in enumerate(encoded['char_first_token_index']):
        pred_id = pred_ids[first_token_index]
        pred_tags.append(label_list[pred_id])

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
