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
# 权重与标签表必须来自同一组实验：训练时 label_list.json 就写在 best_model.pt 旁边，
# 标签表一律优先取"权重文件所在目录"下的那份，--model_path 换权重时标签表会跟着换，
# 不再固定读取 --config 对应目录，避免权重与标签表分属两组实验
def load_model_and_labels(config, model_path_override=None):
    save_dir = config.get_experiment_dir()
    default_model_path = os.path.join(save_dir, 'best_model.pt')

    # 如果用户通过--model_path指定了模型，就优先检查它；否则检查默认的best_model.pt
    model_path = model_path_override or default_model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'未找到模型权重: {model_path}。请先运行训练生成模型，再进行预测。'
        )

    # 标签表与权重视为一个整体：先找权重同目录下那份
    model_dir = os.path.dirname(os.path.abspath(model_path))
    label_path = os.path.join(model_dir, 'label_list.json')

    if not os.path.exists(label_path):
        # 权重被单独拷贝出来时才回退，此时无法保证标签表与权重匹配，显式警告
        label_path = os.path.join(save_dir, 'label_list.json')
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f'未找到标签映射文件: {label_path}。请先运行训练生成 label_list.json。'
            )
        print(f'[警告] 权重目录 {model_dir} 下没有 label_list.json，'
              f'已回退使用 {label_path}。\n'
              f'       两者若不属于同一组实验，标签顺序可能不同，预测结果会静默出错，请核对后再用。')

    with open(label_path, 'r', encoding='utf-8') as f:
        label_list = json.load(f)

    print(f'权重: {model_path}')
    print(f'标签表: {label_path}（共 {len(label_list)} 个标签）')

    return save_dir, model_path, label_list


# 把输入文本转换成token ids，并用word_ids()记录每个字符对应的首个子词位置
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

    # 读一次权重，并用分类头的真实输出维度校验标签数量
    state_dict = torch.load(model_path, map_location=device)
    classifier_weight = state_dict.get('classifier.weight')
    if classifier_weight is None:
        raise KeyError(
            f'权重文件 {model_path} 中没有 classifier.weight，'
            f'请确认这是本项目训练产出的 best_model.pt'
        )
    weight_num_labels = classifier_weight.shape[0]
    if weight_num_labels != len(label_list):
        raise ValueError(
            f'权重与标签表不匹配：{model_path} 的分类头输出维度是 {weight_num_labels}，'
            f'而当前使用的标签表有 {len(label_list)} 个标签。\n'
            f'       请确认 --model_path 与 --config 指向同一组实验，'
            f'或把该实验的 label_list.json 放在权重同目录下。'
        )

    model = BertWithDropout(
        model_name=config.model_name,
        num_labels=len(label_list),
        dropout_rate=config.dropout_rate
    )
    model.to(device)
    model.load_state_dict(state_dict)

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
