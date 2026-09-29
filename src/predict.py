import argparse
import json
import os

import torch
from transformers import BertTokenizerFast

try:
    from .config import Config
    from .model import BertWithDropout
    from .dataset import BertInputEncoder
    from .metrics import EntityLevelEvaluator
except ImportError:
    from config import Config
    from model import BertWithDropout
    from dataset import BertInputEncoder
    from metrics import EntityLevelEvaluator


# 加载某个实验的权重与配套标签表，把文本编码后交给模型，输出逐字标签和实体
class NERPredictor:

    def __init__(self, config, model_path_override=None):
        self.config = config
        self.device = torch.device(config.device)

        self.model_path, self.label_path, self.label_list = self._resolve_model_and_labels(model_path_override)

        tokenizer = BertTokenizerFast.from_pretrained(config.model_name, local_files_only=True)
        self.encoder = BertInputEncoder(tokenizer, config.max_seq_len)
        self.model = self._build_model()

        print(f'权重: {self.model_path}')
        print(f'标签表: {self.label_path}（共 {len(self.label_list)} 个标签）')

    # 确定权重与标签表
    # 两者必须来自同一组实验：训练时label_list.json就写在best_model.pt旁边
    # 标签表一律优先取权重所在目录下那份，--model_path换权重时标签表会跟着换
    def _resolve_model_and_labels(self, model_path_override):
        save_dir = self.config.get_experiment_dir()
        model_path = model_path_override or os.path.join(save_dir, 'best_model.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'未找到模型权重: {model_path}。请先运行训练生成模型，再进行预测。'
            )

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

        return model_path, label_path, label_list

    # 加载权重，并用分类头的真实输出维度校验标签数量，数量不一致时给出提示
    def _build_model(self):
        state_dict = torch.load(self.model_path, map_location=self.device)

        classifier_weight = state_dict.get('classifier.weight')
        if classifier_weight is None:
            raise KeyError(
                f'权重文件 {self.model_path} 中没有 classifier.weight，'
                f'请确认这是本项目训练产出的 best_model.pt'
            )

        weight_num_labels = classifier_weight.shape[0]
        if weight_num_labels != len(self.label_list):
            raise ValueError(
                f'权重与标签表不匹配：{self.model_path} 的分类头输出维度是 {weight_num_labels}，'
                f'而 {self.label_path} 有 {len(self.label_list)} 个标签。\n'
                f'       请确认 --model_path 与 --config 指向同一组实验，'
                f'或把该实验的 label_list.json 放在权重同目录下。'
            )

        model = BertWithDropout(
            model_name=self.config.model_name,
            num_labels=len(self.label_list),
            dropout_rate=self.config.dropout_rate
        )
        model.to(self.device)
        model.load_state_dict(state_dict)
        return model

    # 识别返回逐字标签和识别出的实体
    # chars取的是实际参与推理的字符，避免原文含空格时与标签错位
    def predict(self, text):
        self.model.eval()
        encoded = self.encoder.encode_text(text)

        with torch.no_grad():
            logits = self.model(
                encoded['input_ids'].to(self.device),
                encoded['attention_mask'].to(self.device)
            ).logits

        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        # 一个字符若被切成多个子词，只有首子词在训练时带标签，所以预测也只取首子词
        pred_tags = []
        for token_index in encoded['word_to_first_token']:
            if token_index is None:
                # 该字符没有对应子词（空白符、或被最大长度截断），一律按O处理
                pred_tags.append('O')
                continue
            pred_tags.append(self.label_list[pred_ids[token_index]])

        entities = self.extract_entities(encoded['chars'], pred_tags)
        return encoded['chars'], pred_tags, entities

    # 把BIO标签序列组装成实体
    # 直接复用评测器的解析规则，避免同一套BIO规则出现两份实现
    @staticmethod
    def extract_entities(chars, pred_tags):
        # get_entities返回 (类型, 起始下标, 结束下标)，按起始位置排序后拼回原文
        spans = sorted(EntityLevelEvaluator.get_entities(pred_tags), key=lambda x: x[1])
        return [(entity_type, ''.join(chars[start:end + 1])) for entity_type, start, end in spans]


def main():
    parser = argparse.ArgumentParser(description='中文命名实体识别预测脚本')
    parser.add_argument('--config', type=str, default='configs/01_msra_bert_base.json', help='训练时使用的配置文件路径')
    parser.add_argument('--text', type=str, default=None, help='待识别的文本，若不传则从命令行交互输入')
    parser.add_argument('--model_path', type=str, default=None, help='可选：直接指定模型权重路径')
    args = parser.parse_args()

    config = Config(config_path=args.config)
    predictor = NERPredictor(config, args.model_path)

    if args.text is not None:
        text = args.text
    else:
        text = input('请输入待识别文本：').strip()

    if not text:
        print('输入为空，退出。')
        return

    chars, pred_tags, entities = predictor.predict(text)

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
