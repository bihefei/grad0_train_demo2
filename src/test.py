import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

try:
    from .config import Config
    from .dataset import NERDataset, get_dataset_paths, collate_fn
    from .model import BertWithDropout
    from .metrics import compute_entity_level_metrics
except ImportError:
    from config import Config
    from dataset import NERDataset, get_dataset_paths, collate_fn
    from model import BertWithDropout
    from metrics import compute_entity_level_metrics


# 测试器，负责读取最佳模型并在测试集上做最终评估
class Tester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

    # 读取训练时保存的标签映射文件
    def _load_label_list(self, save_dir):
        label_path = os.path.join(save_dir, 'label_list.json')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f'未找到标签映射文件: {label_path}，请先执行训练以生成最佳模型和标签文件。')
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 评估函数，把模型输出的label id转成实体标签，统计精确率、召回率和 F1
    def _evaluate(self, model, data_loader, label_list):
        model.eval()
        all_pred_tags = []
        all_true_tags = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids, attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                batch_size = preds.shape[0]
                for i in range(batch_size):
                    pred_seq = []
                    true_seq = []
                    seq_len = preds.shape[1]

                    for j in range(seq_len):
                        true_label_id = labels[i][j].item()
                        if true_label_id == -100:
                            continue
                        pred_seq.append(label_list[preds[i][j].item()])
                        true_seq.append(label_list[true_label_id])

                    all_pred_tags.append(pred_seq)
                    all_true_tags.append(true_seq)

        return compute_entity_level_metrics(all_true_tags, all_pred_tags)

    # 测试入口，加载权重和标签表，并在测试集上验证模型效果
    def test(self, checkpoint_path=None):
        save_dir = os.path.join(self.config.save_dir, f'{os.path.basename(self.config.model_name)}_{self.config.dataset}')
        model_path = checkpoint_path or os.path.join(save_dir, 'best_model.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到测试用模型权重: {model_path}，请先执行训练生成最佳模型。')

        label_list = self._load_label_list(save_dir)
        paths, _ = get_dataset_paths(self.config.dataset, self.config.data_dir)
        tokenizer = BertTokenizer.from_pretrained(self.config.model_name, local_files_only=True)

        test_set = NERDataset(
            data_path=paths['test'],
            tokenizer=tokenizer,
            max_seq_len=self.config.max_seq_len,
            label_list=label_list
        )
        test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)

        model = BertWithDropout(
            model_name=self.config.model_name,
            num_labels=len(label_list),
            dropout_rate=self.config.dropout_rate
        )
        model.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        result = self._evaluate(model, test_loader, label_list)

        print('\n===== 测试集最终评估 =====')
        print(f'精确率: {result["precision"]:.4f}')
        print(f'召回率: {result["recall"]:.4f}')
        print(f'F1值: {result["f1"]:.4f}')
        print('\n详细评估说明:')
        print(result['report'])

        return result


def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/01_msra_bert_base.json'

    config = Config(config_path=config_path)
    tester = Tester(config)
    tester.test()


if __name__ == '__main__':
    main()
