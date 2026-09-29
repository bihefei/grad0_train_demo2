import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

try:
    from .config import Config
    from .dataset import NERDataset
    from .model import BertWithDropout
    from .metrics import EntityLevelEvaluator
except ImportError:
    from config import Config
    from dataset import NERDataset
    from model import BertWithDropout
    from metrics import EntityLevelEvaluator


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

    # 评估函数，先收集整个测试集的预测id与标签id，再交给评测器统计精确率、召回率和F1
    def _evaluate(self, model, data_loader, evaluator):
        model.eval()
        all_pred_ids = []
        all_label_ids = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids, attention_mask)
                pred_ids = torch.argmax(outputs.logits, dim=-1)

                all_pred_ids.extend(pred_ids.cpu().tolist())
                all_label_ids.extend(labels.cpu().tolist())

        return evaluator.compute(all_pred_ids, all_label_ids)

    # 测试入口，加载权重和标签表，并在测试集上验证模型效果
    def test(self, checkpoint_path=None):
        save_dir = self.config.get_experiment_dir()
        model_path = checkpoint_path or os.path.join(save_dir, 'best_model.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到测试用模型权重: {model_path}，请先执行训练生成最佳模型。')

        label_list = self._load_label_list(save_dir)
        paths, _ = NERDataset.get_dataset_paths(self.config.dataset, self.config.data_dir)
        tokenizer = BertTokenizerFast.from_pretrained(self.config.model_name, local_files_only=True)

        test_set = NERDataset(
            data_path=paths['test'],
            tokenizer=tokenizer,
            max_seq_len=self.config.max_seq_len,
            label_list=label_list
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=NERDataset.collate_fn
        )

        model = BertWithDropout(
            model_name=self.config.model_name,
            num_labels=len(label_list),
            dropout_rate=self.config.dropout_rate
        )
        model.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        # 评测器使用训练时保存的标签表，保证id到标签的翻译与训练一致
        evaluator = EntityLevelEvaluator(label_list)
        result = self._evaluate(model, test_loader, evaluator)

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
