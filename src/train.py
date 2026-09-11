import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import swanlab

try:
    from .dataset import NERDataset, get_dataset_paths, collate_fn
    from .model import BertWithDropout
    from .metrics import compute_entity_level_metrics
except ImportError:
    from dataset import NERDataset, get_dataset_paths, collate_fn
    from model import BertWithDropout
    from metrics import compute_entity_level_metrics


# 训练器，加载数据、训练模型、验证和保存最佳模型
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

    # 固定随机种子
    def _set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 评估模型精确率、召回率和F1
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

    # 训练过程：加载数据、训练、验证、保存最优模型
    def train_and_evaluate(self):
        self._set_seed()
        if self.config.use_swanlab:
            swanlab.init(
                project=self.config.swanlab_project,
                config={k: v for k, v in self.config.__dict__.items() if not k.startswith('_')},
                experiment_name=f'{os.path.basename(self.config.model_name)}_{self.config.dataset}'
            )
        paths, label_list = get_dataset_paths(self.config.dataset, self.config.data_dir)
        tokenizer = BertTokenizer.from_pretrained(self.config.model_name, local_files_only=True)
        train_set = NERDataset(
            data_path=paths['train'],
            tokenizer=tokenizer,
            max_seq_len=self.config.max_seq_len,
            label_list=label_list
        )
        label_list = train_set.label_list
        num_labels = train_set.num_labels
        dev_set = NERDataset(
            data_path=paths['dev'],
            tokenizer=tokenizer,
            max_seq_len=self.config.max_seq_len,
            label_list=label_list
        )

        # DataLoader的worker固定随机种子
        def _seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            worker_init_fn=_seed_worker,
            collate_fn=collate_fn
        )
        dev_loader = DataLoader(dev_set, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)
        model = BertWithDropout(
            model_name=self.config.model_name,
            num_labels=num_labels,
            dropout_rate=self.config.dropout_rate
        )
        model.to(self.device)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_params, lr=self.config.lr)
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        best_f1 = 0.0
        best_dev_result = None
        save_dir = os.path.join(self.config.save_dir, f'{os.path.basename(self.config.model_name)}_{self.config.dataset}')
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(self.config.epochs):
            model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_train_loss += loss.item()
            avg_loss = total_train_loss / len(train_loader)
            dev_result = self._evaluate(model, dev_loader, label_list)
            print(f'===== Epoch {epoch + 1}/{self.config.epochs} =====')
            print(f'训练损失: {avg_loss:.4f}')
            print(f'验证集: 精确率={dev_result["precision"]:.4f} '
                  f'召回率={dev_result["recall"]:.4f} '
                  f'F1={dev_result["f1"]:.4f}')
            if self.config.use_swanlab:
                swanlab.log({
                    'train/loss': avg_loss,
                    'dev/precision': dev_result['precision'],
                    'dev/recall': dev_result['recall'],
                    'dev/f1': dev_result['f1'],
                    'epoch': epoch + 1
                })
            # 每轮训练结束后，保存在验证集上最好的模型参数
            if dev_result['f1'] > best_f1:
                best_f1 = dev_result['f1']
                best_dev_result = dev_result
                best_model_path = os.path.join(save_dir, 'best_model.pt')
                torch.save(model.state_dict(), best_model_path)
                with open(os.path.join(save_dir, 'label_list.json'), 'w', encoding='utf-8') as f:
                    json.dump(label_list, f, ensure_ascii=False)
                print(f'最佳模型已更新，验证集F1: {best_f1:.4f}')

        if best_dev_result is None:
            raise RuntimeError('训练结束后未找到有效的验证集最佳模型。')
        print('\n===== 训练结束 =====')
        print(f'最佳验证集F1: {best_f1:.4f}')
        print(f'最佳模型保存路径: {os.path.join(save_dir, "best_model.pt")}')
        if self.config.use_swanlab:
            swanlab.finish()

        # 导出实验完整配置
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        config_json_path = os.path.join(save_dir, "experiment_config.json")
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"实验配置文件已保存至：{config_json_path}")

        return {
            'best_f1': best_f1,
            'best_model_path': os.path.join(save_dir, 'best_model.pt'),
            'label_list': label_list,
            'dev_result': best_dev_result,
            'save_dir': save_dir
        }
