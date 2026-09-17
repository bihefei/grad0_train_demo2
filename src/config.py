import glob
import json
import os
import time

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # 读取JSON配置文件
    def __init__(self, config_path):
        config_path = os.path.abspath(os.path.join(PROJECT_ROOT, config_path)) if not os.path.isabs(config_path) else config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'配置文件不存在: {config_path}')

        with open(config_path,"r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.config_path = config_path
        self.experiment_name = os.path.splitext(os.path.basename(config_path))[0]

        self.dataset = cfg.get('dataset', 'MSRA')
        self.data_dir = cfg.get('data_dir', os.path.join(PROJECT_ROOT, 'data'))
        self.model_name = cfg.get('model_name', os.path.join(PROJECT_ROOT, 'bert-base-chinese'))

        self.max_seq_len = cfg.get('max_seq_len', 128)
        self.batch_size = cfg.get('batch_size', 8)
        self.epochs = cfg.get('epochs', 10)
        self.lr = cfg.get('lr', 2e-5)
        self.weight_decay = cfg.get('weight_decay', 0.01)
        self.warmup_ratio = cfg.get('warmup_ratio', 0.1)
        self.dropout_rate = cfg.get('dropout_rate', 0.2)

        self.seed = cfg.get('seed', 101)
        self.device = cfg.get('device', 'cuda')
        self.save_dir = cfg.get('save_dir', os.path.join(PROJECT_ROOT, 'checkpoints'))
        # 本次实验实际的保存目录
        self.experiment_dir = None

        # SwanLab监控开关
        self.use_swanlab = cfg.get('use_swanlab', True)
        self.swanlab_project = cfg.get('swanlab_project', 'Chinese_NER_Demo')

        self.data_dir = self._resolve_path(self.data_dir)
        self.model_name = self._resolve_path(self.model_name)
        self.save_dir = self._resolve_path(self.save_dir)

        if self.device == 'cuda' and not torch.cuda.is_available():
            print('CUDA不可用，自动切换为CPU运行')
            self.device = 'cpu'

        self._validate()

    def _resolve_path(self, path):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(PROJECT_ROOT, path))

    # 关键参数校验
    def _validate(self):
        assert self.dataset in ['MSRA', 'weibo'], \
            f'不支持的数据集: {self.dataset}，可选 MSRA / weibo'
        assert self.batch_size > 0, 'batch_size 必须大于0'
        assert self.epochs > 0, 'epochs 必须大于0'
        assert self.lr > 0, '学习率必须大于0'
        assert 0 <= self.warmup_ratio <= 1, 'warmup_ratio 必须在 0-1 之间'
        assert 0 <= self.dropout_rate <= 1, 'dropout_rate 必须在 0-1 之间'
        assert os.path.isdir(self.data_dir), f'数据目录不存在: {self.data_dir}'
        assert os.path.isdir(self.model_name), \
            f'模型目录不存在: {self.model_name}（已强制本地加载，不会联网下载）'

    # 确定本次实验的保存目录
    # 默认：save_dir / experiment_name（与config文件名一致）
    # 训练时：该目录若已存在训练好的权重，自动追加时间戳新建目录，历史结果不会被覆盖
    # 测试/预测时：默认目录不存在，则自动选用该实验名最新一次带时间戳的结果
    def get_experiment_dir(self, for_training=False):
        if self.experiment_dir:
            return self.experiment_dir

        base = os.path.join(self.save_dir, self.experiment_name)

        if for_training:
            if os.path.exists(os.path.join(base, 'best_model.pt')):
                stamp = time.strftime('%Y%m%d_%H%M%S')
                base = f'{base}_{stamp}'
                print(f'检测到已存在历史训练结果，本次将保存到新目录（不覆盖旧版本）: {base}')
            os.makedirs(base, exist_ok=True)
        elif not os.path.isdir(base):
            candidates = [d for d in glob.glob(f'{base}_*')
                          if os.path.isfile(os.path.join(d, 'best_model.pt'))]
            if candidates:
                base = max(candidates, key=os.path.getmtime)

        self.experiment_dir = base
        return base

    # 当前配置保存成JSON文件
    def save(self, save_path):
        save_path = self._resolve_path(save_path)
        save_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
