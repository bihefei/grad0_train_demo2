import os
import sys

try:
    from .config import Config
    from .train import Trainer
except ImportError:
    from config import Config
    from train import Trainer


def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/01_msra_bert_base.json'

    config = Config(config_path=config_path)

    print(f'  当前实验版本: {os.path.basename(config_path)}')
    print(f'  数据集: {config.dataset}')
    print(f'  预训练模型: {config.model_name}')
    print(f'  运行设备: {config.device}')
    print(f'  训练轮数: {config.epochs}')
    print(f'  批次大小: {config.batch_size}')
    print(f'  学习率: {config.lr}')
    print(f'  Dropout率: {config.dropout_rate}')
    print('-' * 50)

    trainer = Trainer(config)
    result = trainer.train_and_evaluate()

    # 训练完成后，保存当前实验配置，方便复现同一实验
    save_dir = os.path.join(config.save_dir, f'{os.path.basename(config.model_name)}_{config.dataset}')
    config.save(os.path.join(save_dir, 'experiment_config.json'))

    print('\n' + '=' * 50)
    print('   训练与验证完成！最佳验证结果:')
    print(f'  F1值: {result["dev_result"]["f1"]:.4f}')
    print(f'  精确率: {result["dev_result"]["precision"]:.4f}')
    print(f'  召回率: {result["dev_result"]["recall"]:.4f}')
    print(f'\n  最佳模型与配置已保存至: {save_dir}')


if __name__ == '__main__':
    main()
