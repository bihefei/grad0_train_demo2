#   只有 B- 能开启实体
#   I- 只能延续"已经在实体中且类型相同"的标签，裸 I- 按 O 处理，不产生实体
#   在标签序列末尾补一个哨兵 'O'，强制闭合最后一个实体，循环外不用再单独收尾
#   把"实体结束"和"实体开始"拆成两个独立判定，各自只看一半信息，避免在同一个分支里既收尾又开启时漏掉某一种情况
class EntityLevelEvaluator:

    def __init__(self, label_list):
        self.label_list = list(label_list)

    # 从一条BIO标签序列中解析出实体
    # 返回 {(实体类型, 起始下标, 结束下标)}，跨句不需要，按句返回
    @staticmethod
    def get_entities(tags):
        entities = set()
        current_type = None
        begin = 0

        # 末尾补哨兵'O'，保证最后一个实体一定会被闭合
        for index, tag in enumerate(list(tags) + ['O']):
            prefix, _, tag_type = tag.partition('-')
            if prefix not in ('B', 'I'):
                prefix, tag_type = 'O', ''

            # 延续判定：只有"处于实体中"且"是同类型的I-"才算延续，其余一律中断
            # current_type 为None时，即使遇到 I- 也不成立
            if prefix == 'I' and current_type == tag_type:
                continue

            # 当前实体已经结束
            if current_type is not None:
                entities.add((current_type, begin, index - 1))
                current_type = None

            # 开始判定：只有B-能开启实体
            if prefix == 'B':
                current_type, begin = tag_type, index

        return entities

    # 把 (batch, seq_len) 的id序列翻译成逐句的标签序列
    # -100 的位置（padding、[CLS]/[SEP]、同一个词的非首子词）不参与评测
    def _decode(self, pred_ids, labels):
        true_tags = []
        pred_tags = []

        for prediction, label in zip(pred_ids, labels):
            true_tags.append([self.label_list[l] for p, l in zip(prediction, label) if l != -100])
            pred_tags.append([self.label_list[p] for p, l in zip(prediction, label) if l != -100])

        return true_tags, pred_tags

    # 统计实体级精确率、召回率、F1
    # 逐句解析出实体集合后取交集，只有类型与边界完全一致才算命中
    def compute(self, pred_ids, labels):
        true_tags, pred_tags = self._decode(pred_ids, labels)

        correct = 0
        pred_total = 0
        true_total = 0

        # 按类型分别统计，用于生成分类报告
        per_type_correct = {}
        per_type_pred = {}
        per_type_true = {}

        for true_seq, pred_seq in zip(true_tags, pred_tags):
            true_entities = self.get_entities(true_seq)
            pred_entities = self.get_entities(pred_seq)

            true_total += len(true_entities)
            pred_total += len(pred_entities)
            correct += len(true_entities & pred_entities)

            for entity_type, _, _ in true_entities:
                per_type_true[entity_type] = per_type_true.get(entity_type, 0) + 1
            for entity_type, _, _ in pred_entities:
                per_type_pred[entity_type] = per_type_pred.get(entity_type, 0) + 1
            for entity_type, _, _ in (true_entities & pred_entities):
                per_type_correct[entity_type] = per_type_correct.get(entity_type, 0) + 1

        precision = correct / pred_total if pred_total else 0.0
        recall = correct / true_total if true_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report = self._build_report(
            precision, recall, f1,
            correct, pred_total, true_total,
            per_type_correct, per_type_pred, per_type_true
        )

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report,
        }

    # 组装评测报告：按实体类型给出各自的P/R/F1，最后给出整体结果
    @staticmethod
    def _build_report(precision, recall, f1, correct, pred_total, true_total,
                      per_type_correct, per_type_pred, per_type_true):
        lines = [f'{"type":>12}{"precision":>12}{"recall":>12}{"f1":>12}{"support":>12}']

        for entity_type in sorted(set(per_type_true) | set(per_type_pred)):
            hit = per_type_correct.get(entity_type, 0)
            num_pred = per_type_pred.get(entity_type, 0)
            num_true = per_type_true.get(entity_type, 0)

            type_precision = hit / num_pred if num_pred else 0.0
            type_recall = hit / num_true if num_true else 0.0
            type_f1 = (2 * type_precision * type_recall / (type_precision + type_recall)
                       if (type_precision + type_recall) > 0 else 0.0)

            lines.append(f'{entity_type:>12}{type_precision:>12.4f}'
                         f'{type_recall:>12.4f}{type_f1:>12.4f}{num_true:>12}')

        lines.append(f'{"micro":>12}{precision:>12.4f}{recall:>12.4f}{f1:>12.4f}{true_total:>12}')
        lines.append(f'correct={correct}, predicted={pred_total}, true={true_total}')

        return '\n'.join(lines)
