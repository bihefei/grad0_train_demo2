# 从BIO标签序列中解析出实体，用于实体级评测
# 规则：B- 才有资格开启实体；I- 只能延续"同类型且已在实体中"的标签
#       裸I-（前面是O、或类型不同）属于非法标注，按O处理，不能算成实体
def _extract_entities(tags):
    entities = set()
    start = None
    cur_type = None

    for idx, tag in enumerate(tags):
        prefix, _, tag_type = tag.partition('-')

        if prefix == 'B':
            # B- 开启一个新实体；若上一个实体尚未闭合则先收尾
            if start is not None:
                entities.add((cur_type, start, idx - 1))
            start, cur_type = idx, tag_type
        elif prefix == 'I':
            if start is not None and cur_type == tag_type:
                # 正常延续：和当前实体同类型，什么都不用做
                continue
            # 裸I-：先给上一个实体收尾，然后丢弃这个token
            if start is not None:
                entities.add((cur_type, start, idx - 1))
            start, cur_type = None, None
        else:
            # O：收尾当前实体
            if start is not None:
                entities.add((cur_type, start, idx - 1))
            start, cur_type = None, None

    # 序列结束时补上最后一个尚未闭合的实体
    if start is not None:
        entities.add((cur_type, start, len(tags) - 1))

    return entities


def compute_entity_level_metrics(all_true_tags, all_pred_tags):
    true_total = 0
    pred_total = 0
    correct = 0

    for true_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        true_entities = _extract_entities(true_tags)
        pred_entities = _extract_entities(pred_tags)

        true_total += len(true_entities)
        pred_total += len(pred_entities)
        correct += len(true_entities & pred_entities)

    precision = correct / pred_total if pred_total else 0.0
    recall = correct / true_total if true_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    report = (
        f"entity-level precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}\n"
        f"correct={correct}, predicted={pred_total}, true={true_total}"
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
    }
