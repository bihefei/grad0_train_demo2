def _extract_entities(tags):
    
    entities = set()
    start = None
    cur_type = None

    for idx, tag in enumerate(tags):
        if tag == 'O':
            # 遇到 O，收尾当前实体
            if start is not None:
                entities.add((cur_type, start, idx - 1))
                start, cur_type = None, None
            continue

        prefix, _, tag_type = tag.partition('-')
        if prefix == 'B':
            # B- 开启一个新实体；若上一个实体尚未闭合则先收尾
            if start is not None:
                entities.add((cur_type, start, idx - 1))
            start, cur_type = idx, tag_type
        elif prefix == 'I':
            # I- 只有紧跟同类型实体时才算延续，否则当作新实体的起点
            if start is None or cur_type != tag_type:
                if start is not None:
                    entities.add((cur_type, start, idx - 1))
                start, cur_type = idx, tag_type

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
