import torch
import torch.nn as nn
from transformers import BertModel


class BertWithDropout(nn.Module):

    def __init__(self, model_name, num_labels, dropout_rate=0.2):
        super().__init__()
        self.num_labels = num_labels

        # 从本地加载预训练BERT
        self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    # 前向传播，输出每个token的logits，用于最终分类
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # -100会被CrossEntropyLoss忽略，方便跳过填充和特殊token
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        class Output:
            pass

        output = Output()
        output.loss = loss
        output.logits = logits
        return output
