import torch
from torch import nn
from transformers import DistilBertModel


class DistilBertForSequenceClassification(nn.Module):

    def __init__(self, config, num_labels=2):
        super(DistilBertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.bert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased',
            output_hidden_states=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        last_hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = torch.mean(last_hidden[0], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits