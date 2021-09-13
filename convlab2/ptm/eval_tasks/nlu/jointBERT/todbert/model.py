import torch
from torch import nn
from transformers import BertModel


class ToDBertForNLU(nn.Module):
    def __init__(self, model_config, device, slot_dim, intent_dim, intent_weight):
        super(ToDBertForNLU, self).__init__()
        self.model_config = model_config
        self.device = device
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.intent_weight = torch.tensor(intent_weight)

        print('pre-trained weights:', model_config.pretrained_weights)
        self.bert = BertModel.from_pretrained(model_config.pretrained_weights)
        self.dropout = nn.Dropout(model_config.dropout)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None):
        if self.model_config.probing:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits, intent_logits, slot_loss, intent_loss,
