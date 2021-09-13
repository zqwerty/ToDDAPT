import torch
import torch.nn.functional as F

from collections import defaultdict
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss 
from torch.nn import Dropout
from transformers import BertConfig, BertModel, BertForMaskedLM
from typing import Any
import sys
from convlab2.ptm.pretraining.model import DialogBertModel, DialogBertForMaskedLM

class DialogBertPretrain(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str):
        super(DialogBertPretrain, self).__init__()
        self.bert_model = DialogBertForMaskedLM.from_pretrained(model_name_or_path)

    def forward(self, 
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                turn_ids: torch.tensor,
                position_ids: torch.tensor,
                role_ids: torch.tensor,
                mlm_labels: torch.tensor):
        outputs = self.bert_model(input_ids, attention_mask, turn_ids, position_ids, role_ids, masked_lm_labels=mlm_labels)

        return outputs[0]

class IntentDialogBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 probing: bool,
                 ignore_pooler: bool,
                 **unused_args):
        super(IntentDialogBertModel, self).__init__()
        self.probing = probing
        self.ignore_pooler = ignore_pooler
        self.bert_model = DialogBertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                turn_ids: torch.tensor,
                position_ids: torch.tensor,
                role_ids: torch.tensor,
                intent_label: torch.tensor = None,
                **unused_args):
        hidden_states, pooled_output = self.bert_model(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       turn_ids=turn_ids,
                                                       position_ids=position_ids,
                                                       role_ids=role_ids)
        if self.ignore_pooler:
            pooled_output = hidden_states[:, 0, :]
        if self.probing:
            pooled_output.detach_()
        intent_logits = self.intent_classifier(self.dropout(pooled_output))

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_logits, intent_loss


class SlotDialogBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_slot_labels: int,
                 probing: bool,
                 ignore_pooler: bool,
                 **unused_args):
        super(SlotDialogBertModel, self).__init__()
        self.probing = probing
        self.bert_model = DialogBertModel.from_pretrained(model_name_or_path)
        self.dropout = Dropout(dropout)
        self.num_slot_labels = num_slot_labels
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

    def encode(self,
               input_ids: torch.tensor,
               attention_mask: torch.tensor,
               turn_ids: torch.tensor,
               position_ids: torch.tensor,
               role_ids: torch.tensor,
               ):
        hidden_states, _ = self.bert_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           turn_ids=turn_ids,
                                           position_ids=position_ids,
                                           role_ids=role_ids)
        return hidden_states

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                turn_ids: torch.tensor,
                position_ids: torch.tensor,
                role_ids: torch.tensor,
                slot_labels: torch.tensor = None,
                **unused_args):
        hidden_states = self.encode(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    turn_ids=turn_ids,
                                    position_ids=position_ids,
                                    role_ids=role_ids)
        if self.probing:
            hidden_states.detach_()
        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        return slot_logits, slot_loss

class JointSlotIntentDialogBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 num_slot_labels: int,
                 probing: bool,
                 ignore_pooler: bool,
                 ):
        super(JointSlotIntentDialogBertModel, self).__init__()
        self.bert_model = DialogBertModel.from_pretrained(model_name_or_path)
        self.probing = probing
        self.ignore_pooler = ignore_pooler
        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                turn_ids: torch.tensor,
                position_ids: torch.tensor,
                role_ids: torch.tensor,
                intent_label: torch.tensor = None,
                slot_labels: torch.tensor = None,
                **unused_args):
        hidden_states, pooled_output = self.bert_model(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       turn_ids=turn_ids,
                                                       position_ids=position_ids,
                                                       role_ids=role_ids)
        if self.ignore_pooler:
            pooled_output = hidden_states[:, 0, :]
        if self.probing:
            hidden_states.detach_()
            pooled_output.detach_()

        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_logits, slot_logits, intent_loss + slot_loss
