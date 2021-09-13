import os
import json
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import LayerNorm
from transformers import BertModel


class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


class MLP(nn.Module):
    def __init__(self, inp_size, hidden_size, out_size, dropout):
        super().__init__()
        self.layer1 = nn.Linear(inp_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = GELU()
        self.layernorm = LayerNorm(hidden_size)

    def forward(self, input_tensor):
        output = self.layer1(input_tensor)
        output = self.act_fn(output)
        output = self.layernorm(output)
        output = self.dropout(output)
        output = self.layer2(output)
        return output


class BertForBIOTag(nn.Module):
    def __init__(self, model_config, device, tag_dim, tag_weight=None, slot_ontology=None):
        super(BertForBIOTag, self).__init__()
        self.model_config = model_config
        self.device = device
        self.num_labels = tag_dim
        self.tag_weight = torch.tensor(tag_weight, requires_grad=False) if tag_weight is not None else torch.tensor([1.]*tag_dim)
        print('tag_weight:')
        print(self.tag_weight)

        print('pre-trained weights:', model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.tag_classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        nn.init.xavier_uniform_(self.tag_classifier.weight)

        self.tag_loss_fct = torch.nn.CrossEntropyLoss(weight=self.tag_weight)

        if slot_ontology is not None:
            self.slot_ontology = slot_ontology
            self.num_cate_slot = len(slot_ontology['categorical_slot_list'])
            self.cate_slot_ontology = slot_ontology['categorical_slot_ontology']
            self.num_cate_slot_value = []
            for _domain, _slot in slot_ontology['categorical_slot_list']:
                self.num_cate_slot_value.append(len(self.cate_slot_ontology[_domain][_slot]))

            self.num_noncate_slot = len(slot_ontology['non-categorical_slot_list'])
            self.cate_slot_classifier = nn.ModuleList([MLP(self.bert.config.hidden_size, self.bert.config.hidden_size, self.num_cate_slot_value[slot_idx], model_config['dropout']) for slot_idx in range(self.num_cate_slot)])
            self.noncate_slot_classifier = nn.ModuleList([MLP(self.bert.config.hidden_size, self.bert.config.hidden_size, 2, model_config['dropout']) for _ in range(self.num_noncate_slot)])
            self.cate_slot_loss_fct = torch.nn.CrossEntropyLoss()
            self.noncate_slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None, cate_slot_labels=None, noncate_slot_labels=None):
        outputs = self.bert(input_ids=word_seq_tensor, attention_mask=word_mask_tensor)

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        tag_logits = self.tag_classifier(sequence_output)
        outputs = (tag_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = tag_logits.view(-1, self.num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            tag_loss = self.tag_loss_fct(active_tag_logits, active_tag_labels)
            outputs = outputs + (tag_loss,)

        all_slot_logits = []
        cate_slot_loss = 0
        for cate_idx in range(len(self.cate_slot_classifier)):
            cate_slot_logits = self.cate_slot_classifier[cate_idx](pooled_output)
            if cate_slot_labels is not None:
                cate_slot_loss += self.cate_slot_loss_fct(cate_slot_logits, cate_slot_labels[cate_idx])
            all_slot_logits.append(cate_slot_logits)

        if cate_slot_labels is not None:
            cate_slot_loss /= self.num_cate_slot

        noncate_slot_loss = 0
        for noncate_idx in range(len(self.noncate_slot_classifier)):
            noncate_slot_logits = self.noncate_slot_classifier[noncate_idx](pooled_output)
            if noncate_slot_labels is not None:
                noncate_slot_loss += self.noncate_slot_loss_fct(noncate_slot_logits, noncate_slot_labels[noncate_idx])
            all_slot_logits.append(noncate_slot_logits)

        if noncate_slot_labels is not None:
            noncate_slot_loss /= self.num_noncate_slot

        outputs += (all_slot_logits, )  # 30 * (bs, n_values)
        if cate_slot_labels is not None:
            outputs += (cate_slot_loss, )
        if noncate_slot_labels is not None:
            outputs += (noncate_slot_loss, )

        return outputs  # tag_logits, tag_loss, all_slot_logits, cate_slot_loss, noncate_slot_loss,
