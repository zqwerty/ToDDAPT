import numpy as np
import torch
import torch.nn as nn
from transformers import *

from .base import BaseModel


class IntentRecognitionModel(BaseModel):
    def __init__(self, args):  # , num_labels, device):
        super().__init__(args)
        self.hidden_dim = args["hdd_size"]
        self.rnn_num_layers = args["num_rnn_layers"]
        self.num_labels = args["num_labels"]
        self.xeloss = nn.CrossEntropyLoss()
        self.n_gpu = args["n_gpu"]

        self.bert_output_dim = args["config"].hidden_size
        # self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob

        if self.args["more_linear_mapping"]:
            self.one_more_layer = nn.Linear(self.bert_output_dim, self.bert_output_dim)

        self.classifier = nn.Linear(self.bert_output_dim, self.num_labels)

        # Prepare Optimizer
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args["learning_rate"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args["learning_rate"]},
            ]
            return optimizer_grouped_parameters

        if self.n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.module)

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=args["learning_rate"], )
        # warmup=args["warmup_proportion"],
        # t_total=t_total)

    def optimize(self):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args["grad_clip"])
        self.optimizer.step()

    def forward(self, data):
        context, intent = data
        self.optimizer.zero_grad()

        hidden_head = self.encode_context(context)

        # loss
        if self.args["more_linear_mapping"]:
            hidden_head = self.one_more_layer(hidden_head)

        logits = self.classifier(hidden_head)
        loss = self.xeloss(logits, intent)

        if self.training:
            self.loss_grad = loss
            self.optimize()

        softmax = nn.Softmax(-1)
        predictions = torch.argmax(logits, -1)

        outputs = {
            "loss": loss.item(),
            "pred": predictions.detach().cpu().numpy(),
            "label": intent.detach().cpu().numpy(),
            "prob": softmax(logits)
        }

        return outputs

    def evaluation(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)

        if self.args["task_name"] == "intent":
            oos_idx = self.args["unified_meta"]["intent"]["oos"]
            acc = (preds == labels).mean()
            oos_labels, oos_preds = [], []
            ins_labels, ins_preds = [], []
            for i in range(len(preds)):
                if labels[i] != oos_idx:
                    ins_preds.append(preds[i])
                    ins_labels.append(labels[i])

                oos_labels.append(int(labels[i] == oos_idx))
                oos_preds.append(int(preds[i] == oos_idx))

            ins_preds = np.array(ins_preds)
            ins_labels = np.array(ins_labels)
            oos_preds = np.array(oos_preds)
            oos_labels = np.array(oos_labels)
            ins_acc = (ins_preds == ins_labels).mean()
            oos_acc = (oos_preds == oos_labels).mean()

            # for oos samples recall = tp / (tp + fn) 
            TP = (oos_labels & oos_preds).sum()
            FN = ((oos_labels - oos_preds) > 0).sum()
            recall = TP / (TP + FN)
            results = {"acc": acc, "ins_acc": ins_acc, "oos_acc": oos_acc, "oos_recall": recall}
        else:
            acc = (preds == labels).mean()
            results = {"acc": acc}

        return results