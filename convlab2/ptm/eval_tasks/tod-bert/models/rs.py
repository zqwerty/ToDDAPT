import numpy as np
import torch
import torch.nn as nn
from transformers import *

from .base import BaseModel


class ResponseSelectionModel(BaseModel):
    def __init__(self, args):  # , num_labels, device):
        super().__init__(args)
        self.args = args
        self.device = torch.device(args['device'])
        self.xeloss = nn.CrossEntropyLoss()
        self.n_gpu = args["n_gpu"]

        if self.args["fix_encoder"]:
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        ## Prepare Optimizer
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
        # input_ids, input_len, labels=None, n_gpu=1, target_slot=None):

        self.optimizer.zero_grad()
        context, response = data
        batch_size = context['input_ids'].size(0)
        # max_seq_len = 256

        interval = 25
        start_list = list(np.arange(0, batch_size, interval))
        end_list = start_list[1:] + [None]
        context_outputs, response_outputs = [], []

        for start, end in zip(start_list, end_list):
            context_inputs = {
                k: v[start:end] for k, v in context.items()
            }
            response_inputs = {
                k: v[start:end] for k, v in response.items()
            }
            if "bert" in self.args["model_type"]:
                context_hidden, context_output = self.utterance_encoder(**context_inputs)
                response_hidden, response_output = self.utterance_encoder(**response_inputs)
            elif self.args["model_type"] == "gpt2":
                context_output = self.utterance_encoder(**context_inputs)[0].mean(1)
                response_output = self.utterance_encoder(**response_inputs)[0].mean(1)
            elif self.args["model_type"] == "dialogpt":
                transformer_outputs = self.utterance_encoder.transformer(**context_inputs)
                context_output = transformer_outputs[0].mean(1)
                transformer_outputs = self.utterance_encoder.transformer(**response_inputs)
                response_output = transformer_outputs[0].mean(1)
            else:
                raise ValueError

            context_outputs.append(context_output.cpu())
            response_outputs.append(response_output.cpu())

        # evaluation for k-to-100
        if (not self.training) and (batch_size < self.args["eval_batch_size"]):
            response_outputs.append(self.final_response_output[:self.args["eval_batch_size"] - batch_size, :])

        final_context_output = torch.cat(context_outputs, 0).to(self.device)
        final_response_output = torch.cat(response_outputs, 0).to(self.device)

        if not self.training:
            self.final_response_output = final_response_output.cpu()

        # mat
        logits = torch.matmul(final_context_output, final_response_output.transpose(1, 0))

        # loss
        labels = torch.tensor(np.arange(batch_size)).to(self.device)
        loss = self.xeloss(logits, labels)

        if self.training:
            self.loss_grad = loss
            self.optimize()

        predictions = np.argsort(logits.detach().cpu().numpy(), axis=1)  # torch.argmax(logits, -1)

        outputs = {"loss": loss.item(),
                   "pred": predictions,
                   "label": np.arange(batch_size)}

        return outputs

    def evaluation(self, preds, labels):
        assert len(preds) == len(labels)

        preds = np.array(preds)
        labels = np.array(labels)

        def _recall_topk(preds_top10, labels, k):
            preds = preds_top10[:, -k:]
            acc = 0
            for li, label in enumerate(labels):
                if label in preds[li]: acc += 1
            acc = acc / len(labels)
            return acc

        results = {"top-1": _recall_topk(preds, labels, 1),
                   "top-3": _recall_topk(preds, labels, 3),
                   "top-5": _recall_topk(preds, labels, 5),
                   "top-10": _recall_topk(preds, labels, 10)}

        print(results)

        return results
