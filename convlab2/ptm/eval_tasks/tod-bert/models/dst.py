import torch
import math
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertTokenizer
from convlab2.ptm.pretraining.model import DialogBertModel
import numpy as np
from transformers import *

def _gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class DialogBertForDST(nn.Module):
    def __init__(self, args):
        super(DialogBertForDST, self).__init__()
        
        self.args = args
        self.n_gpu = args["n_gpu"]
        self.hidden_dim = args["hdd_size"]
        self.num_labels = [len(v) for k, v in args["unified_meta"]["slots"].items()]
        self.num_slots = len(self.num_labels)
        self.tokenizer = args["tokenizer"]

        self.slots = [k for k, v in self.args["unified_meta"]["slots"].items()]
        self.slot_value2id_dict = self.args["unified_meta"]["slots"]
        self.slot_id2value_dict = {}
        for k, v in self.slot_value2id_dict.items():
            self.slot_id2value_dict[k] = {vv: kk for kk, vv in v.items()}

        self.utterance_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])
        self.bert_output_dim = args["config"].hidden_size
        
        if self.args["fix_encoder"]:
            print("[Info] Utterance Encoder does not requires grad...")
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])
        print("[Info] SV Encoder does not requires grad...")
        for p in self.sv_encoder.parameters():
            p.requires_grad = False

        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in self.num_labels])
        self.nll = CrossEntropyLoss(ignore_index=-1)

        self.project_W_1 = nn.ModuleList([nn.Linear(self.bert_output_dim, self.bert_output_dim) for _ in range(self.num_slots)])
        self.project_W_2 = nn.ModuleList([nn.Linear(2*self.bert_output_dim, self.bert_output_dim) for _ in range(self.num_slots)])
        self.project_W_3 = nn.ModuleList([nn.Linear(self.bert_output_dim, 1) for _ in range(self.num_slots)])

        if self.args["gate_supervision_for_dst"]:
            self.gate_classifier = nn.Linear(self.bert_output_dim, 2)

        self.start_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

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

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["eps"])
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=args["max_step"])

        self.initialize_slot_value_lookup()

    def get_turn_embeddings_num(self):
        return self.utterance_encoder.get_turn_embeddings().weight.size(0)

    def resize_turn_embeddings(self, new_tokens_num):
        self.utterance_encoder.resize_turn_embeddings(new_tokens_num)

    def optimize(self):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args["grad_clip"])
        self.optimizer.step()
        # self.scheduler.step()

    def initialize_slot_value_lookup(self, max_seq_length=32):

        self.sv_encoder.eval()
        
        label_ids = []
        position_ids = []
        turn_ids = []
        role_ids = []
        for dslot, value_dict in self.slot_value2id_dict.items():
            label_id = []
            position_id = []
            turn_id = []
            role_id = []
            value_dict_rev = {v:k for k, v in value_dict.items()}
            for i in range(len(value_dict)):
                label = value_dict_rev[i]
                label = " ".join([i for i in label.split(" ") if i != ""])

                label_tokens = [self.start_token] + self.tokenizer.tokenize(label) + [self.sep_token]
                label_token_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)
                if "dialog" in self.args["model_type"]:
                    pids = self.tokenizer.create_positional_ids_from_sequences(label_token_ids)
                    tids, rids = self.tokenizer.create_token_type_ids_from_sequences(label_token_ids)
                label_len = len(label_token_ids)

                label_padding = [0] * (max_seq_length - len(label_token_ids))
                label_token_ids += label_padding
                assert len(label_token_ids) == max_seq_length
                label_id.append(label_token_ids)
                if "dialog" in self.args["model_type"]:
                    pids += label_padding
                    tids += label_padding
                    rids += label_padding
                    assert len(pids) == max_seq_length
                    assert len(tids) == max_seq_length
                    assert len(rids) == max_seq_length
                    position_id.append(pids)
                    turn_id.append(tids)
                    role_id.append(rids)
                
            label_id = torch.tensor(label_id).long()
            label_ids.append(label_id)

            if "dialog" in self.args["model_type"]:
                position_id = torch.tensor(position_id).long()
                turn_id = torch.tensor(turn_id).long()
                role_id = torch.tensor(role_id).long()

                position_ids.append(position_id)
                turn_ids.append(turn_id)
                role_ids.append(role_id)

        if "dialog" in self.args["model_type"]:
            iter_data = zip(label_ids, position_ids, turn_ids, role_ids)
        else:
            iter_data = label_ids

        for s, data in enumerate(iter_data):
            if "dialog" in self.args["model_type"]:
                label_id, position_id, turn_id, role_id = data
                inputs = {
                    "input_ids": label_id, 
                    "attention_mask": (label_id > 0).long(),
                    "position_ids": position_id,
                    "turn_ids": turn_id,
                    "role_ids": role_id
                    }
            else:
                label_id = data
                inputs = {
                    "input_ids": label_id, 
                    "attention_mask": (label_id > 0).long(),
                    }
            
            if self.args["sum_token_emb_for_value"]:
                hid_label = self.utterance_encoder.embeddings(input_ids=label_id).sum(1)
            else:
                hid_label = self.sv_encoder(**inputs)[0]
                hid_label = hid_label[:, 0, :]
            
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")


    def forward(self, data):
        batch_size = data["context"].size(0)
        labels = data["belief_ontology"]

        input_ids = data["context"]
        attention_mask = (data["context"] > 0).long()
        if "dialog" in self.args["model_type"]:
            position_id_tensor = data["position_id"]
            turn_id_tensor = data["turn_id"]
            role_id_tensor = data["role_id"]
            inputs = {
                "input_ids": input_ids, 
                "attention_mask": attention_mask,
                "position_ids": position_id_tensor,
                "turn_ids": turn_id_tensor,
                "role_ids": role_id_tensor
            }
        else:
            inputs = {
                "input_ids": input_ids, 
                "attention_mask": attention_mask,
            }

        hidden = self.utterance_encoder(**inputs)[0]
        hidden_rep = hidden[:, 0, :]

        loss = 0
        pred_slot = []


        if self.args["oracle_domain"]:
            
            for slot_id in range(self.num_slots):
                pred_slot_local = []
                for bsz_i in range(batch_size):
                    hidden_bsz = hidden[bsz_i, :, :]
                    
                    if slot_id in data["triggered_ds_idx"][bsz_i]:

                        temp = [i for i, idx in enumerate(data["triggered_ds_idx"][bsz_i]) if idx == slot_id]
                        assert len(temp) == 1
                        ds_pos = data["triggered_ds_pos"][bsz_i][temp[0]]

                        hid_label = self.value_lookup[slot_id].weight # v * d
                        hidden_ds = hidden_bsz[ds_pos, :].unsqueeze(1) # d * 1
                        hidden_ds = torch.cat([hidden_ds, hidden_bsz[0, :].unsqueeze(1)], 0) # 2d * 1
                        hidden_ds = self.project_W_2[0](hidden_ds.transpose(1, 0)).transpose(1, 0) # d * 1

                        _dist = torch.mm(hid_label, hidden_ds).transpose(1, 0) # 1 * v, 51.6%
                        
                        _, pred = torch.max(_dist, -1)
                        pred_item = pred.item()

                        if labels is not None:
                            
                            if (self.args["gate_supervision_for_dst"] and labels[bsz_i, slot_id] != 0) or\
                               (not self.args["gate_supervision_for_dst"]):
                                _loss = self.nll(_dist, labels[bsz_i, slot_id].unsqueeze(0))
                                loss += _loss
                        
                        if self.args["gate_supervision_for_dst"]:
                            _dist_gate = self.gate_classifier(hidden_ds.transpose(1, 0))
                            _loss_gate = self.nll(_dist_gate, data["slot_gate"][bsz_i, slot_id].unsqueeze(0))
                            loss += _loss_gate
                            
                            if torch.max(_dist_gate, -1)[1].item() == 0:
                                pred_item = 0
                            
                        pred_slot_local.append(pred_item)
                    else:
                        # print("slot_id Not Found")
                        pred_slot_local.append(0)
                
                pred_slot.append(torch.tensor(pred_slot_local).unsqueeze(1))
            
            predictions = torch.cat(pred_slot, 1).numpy()
            labels = labels.detach().cpu().numpy()
            
        else:
            for slot_id in range(self.num_slots): ## note: target_slots are successive
                # loss calculation
                hid_label = self.value_lookup[slot_id].weight # v * d
                num_slot_labels = hid_label.size(0)

                _hidden = _gelu(self.project_W_1[slot_id](hidden_rep))
                _hidden = torch.cat([hid_label.unsqueeze(0).repeat(batch_size, 1, 1), _hidden.unsqueeze(1).repeat(1, num_slot_labels, 1)], dim=2)
                _hidden = _gelu(self.project_W_2[slot_id](_hidden))
                _hidden = self.project_W_3[slot_id](_hidden)
                _dist = _hidden.squeeze(2) # b * 1 * num_slot_labels

                _, pred = torch.max(_dist, -1)
                pred_slot.append(pred.unsqueeze(1))

                if labels is not None:
                    _loss = self.nll(_dist, labels[:, slot_id])
                    loss += _loss

            predictions = torch.cat(pred_slot, 1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

        if self.training:
            self.loss_grad = loss
            self.optimize()

        outputs = {"loss": loss, "pred": predictions, "label": labels}

        return outputs

    def evaluation(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)

        slot_acc, joint_acc, slot_acc_total, joint_acc_total = 0, 0, 0, 0
        for pred, label in zip(preds, labels):
            joint = 0

            assert len(pred) == len(label)

            for i, p in enumerate(pred):
                pred_str = self.slot_id2value_dict[self.slots[i]][p]
                gold_str = self.slot_id2value_dict[self.slots[i]][label[i]]

                if pred_str == gold_str or pred_str in gold_str.split("|"):
                    slot_acc += 1
                    joint += 1
                slot_acc_total += 1

            if joint == len(pred):
                joint_acc += 1

            joint_acc_total += 1

        joint_acc = joint_acc / joint_acc_total
        slot_acc = slot_acc / slot_acc_total
        results = {"joint_acc": joint_acc, "slot_acc": slot_acc}
        print("Results 1: ", results)

        return results
