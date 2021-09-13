"""
    DST model based on pre-training BERT
"""

__all__ = [
    'DstModel'
]

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from convlab2.ptm.eval_tasks.dst import parser, get_transformer_settings
from convlab2.ptm.eval_tasks.dst.utils import cal_scores, cal_added_scores, cal_matrix, add_states, cal_error

class DstModel(nn.Module):
    def __init__(self, args, ontology, tokenizer=None):
        super().__init__()
        self.args = args
        self.ontology = ontology
        self.device = args.device

        tokenizer_cls, model_cls, config_cls = get_transformer_settings(args.transformer)
        self.transformer = model_cls.from_pretrained(args.transformer_path)
        hidden_size = self.transformer.config.hidden_size

        if tokenizer is None:
            self.tokenizer = tokenizer_cls.from_pretrained(args.transformer_path)
        else:
            self.tokenizer = tokenizer

        self.slot2id = {}
        self.id2slot = []
        self.slot_num = 0
        self.all_values = []
        # only consider slots that tracked in state
        for domain_name in ontology['state']:
            for slot_name in ontology['state'][domain_name]:
                self.slot2id[(domain_name, slot_name)] = self.slot_num
                self.id2slot.append((domain_name, slot_name))
                self.slot_num += 1
                self.all_values.append(ontology['state'][domain_name][slot_name]['possible_values'])

        print(f'slot num: {self.slot_num}')

        self.update_project = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4)
        )

        self.value_project = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        self.span_project = nn.ModuleDict({
            'start': nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            ),
            'end': nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            ),
        })

        self.states = []
        self.init_states()

    def init_states(self):
        self.states = ["" for _ in range(len(self.slot2id))]

    def resize_token_embeddings(self, new_len):
        self.transformer.resize_token_embeddings(new_len)

    def forward(self, inputs, domain_pos, slot_pos, value_mask, span_mask, labels=None):
        r"""
        Parameters:
            inputs (dict, `required`):
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `required`):
                    BERT input ids, for current slot, the ids is the tokenized result of the
                    following components (divided by space):
                        - [CLS]
                        - domain: [DOMAIN] {domain description} [SEP]
                        - slot: [SLOT] {slot description} [SEP]
                        - values: [VALUE] {value1} [SEP] [VALUE] {value2} [SEP] ...
                        - context: [USR] {U_t} [SEP] [SYS] {S_{t-1}} [SEP] [USR] {U_{t-1}} [SEP] ...
                attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`)

                position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`)

                role_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`):
                    available when using `DialogBert`

                turn_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`)
                    available when using `DialogBert`

            domain_pos (list of length `batch size`):
                index of [DOMAIN] token in the input sequence

            slot_pos (list of length `batch size`):
                index of [SLOT] token in the input sequence

            value_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `required`):
                marked each [VALUE] token index in the input sequence

            span_mask (dict, `required`):
                start (`torch.LongTensor` of shape `batch_size, sequence_length)`):
                    marked tokens which are the candidates of the start token

                end (`torch.LongTensor` of shape `batch_size, sequence_length)`):
                    marked tokens which are the candidates of the end token

            labels (dict, `optional`): label dict
                update (`torch.LongTensor` of shape `(batch_size,)`):
                    update label for current slot, possible values are in {0, 1, 2, 3}, respectively denotes:
                        0: keep the previous value
                        1: update the value of the slot with special value of `dontcare`
                        2: update the value of the slot by clearing, i.e. set it to an empty value of `""`
                        3: update the value of the slot by a classifier (for categorical slots) or extracting it
                            it from the utterance according to the span (for non-categorical slots)

                value ((`torch.LongTensor` of shape `(batch_size,)`)):
                    update value index (in `possible_values` field defined in ontology),
                    making sense when the slot is categorical and the update label is 3,
                    set to -1 when the slot is non-categorical or is not updated

                start (`torch.LongTensor` of shape `(batch_size,)`):
                    start token index, making sense when the slot is non-categorical
                    and the update label is 3, must be one of the element of `start`
                    field in `candidates`, set to -1 when the slot is categorical
                    or is not updated

                end (`torch.LongTensor` of shape `(batch_size,)`):
                    end token index, making sense when the slot is non-categorical
                    and the update label is 3, must be one of the element of `end`
                    field in `candidates`, set to -1 when the slot is categorical
                    or is not updated

        Returns:
            pred (dict): prediction dict
                similar almost the same definition as `label` paratewith 4 fields,:
                - update
                - value
                - start
                - end

            loss (`torch.Tensor`): loss scalar tensor
                making sense when `label` is not `None`
        """

        # modify later
        inputs = deepcopy(inputs)
        labels = deepcopy(labels)
        input_ids = inputs['input_ids']
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_size = self.transformer.config.hidden_size

        hidden = self.transformer.forward(**inputs)[0]
        # loss = torch.tensor(0.0).to(self.device)
        pred = {
            k: torch.full((batch_size,), -1, dtype=torch.long).to(self.device)
            for k in ['update', 'value', 'start', 'end']
        }
        loss = {
            k: torch.tensor(0.0).to(self.device)
            for k in ['update', 'value', 'start', 'end']
        }

        cls_hidden = hidden[:, 0, :]
        _domain_hidden = torch.cat([hidden[bid, domain_pos[bid], :][None, :] for bid in range(batch_size)], dim=0)
        slot_hidden = torch.cat([hidden[bid, slot_pos[bid], :][None, :] for bid in range(batch_size)], dim=0)
        update_score = self.update_project.forward(torch.cat((cls_hidden, slot_hidden), dim=1))
        pred['update'] = update_score.argmax(dim=1)
        # not update mask
        not_update_mask = pred['update'] != 3
        if labels is not None:
            if self.args.balance:
                loss['update'] += F.cross_entropy(update_score, labels['update'], weight=torch.tensor([0.05, 0.35, 0.35, 2]).to(self.device))
            else:
                loss['update'] += F.cross_entropy(update_score, labels['update'])

        # input 3-d, mask 2-d, output 2-d (after unsqueeze), set unmasked outputs with -inf
        # module: nn.Linear, n -> 1
        def forward_with_mask(module, input, mask):
            assert input.shape[:-1] == mask.shape
            output = torch.zeros(*input.shape[:-1]).to(self.device)
            output[mask == 0] = -float("inf")
            if mask.sum() > 0:
                input = input[mask == 1]
                unfolded_output = module.forward(input)
                assert unfolded_output.shape[1] == 1
                output[mask == 1] = unfolded_output[:, 0]
            return output

        def get_hidden_cat_with(cat_with):
            if cat_with == 'cls':
                return cls_hidden[:, None, :].repeat(1, seq_len, 1)
            else:
                assert cat_with == 'slot'
                return slot_hidden[:, None, :].repeat(1, seq_len, 1)

        categorical_mask = value_mask.sum(dim=1) > 0
        value_hidden = torch.cat((get_hidden_cat_with(self.args.value_cat_with), hidden), dim=2)
        value_score = forward_with_mask(self.value_project, value_hidden, value_mask)
        pred['value'] = value_score.argmax(dim=1)
        # pred['value'][~categorical_mask | not_update_mask] = -1
        # pred['value'][~categorical_mask] = -1

        if labels is not None:
            # ignore -1 that is not updated
            loss['value'] += F.cross_entropy(
                value_score,
                labels['value'],
                ignore_index=-1,
            )

        def process_span(span):
            span_hidden = torch.cat((get_hidden_cat_with(self.args.span_cat_with), hidden), dim=2)
            span_score = forward_with_mask(self.span_project[span], span_hidden, span_mask[span])
            pred[span] = span_score.argmax(dim=1)
            # pred[span][categorical_mask | not_update_mask] = -1
            # pred[span][categorical_mask] = -1

            if labels is not None:
                loss[span] += F.cross_entropy(
                    span_score,
                    labels[span],
                    ignore_index=-1,
                )

        process_span('start')
        process_span('end')

        loss['tot'] = self.args.w_update * loss['update'] + self.args.w_value * loss['value'] + self.args.w_span * (
                    loss['start'] + loss['end'])

        return pred, loss

    def test(self, test_loader, eval_config, error_dir=None):
        self.eval()
        
        # val
        all_val_slot_ids = []
        all_val_preds = {"update": [], "value": [], "start": [], "end": []}
        all_val_labels = {"update": [], "value": [], "start": [], "end": []}
        losses = {'tot': 0, 'update': 0, 'value': 0, 'start': 0, 'end': 0}
        eval_steps = 0

        # dst
        all_dst_outputs = []
        all_dst_labels = {
            "origin": [],
            "fixed": []
        }

        with torch.no_grad():
            for ids, inputs, domain_pos, slot_pos, value_mask, span_mask, state_labels, labels in tqdm(test_loader):
                outputs, loss = self.forward(inputs, domain_pos, slot_pos, value_mask, span_mask, labels=labels)
                _, utt_id, slot_id = ids

                # for val
                all_val_slot_ids.extend(slot_id)
                for k, v in outputs.items():
                    all_val_preds[k].extend(v.detach().cpu().numpy().tolist())
                for k, v in labels.items():
                    all_val_labels[k].extend(v.detach().cpu().numpy().tolist())

                for key in loss:
                    losses[key] += loss[key].item()
                eval_steps += 1

                # for test
                outputs["inputs"] = inputs["input_ids"]
                outputs["value_mask"] = value_mask
                outputs["update_truth"] = labels["update"]
                outputs["value_label"] = labels["value"]
                outputs["start_label"] = labels["start"]
                outputs["end_label"] = labels["end"]
                outputs = {k: v.detach().cpu().numpy().tolist() for k, v in outputs.items()}
                outputs = [{k: v[i] for k, v in outputs.items()} for i in range(len(slot_id))]
                for i, (uid, sid, output) in enumerate(zip(utt_id, slot_id, outputs)):
                    if sid == 0:
                        if uid == 0:
                            # a new dialog
                            all_dst_outputs.append([])
                            for k in all_dst_labels:
                                all_dst_labels[k].append([])
                        # a new utterance
                        all_dst_outputs[-1].append([])
                        for k in all_dst_labels:
                            all_dst_labels[k][-1].append([])
                    all_dst_outputs[-1][-1].append(output)
                    for k in all_dst_labels:
                        all_dst_labels[k][-1][-1].append(state_labels[k][i])

        if error_dir is not None:
            cal_error(all_dst_outputs, all_dst_labels["fixed"], self.tokenizer, self.all_values, self.id2slot, self.ontology, error_dir)

        dst_res = self._test_dst(all_dst_outputs, all_dst_labels, eval_config)
        val_res = self._test_val(all_val_preds, all_val_labels, all_val_slot_ids, losses, eval_steps)

        return dst_res, val_res

    def _test_dst(self, all_outputs, all_labels, eval_config):
        accs = {}
        for k, v in eval_config.items():
            all_states = []
            all_upd_res = []
            err_num = 0
            err_only_num = {
                "clear": 0,
                "dontcare": 0
            }
            for pred, label in zip(all_outputs, all_labels[v["label_type"]]):
                states, upd_res, upd_err = add_states(label, pred, self.tokenizer, self.all_values, self.id2slot, self.ontology, split=False, prev_truth=v["prev_truth"], upd_truth=v["upd_truth"], value_truth=v["value_truth"])
                all_states.append(states)
                all_upd_res.append(upd_res)
                for upds in upd_err:
                    if len(upds) > 0:
                        err_num += 1
                        if len(upds) == 1:
                            if upds[0] == 2:
                                err_only_num["clear"] += 1
                            if upds[0] == 1:
                                err_only_num["dontcare"] += 1


            joint_acc, slot_acc = cal_added_scores(all_states, all_labels[v["label_type"]], all_upd_res, self.slot_num)
        
            accs[k] = {
                "joint_acc": joint_acc,
                "slot_acc": slot_acc,
                "join_upd_acc": sum([sum(x) for x in all_upd_res]) / sum([len(x) for x in all_upd_res]),
                "err_num": err_num,
                "err_only_num": err_only_num
            }
        
        return accs
    
    def _test_val(self, all_preds, all_labels, all_slot_ids, losses, eval_steps):
        results = {}
        results["update"] = cal_scores(all_preds["update"], all_labels["update"])
        value_preds, value_labels = [], []
        start_preds, start_labels = [], []
        end_preds, end_labels = [], []
        
        results["loss"] = {}
        for k, v in losses.items():
            results["loss"][k] = v / eval_steps

        for s, u, p, l in zip(all_slot_ids, all_labels["update"], all_preds["value"], all_labels["value"]):
            domain_name, slot_name = self.id2slot[s]
            if u == 3 and self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
                if l >= 0:
                    value_preds.append(p)
                    value_labels.append(l)
                else:
                    print("l < 0, may be a bug")
        results["value"] = cal_scores(value_preds, value_labels, only_acc=True)
        
        for s, u, p, l in zip(all_slot_ids, all_labels["update"], all_preds["start"], all_labels["start"]):
            domain_name, slot_name = self.id2slot[s]
            if u == 3 and not self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
                if l >= 0:
                    # l < 0 means can not find a span
                    start_preds.append(p)
                    start_labels.append(l)
        results["start"] = cal_scores(start_preds, start_labels, only_acc=True)

        for s, u, p, l in zip(all_slot_ids, all_labels["update"], all_preds["end"], all_labels["end"]):
            domain_name, slot_name = self.id2slot[s]
            if u == 3 and not self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
                if l >= 0:
                    # l < 0 means can not find a span
                    end_preds.append(p)
                    end_labels.append(l)
        results["end"] = cal_scores(end_preds, end_labels, only_acc=True)

        span_preds = [(s, e) for s, e in zip(start_preds, end_preds)]
        span_labels = [(s, e) for s, e in zip(start_labels, end_labels)]

        results["span"] = sum([p == l for p, l in zip(span_preds, span_labels)]) / len(span_preds)

        results["update_matrix"] = cal_matrix(all_preds["update"], all_labels["update"])

        return results


    def update(self, utt_inputs, utt_cands):
        preds, _ = self.forward(utt_inputs.unsqueeze(0), [utt_cands])
        for slot_id, (domain_name, slot_name) in enumerate(self.id2slot):
            update_type = preds["update"][0, slot_id]
            update_value = ""
            if update_type == 1:
                update_value = "dontcare"
            elif update_type == 2:
                update_value = ""
            elif update_type == 3:
                if self.ontology['domains'][domain_name]['slots'][slot_name]["is_categorical"]:
                    update_value = self.all_values[slot_id][preds["value"][0, slot_id]]
                else:
                    start = preds["start"][0, slot_id]
                    end = preds["end"][0, slot_id]
                    tokens = utt_inputs[slot_id][start: end]
                    update_value = self.tokenizer.decode(tokens).strip()
            
            self.states[slot_id] = update_value

        return self.states
