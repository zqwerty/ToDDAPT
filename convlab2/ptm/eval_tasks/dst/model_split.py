"""
    DST model based on pre-training BERT
"""

__all__ = [
    'DstModelSplit'
]

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from convlab2.ptm.eval_tasks.dst import parser, get_transformer_settings
from convlab2.ptm.eval_tasks.dst.utils import cal_scores, cal_added_scores, cal_matrix, add_states, cal_error


class DstModelSplit(nn.Module):
    def __init__(self, args, ontology, tokenizer=None):
        super().__init__()
        self.args = args
        self.ontology = ontology
        self.device = args.device

        tokenizer_cls, model_cls, config_cls = get_transformer_settings(args.transformer)
        if tokenizer is None:
            tokenizer = tokenizer_cls.from_pretrained(args.transformer_path)
        self.tokenizer = tokenizer
        self.transformer = model_cls.from_pretrained(args.transformer_path)
        hidden_size = self.transformer.config.hidden_size
        max_seq_len = self.transformer.config.max_position_embeddings

        self.slot2id = {}
        self.id2slot = []
        self.slot_num = 0
        self.all_values = []
        # only consider slots that tracked in state

        # prepare slots inputs
        slots_desc_input_ids = []
        slots_desc_turn_ids = []
        slots_desc_role_ids = []
        slots_desc_position_ids = []
        for domain_name, domain_state in self.ontology['state'].items():
            domain_def = self.ontology['domains'][domain_name]
            for slot_name, slot_state in domain_state.items():
                slot_def = domain_def['slots'][slot_name]
                self.slot2id[(domain_name, slot_name)] = self.slot_num
                self.id2slot.append((domain_name, slot_name))
                self.slot_num += 1
                self.all_values.append(slot_state['possible_values'])
                desc_text = ' '.join([
                    f"[CLS]",
                    f"[DOMAIN] {domain_def['description']} [SEP]",
                    f"[SLOT] {slot_def['description']} [SEP]",
                ] + [
                    f"[VALUE] {value} [SEP]" for value in slot_state['possible_values']
                ])
                desc_tokens = tokenizer.tokenize(desc_text)
                desc_input_ids = tokenizer.convert_tokens_to_ids(desc_tokens)
                slots_desc_input_ids.append(desc_input_ids[:max_seq_len])
        print(f'slot num: {self.slot_num}')
        self.desc_seq_len = max(len(x) for x in slots_desc_input_ids)
        for i, desc_input_ids in enumerate(slots_desc_input_ids):
            slots_desc_input_ids[i] += (self.desc_seq_len - len(desc_input_ids)) * [tokenizer.pad_token_id]
        slots_desc_input_ids = torch.tensor(slots_desc_input_ids).to(self.device)

        if args.transformer == "dialog-bert":
            for desc_input_ids in slots_desc_input_ids:
                desc_turn_ids, desc_role_ids = self.tokenizer.create_token_type_ids_from_sequences(desc_input_ids)
                desc_position_ids = self.tokenizer.create_positional_ids_from_sequences(desc_input_ids)
                slots_desc_turn_ids.append(desc_turn_ids)
                slots_desc_role_ids.append(desc_role_ids)
                slots_desc_position_ids.append(desc_position_ids)
            slots_desc_turn_ids = torch.tensor(slots_desc_turn_ids).to(self.device)
            slots_desc_role_ids = torch.tensor(slots_desc_role_ids).to(self.device)
            slots_desc_position_ids = torch.tensor(slots_desc_position_ids).to(self.device)

        special_token_ids = {
            token: tokenizer.convert_tokens_to_ids(f'[{token.upper()}]')
            for token in ['domain', 'slot', 'value']
        }
        for token_id in special_token_ids:
            assert token_id != tokenizer.unk_token_id
        self.domain_pos = (slots_desc_input_ids == special_token_ids['domain']).nonzero()[:, 1]
        self.slot_pos = (slots_desc_input_ids == special_token_ids['slot']).nonzero()[:, 1]
        assert len(self.domain_pos) == self.slot_num
        assert len(self.slot_pos) == self.slot_num
        self.value_mask = slots_desc_input_ids == special_token_ids['value']
        self.value_num = self.value_mask.sum(dim=1)

        if args.transformer == "dialog-bert":
            desc_turn_ids, desc_role_ids = self.tokenizer.create_token_type_ids_from_sequences(slots_desc_input_ids)
            desc_position_ids = self.tokenizer.create_positional_ids_from_sequences(slots_desc_input_ids)
            self.slots_desc_inputs = {
                'input_ids': slots_desc_input_ids,
                'attention_mask': slots_desc_input_ids != tokenizer.pad_token_id,
                "turn_ids": slots_desc_turn_ids,
                "role_ids": slots_desc_role_ids,
                "position_ids": slots_desc_position_ids,
            }
        else:
            self.slots_desc_inputs = {
                'input_ids': slots_desc_input_ids,
                'attention_mask': slots_desc_input_ids != tokenizer.pad_token_id
            }

        self.update_project = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4)
        )

        self.value_project = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        self.span_project = nn.ModuleDict({
            span: nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            )
            for span in ['start', 'end']
        })

        self.states = []
        self.init_states()

    def init_states(self):
        self.states = ["" for _ in range(len(self.slot2id))]

    def resize_token_embeddings(self, new_len):
        self.transformer.resize_token_embeddings(new_len)

    def forward(self, dialog_inputs, span_mask, slots_labels=None):
        r"""
        Parameters:
            dialog_inputs (dict, `required`): transformer input of dialog history
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `required`):
                    the token ids of current dialogue history:
                        [CLS] [USR] {U_t} [SEP] [SYS] {S_{t-1}} [SEP] [USR] {U_{t-1}} [SEP] ...

                attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`)

                position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`)

                role_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`):
                    available when using `DialogBert`

                turn_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, `optional`)
                    available when using `DialogBert`

            TODO: can span mask be specified for each slot?
            span_mask (dict, `required`):
                start (`torch.LongTensor` of shape `batch_size, sequence_length)`):
                    marked tokens which are the candidates of the start token

                end (`torch.LongTensor` of shape `batch_size, sequence_length)`):
                    marked tokens which are the candidates of the end token

            slots_labels (dict, `optional`): label dict
                update (`torch.LongTensor` of shape `(batch_size, slot_num)`):
                    update label for all slots, possible values are in {0, 1, 2, 3}, respectively denotes:
                        0: keep the previous value
                        1: update the value of the slot with special value of `dontcare`
                        2: update the value of the slot by clearing, i.e. set it to an empty value of `""`
                        3: update the value of the slot by a classifier (for categorical slots) or extracting it
                            it from the utterance according to the span (for non-categorical slots)

                value ((`torch.LongTensor` of shape `(batch_size, slot_num)`)):
                    update value index (in `possible_values` field defined in ontology),
                    making sense when the slot is categorical and the update label is 3,
                    set to -1 when the slot is non-categorical or is not updated

                start (`torch.LongTensor` of shape `(batch_size, slot_num)`):
                    start token index label, making sense when the slot is non-categorical
                    and the update label is 3, must be masked by span_mask['start'],
                    set to -1 when the slot is categorical or is not updated

                end (`torch.LongTensor` of shape `(batch_size, slot_num)`):
                    end token index label, making sense when the slot is non-categorical
                    and the update label is 3, must be masked by span_mask['end'],
                    set to -1 when the slot is categorical or is not updated

        Returns:
            slots_pred (dict): prediction dict
                similar almost the same definition as `label` paratewith 4 fields,:
                - update
                - value
                - start
                - end

            slots_loss (`torch.Tensor` of shape `(slot_num)`): loss for each slot (mean by batch)
                making sense when `label` is not `None`
        """

        # modify later
        dialog_inputs = deepcopy(dialog_inputs)
        slots_labels = deepcopy(slots_labels)
        dialog_input_ids = dialog_inputs['input_ids']
        batch_size = dialog_input_ids.shape[0]
        dialog_seq_len = dialog_input_ids.shape[1]

        slots_desc_hidden = self.transformer.forward(**self.slots_desc_inputs)[0]
        turn_hidden = self.transformer.forward(**dialog_inputs)[0]

        """
            input 3-d, mask 2-d, output 2-d (after unsqueeze), set unmasked outputs with -inf
            score_module: nn.Linear, n -> 1
        """
        def forward_with_mask(score_module, input, mask):
            assert input.shape[:-1] == mask.shape
            output = torch.zeros(*input.shape[:-1]).to(self.device)
            output[mask == 0] = -float("inf")
            if mask.sum() > 0:
                input = input[mask == 1]
                unfolded_output = score_module.forward(input)
                assert unfolded_output.shape[1] == 1
                output[mask == 1] = unfolded_output[:, 0]
            return output

        # collect prediction and loss for each slot
        slots_pred = []
        slots_loss = []
        for slot_id, (desc_hidden, domain_pos, slot_pos, value_mask, value_num) in enumerate(
            zip(
                slots_desc_hidden,
                self.domain_pos,
                self.slot_pos,
                self.value_mask,
                self.value_num,
            )
        ):
            _domain_hidden = desc_hidden[domain_pos, :]
            _slot_hidden = slots_desc_hidden[slot_pos, :]
            labels = None if slots_labels is None else {
                k: v[:, slot_id]
                for k, v in slots_labels.items()
            }
            pred = {
                k: torch.full((batch_size,), -1, dtype=torch.long).to(self.device)
                for k in ['update', 'value', 'start', 'end']
            }
            loss = {
                k: torch.tensor(0.0).to(self.device)
                for k in ['update', 'value', 'start', 'end']
            }

            update_score = self.update_project.forward(
                torch.cat(
                    (
                        desc_hidden[0, :].repeat(batch_size, 1),
                        turn_hidden[:, 0, :],
                    ),
                    dim=1
                ),
            )
            pred['update'] = update_score.argmax(dim=1)
            if labels is not None:
                if self.args.balance:
                    loss['update'] += F.cross_entropy(update_score, labels['update'], weight=torch.tensor([0.05, 0.35, 0.35, 0.25]).to(self.device))
                else:
                    loss['update'] += F.cross_entropy(update_score, labels['update'])

            if value_num > 0:
                value_score = self.value_project.forward(
                    torch.cat(
                        (
                            desc_hidden[0, :].repeat(batch_size, value_num, 1),
                            desc_hidden[value_mask, :].repeat(batch_size, 1, 1),
                            turn_hidden[:, 0, :].unsqueeze(1).repeat(1, value_num, 1)
                        ),
                        dim=2,
                    ),
                )[:, :, 0]
                pred['value'] = value_score.argmax(dim=1)
                if labels is not None:
                    # ignore -1 that is not updated
                    loss['value'] += F.cross_entropy(
                        value_score,
                        labels['value'],
                        ignore_index=-1,
                    )

            def process_span(span):
                span_score = forward_with_mask(
                    self.span_project[span],
                    torch.cat(
                        (
                            desc_hidden[0, :].repeat(batch_size, dialog_seq_len, 1),
                            turn_hidden[:, 0, :].unsqueeze(1).repeat(1, dialog_seq_len, 1),
                            turn_hidden,
                        ),
                        dim=2,
                    ),
                    span_mask[span]
                )
                pred[span] = span_score.argmax(dim=1)

                if labels is not None:
                    loss[span] += F.cross_entropy(
                        span_score,
                        labels[span],
                        ignore_index=-1,
                    )

            process_span('start')
            process_span('end')

            loss['tot'] = self.args.w_update * loss['update'] \
                        + self.args.w_value * loss['value'] \
                        + self.args.w_span * (loss['start'] + loss['end'])
            slots_pred.append(pred)
            slots_loss.append(loss)

        slots_pred = {
            k: torch.stack([pred[k] for pred in slots_pred], dim=1)
            for k in ['update', 'value', 'start', 'end']
        }
        slots_loss = {
            k: torch.stack([loss[k] for loss in slots_loss])
            for k in ['update', 'value', 'start', 'end', 'tot']
        }

        return slots_pred, slots_loss

    def test(self, test_loader, eval_config, error_dir=None):
        self.eval()

        # val
        # all_val_slot_ids = []
        # all_val_preds = {"update": [], "value": [], "start": [], "end": []}
        # all_val_labels = {"update": [], "value": [], "start": [], "end": []}
        all_val_preds = {k: [[] for _ in range(self.slot_num)] for k in ['update', 'value', 'start', 'end']}
        all_val_labels = {k: [[] for _ in range(self.slot_num)] for k in ['update', 'value', 'start', 'end']}
        losses = {'tot': 0, 'update': 0, 'value': 0, 'start': 0, 'end': 0}
        eval_steps = 0

        # dst
        all_dst_outputs = []
        all_dst_labels = {
            "origin": [],
            "fixed": []
        }

        with torch.no_grad():
            # for ids, inputs, domain_pos, slot_pos, value_mask, span_mask, state_labels, labels in tqdm(test_loader):
            for ids, inputs, span_mask, state_labels, labels in tqdm(test_loader):
                outputs, loss = self.forward(inputs, span_mask, labels)
                # _, utt_id, slot_id = ids
                _, utt_id = ids
                batch_size = len(utt_id)

                # for val
                # all_val_slot_ids.extend(slot_id)
                # for k, v in outputs.items():
                #     all_val_preds[k].extend(v.detach().cpu().numpy().tolist())
                # for k, v in labels.items():
                #     all_val_labels[k].extend(v.detach().cpu().numpy().tolist())

                for all_val, array in zip([all_val_preds, all_val_labels], [outputs, labels]):
                    for k, v in array.items():
                        for slot_id in range(self.slot_num):
                            all_val[k][slot_id].extend(v[:, slot_id].tolist())

                for key in loss:
                    losses[key] += loss[key].sum()
                eval_steps += 1

                # for test
                outputs["inputs"] = inputs["input_ids"]
                # outputs["value_mask"] = value_mask
                outputs["update_truth"] = labels["update"]
                outputs["value_label"] = labels["value"]
                outputs["start_label"] = labels["start"]
                outputs["end_label"] = labels["end"]

                outputs = {k: v.detach().cpu().numpy().tolist() for k, v in outputs.items()}
                # outputs = [{k: v[i] for k, v in outputs.items()} for i in range(len(slot_id))]
                outputs = [{k: v[i] for k, v in outputs.items()} for i in range(batch_size)]

                for bid, (uid, output) in enumerate(zip(utt_id, outputs)):
                    if uid == 0:
                        # a new dialog
                        all_dst_outputs.append([])
                        for k in all_dst_labels:
                            all_dst_labels[k].append([])

                    # # a new utterance
                    # all_dst_outputs[-1].append([])
                    # for k in all_dst_labels:
                    #     all_dst_labels[k][-1].append([])
                    # for sid in range(self.slot_num):
                    #     all_dst_outputs[-1][-1].append(output)
                    #     for k in all_dst_labels:
                    #         all_dst_labels[k][-1][-1].append(state_labels[k][i])
                    all_dst_outputs[-1].append([
                        {k: v if k == "inputs" else v[slot_id] for k, v in output.items()}
                        for slot_id in range(self.slot_num)
                    ])
                    for k in all_dst_labels:
                        all_dst_labels[k][-1].append([state_labels[k][bid][f'{domain_name}-{slot_name}'] for domain_name, slot_name in self.id2slot])

        if error_dir is not None:
            cal_error(all_dst_outputs, all_dst_labels["fixed"], self.tokenizer, self.all_values, self.id2slot, self.ontology, error_dir, split=True)

        dst_res = self._test_dst(all_dst_outputs, all_dst_labels, eval_config)
        val_res = self._test_val(all_val_preds, all_val_labels, losses, eval_steps)

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
                states, upd_res, upd_err = add_states(label, pred, self.tokenizer, self.all_values, self.id2slot, self.ontology, split=True, prev_truth=v["prev_truth"], upd_truth=v["upd_truth"], value_truth=v["value_truth"])
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

    def _test_val(self, all_preds, all_labels, losses, eval_steps):
        # for k, v in all_preds.items():
        #     all_preds[k] = np.array(v)
        # for k, v in all_labels.items():
        #     all_labels[k] = np.array(v)
        for array in [all_preds, all_labels]:
            for k, v in array.items():
                array[k] = np.array(v)

        results = {"update": cal_scores(all_preds["update"].flatten(), all_labels["update"].flatten())}
        value_preds, value_labels = [], []
        # start_preds, start_labels = [], []
        # end_preds, end_labels = [], []
        span_preds = {'start': [], 'end': []}
        span_labels = {'start': [], 'end': []}

        results["loss"] = {}
        for k, v in losses.items():
            results["loss"][k] = v / eval_steps

        # for s, u, p, l in zip(all_slot_ids, all_labels["update"], all_preds["value"], all_labels["value"]):
        #     domain_name, slot_name = self.id2slot[s]
        #     if u == 3 and self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
        #         if l >= 0:
        #             value_preds.append(p)
        #             value_labels.append(l)
        #         else:
        #             print("l < 0, may be a bug")
        # results["value"] = cal_scores(value_preds, value_labels, only_acc=True)
        #
        # for s, u, p, l in zip(all_slot_ids, all_labels["update"], all_preds["start"], all_labels["start"]):
        #     domain_name, slot_name = self.id2slot[s]
        #     if u == 3 and not self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
        #         if l >= 0:
        #             # l < 0 means can not find a span
        #             start_preds.append(p)
        #             start_labels.append(l)
        # results["start"] = cal_scores(start_preds, start_labels, only_acc=True)
        #
        # for s, u, p, l in zip(all_slot_ids, all_labels["update"], all_preds["end"], all_labels["end"]):
        #     domain_name, slot_name = self.id2slot[s]
        #     if u == 3 and not self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
        #         if l >= 0:
        #             # l < 0 means can not find a span
        #             end_preds.append(p)
        #             end_labels.append(l)
        # results["end"] = cal_scores(end_preds, end_labels, only_acc=True)

        for slot_id, (domain_name, slot_name) in enumerate(self.id2slot):
            for u, p, l in zip(all_labels["update"][slot_id], all_preds["value"][slot_id], all_labels["value"][slot_id]):
                if u == 3 and self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
                    if l >= 0:
                        value_preds.append(p)
                        value_labels.append(l)
                    else:
                        print("l < 0, may be a bug")

        for span in ['start', 'end']:
            for slot_id, (domain_name, slot_name) in enumerate(self.id2slot):
                for u, p, l in zip(all_labels["update"][slot_id], all_preds[span][slot_id], all_labels[span][slot_id]):
                    if u == 3 and not self.ontology['domains'][domain_name]['slots'][slot_name]['is_categorical']:
                        if l >= 0:
                            # l < 0 means can not find a span
                            span_preds[span].append(p)
                            span_labels[span].append(l)
            results[span] = cal_scores(span_preds[span], span_labels[span], only_acc=True)

        span_preds = [(s, e) for s, e in zip(span_preds['start'], span_preds['end'])]
        span_labels = [(s, e) for s, e in zip(span_labels['start'], span_labels['end'])]

        results["span"] = sum([p == l for p, l in zip(span_preds, span_labels)]) / len(span_preds)

        results["update_matrix"] = cal_matrix(all_preds["update"].flatten(), all_labels["update"].flatten())

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
