import torch
import numpy as np
import torch.utils.data as data
from .utils_function import to_cuda, merge, merge_multi_response, merge_sent_and_word

# SLOT_GATE = {"ptr":0, "dontcare":1, "none":2}

class Dataset_dst(torch.utils.data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, tokenizer, args, unified_meta, mode, max_length=512):
        """Reads source and target sequences from txt files."""
        self.data = data_info
        self.tokenizer = tokenizer
        self.num_total_seqs = len(data_info["ID"])
        self.usr_token = args["usr_token"]
        self.sys_token = args["sys_token"]
        self.max_length = max_length
        self.args = args
        self.unified_meta = unified_meta
        self.slots = list(unified_meta["slots"].keys())
        self.mask_token_idx = tokenizer.convert_tokens_to_ids("[MASK]")
        self.sep_token_idx = tokenizer.convert_tokens_to_ids("[SEP]")
        
        self.start_token = self.tokenizer.cls_token if "bert" in self.args["model_type"] else self.tokenizer.bos_token
        self.sep_token = self.tokenizer.sep_token if "bert" in self.args["model_type"] else self.tokenizer.eos_token

        self.all_num = 0
        self.large_turn_nums = {}

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        
        if self.args["example_type"] == "turn":
            dialog_history = self.data["dialog_history"][index]
            if len(dialog_history) > 0 and len(dialog_history[0]) == 0:
                dialog_history = dialog_history[1:]
            context = dialog_history + [self.data["turn_sys"][index], self.data["turn_usr"][index]]
            # context_plain = " ".join(context)
            reversed_context = list(reversed(context))
            # reverse_context_plain = " ".join(reversed_context)
            encoding = self.preprocess(reversed_context)

            gate_label = self.data["slot_gate"][index]
            slot_values_plain = self.data["slot_values"][index]
            slot_values = self.preprocess_slot(slot_values_plain)

            triggered_domains = set([domain_slot.split("-")[0] for domain_slot in self.data["belief"][index].keys()])
            triggered_domains.add(self.data["turn_domain"][index])
            assert len(triggered_domains) != 0
            
            triggered_ds_mask = [1 if s.split("-")[0] in triggered_domains else 0 for s in self.slots]
            triggered_ds_idx = []
            triggered_ds_pos = []
            
            ontology_idx = []
            for si, sv in enumerate(slot_values_plain):
                try:
                    ontology_idx.append(self.unified_meta["slots"][self.slots[si]][sv])
                except Exception as e:
                    print("Not In Ontology")
                    print(e)
                    print(self.slots[si], sv)
                    ontology_idx.append(-1)
            
        elif self.args["example_type"] == "dial":
            raise NotImplementedError()
        
        assert len(encoding["input_ids"]) <= self.max_length, "encoding: {}, max_length: {}".format(
            len(encoding["input_ids"]), self.max_length)

        item_info = {
            "ID":self.data["ID"][index],
            "del_belief":self.data["del_belief"][index], 
            "slot_gate":gate_label, 
            "context":torch.Tensor(encoding["input_ids"]), 
            "turn_id":torch.Tensor(encoding["turn_ids"]),
            "role_id":torch.Tensor(encoding["role_ids"]),
            "position_id":torch.Tensor(encoding["position_ids"]),
            "attention_mask":torch.Tensor(encoding["attention_mask"]),
            # "context_plain":context_plain,
            # "reversed_context_plain":reverse_context_plain,
            "slot_values":slot_values,
            "belief":self.data["belief"][index],
            "slots":self.data["slots"][index],
            "belief_ontology":ontology_idx,
            "triggered_ds_mask":triggered_ds_mask,
            "triggered_ds_idx":triggered_ds_idx,
            "triggered_ds_pos":triggered_ds_pos}

        # if "dialog" in self.args["model_type"]:
            # position_ids = self.tokenizer.create_positional_ids_from_sequences(context)
            # _t, _r = self.tokenizer.create_token_type_ids_from_sequences(context)
            # 
            # self.all_num += 1
            # m = max(_t)
            # if str(m + 1) in self.large_turn_nums:
                # self.large_turn_nums[str(m + 1)] += 1
            # else:
                # self.large_turn_nums[str(m + 1)] = 1
            # 
            # item_info["turn_id"] = torch.LongTensor(_t)
            # item_info["role_id"] = torch.LongTensor(_r)
            # item_info["position_id"] = torch.LongTensor(position_ids)

        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def concat_dh_sys_usr(self, dialog_history, sys, usr):
        return dialog_history + " {} ".format(self.sys_token) + sys + " {} ".format(self.sep_token) + " {} ".format(self.usr_token) + usr + " {} ".format(self.sep_token)

    def preprocess(self, sequence):
        """Converts words to ids."""
        # story = torch.Tensor(self.tokenizer.encode(sequence))
        tokens = [self.tokenizer.tokenize(s) for s in sequence]
        input_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]
        encoding = self.tokenizer.prepare_for_model(input_ids, max_length=self.max_length)
        return encoding

    def preprocess_slot(self, sequence):
        """Converts words to ids."""
        story = []
        for value in sequence:
            if "dialog" in self.args["model_type"]:
                tokens = self.tokenizer.tokenize(value + " {}".format(self.sep_token), add_special_tokens=True)
                # tokens = [self.tokenizer.convert_tokens_to_ids(tokens)]
            else:
                tokens = value + " {}".format(self.sep_token)
            v = list(self.tokenizer.encode(tokens))
            story.append(v)
        return story
    
    def get_concat_context(self, dialog_history):
        dialog_history_str = ""
        for ui, uttr in enumerate(dialog_history):
            if ui % 2 == 0:
                dialog_history_str += "{} {} {}".format(self.sys_token, uttr, self.sep_token)
            else:
                dialog_history_str += "{} {} {}".format(self.usr_token, uttr, self.sep_token)
        dialog_history_str = dialog_history_str.strip()
        return dialog_history_str

    def collate_fn(self, data):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['context']), reverse=True)

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        src_seqs, src_lengths = merge(item_info['context'])

        y_seqs, y_lengths = merge_multi_response(item_info["slot_values"])
        gates = torch.tensor(item_info["slot_gate"])
        belief_ontology = torch.tensor(item_info["belief_ontology"])
        triggered_ds_mask = torch.tensor(item_info["triggered_ds_mask"])

        item_info["context"] = to_cuda(src_seqs)
        item_info["context_len"] = src_lengths
        item_info["slot_gate"] = to_cuda(gates)
        item_info["slot_values"] = to_cuda(y_seqs)
        item_info["slot_values_len"] = y_lengths
        item_info["belief_ontology"] = to_cuda(belief_ontology)
        item_info["triggered_ds_mask"] = to_cuda(triggered_ds_mask)

        if 'turn_id' in item_info and 'role_id' in item_info and 'position_id' in item_info:
            turn_id, turn_id_lengths = merge(item_info['turn_id'])
            role_id, role_id_lengths = merge(item_info['role_id'])
            position_id, position_id_lengths = merge(item_info['position_id'])

            assert src_lengths == turn_id_lengths
            assert src_lengths == role_id_lengths
            assert src_lengths == position_id_lengths

            item_info["turn_id"] = to_cuda(turn_id)
            item_info["role_id"] = to_cuda(role_id)
            item_info["position_id"] = to_cuda(position_id)

        return item_info


def collate_fn_dst_dial(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge_sent_and_word(item_info['context'])
    y = [merge_multi_response(sv) for sv in item_info["slot_values"]]
    y_seqs = [_y[0] for _y in y]
    y_lengths = [_y[1] for _y in y]
    gates, gate_lengths = merge_sent_and_word(item_info['slot_gate'], ignore_idx=-1)
    belief_ontology = torch.tensor(item_info["belief_ontology"])
    
    item_info["context"] = to_cuda(src_seqs)
    item_info["context_len"] = src_lengths
    item_info["slot_gate"] = to_cuda(gates)
    item_info["slot_values"] = [to_cuda(y) for y in y_seqs] # TODO
    item_info["slot_values_len"] = y_lengths # TODO
    
    return item_info
