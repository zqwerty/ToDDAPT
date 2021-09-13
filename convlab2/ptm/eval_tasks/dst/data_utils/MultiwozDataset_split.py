import json
import os
from zipfile import ZipFile

import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer
from convlab2.ptm.pretraining.model import DialogBertTokenizer

from convlab2.ptm.eval_tasks.dst import parser

__all__ = [
    'MultiwozDatasetSplit'
]


class MultiwozDatasetSplit(Dataset):
    def __init__(self, args, tokenizer, data_dir: str, split: str, evaluate=False, ratio=1, update_ratio=1.0):
        self.tokenizer = tokenizer
        self.split = split
        self.block_size = args.block_size
        self.transformer = args.transformer
        self.evaluate = evaluate
        self.device = args.device
        self.ratio = ratio
        self.update_ratio = update_ratio
        self.dataset = args.dataset
        print(data_dir)
        print("Loading Ontology")
        with open(os.path.join(data_dir, "ontology.json")) as f:
            self.ontology = self.preprocess_ontology(json.load(f))

        print("Loading Data")
        with ZipFile(os.path.join(data_dir, "data.zip")) as z:
            with z.open("data.json", "r") as f:
                all_data = json.load(f)

        print("Load Data End")
        self.data = {"train": [], "val": [], "test": []}
        for d in tqdm(all_data, desc="Spliting", ncols=80):
            self.data[d["data_split"]].append(d)
        self.cls_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
        self.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
        self.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
        self.usr_token_id = tokenizer.convert_tokens_to_ids("[USR]")
        self.sys_token_id = tokenizer.convert_tokens_to_ids("[SYS]")
        self.domain_token_id = tokenizer.convert_tokens_to_ids("[DOMAIN]")
        self.slot_token_id = tokenizer.convert_tokens_to_ids("[SLOT]")
        self.value_token_id = tokenizer.convert_tokens_to_ids("[VALUE]")

        print("Processing Domain Slot Lookup")
        self.domain_ids_lookup, self.slot_ids_lookup = self.preprocess_domain_slot(self.ontology)

        print("Processing Value Lookup")
        self.ctg_value_lookup, self.value_ids_lookup = self.preprocess_ctg_value(self.ontology)

        print("Processing Instances")
        # dial * utt * slot * token_num
        self.no_span_utt = 0
        self.utt_num = 0
        self.no_span_L = []
        self.seq_instances, self.int_instances, self.no_gpu_instance, self.offset = self.preprocess(self.data[split])

        # with open("seq_cache.pkl", "wb") as f:
        #     pickle.dump(self.seq_instances, f)

        # with open("int_cache.pkl", "wb") as f:
        #     pickle.dump(self.int_instances, f)

        # with open("no_gpu_cache.pkl", "wb") as f:
        #     pickle.dump(self.no_gpu_instance, f)

        self.pads = {
            "input_ids": self.pad_token_id,
            "turn_ids": 0,
            "role_ids": 0,
            "position_ids": 0,
            "attention_mask": 0,
            # "value_mask": 0,
            "start_cand": 0,
            "end_cand": 0
        }


    def preprocess_ontology(self, ontology):
        domains = ontology["domains"]
        state = ontology["state"]
        new_ontology = {}

        for domain in state:
            new_ontology[domain] = {"description": domains[domain]["description"], "slots": {}}
            for slot in state[domain]:
                new_ontology[domain]["slots"][slot] = state[domain][slot]

        return new_ontology
        
    def preprocess_domain_slot(self, ontology):
        domain_ids_lookup = {}
        slot_ids_lookup = {}
        for domain in ontology.keys():
            domain_desc = ontology[domain]["description"]
            domain_tokens = self.tokenizer.tokenize(domain) + self.tokenizer.tokenize(domain_desc)
            domain_ids = self.tokenizer.convert_tokens_to_ids(domain_tokens)
            domain_ids_lookup[domain] = domain_ids
            for slot in ontology[domain]["slots"].keys():
                slot_desc = ontology[domain]["slots"][slot]["description"]
                slot_tokens = self.tokenizer.tokenize(slot) + self.tokenizer.tokenize(slot_desc)
                slot_ids = self.tokenizer.convert_tokens_to_ids(slot_tokens)
                slot_ids_lookup[slot] = slot_ids

        return domain_ids_lookup, slot_ids_lookup

    def preprocess_ctg_value(self, ontology):
        """
        domain-slot {
            tokens: [],
            id: (0 for none)
        }
        """
        ctg_value_lookup = {}
        value_ids_lookup = {}
        for domain in ontology.keys():
            for slot in ontology[domain]["slots"]:
                i = 0
                if ontology[domain]["slots"][slot]["is_categorical"]:
                    value_ids, value_mask = [], []
                    ctg_value_lookup[domain + "-" + slot] = {}
                    for value in ontology[domain]["slots"][slot]["possible_values"]:
                        value_tokens = self.tokenizer.tokenize(value)
                        ctg_value_lookup[domain + "-" + slot][value] = {
                            "tokens": value_tokens,
                            "id": i
                        }
                        value_ids += [self.value_token_id] + self.tokenizer.convert_tokens_to_ids(value_tokens) + [self.sep_token_id]
                        i += 1
                    value_ids_lookup[domain + "-" + slot] = value_ids
        return ctg_value_lookup, value_ids_lookup

    def preprocess_one_turn(self, turn, history_ids, history_utts):
        context_utts = history_utts + [turn["utterance"]]
        cur_utt_tokens = self.tokenizer.tokenize(turn["utterance"])
        cur_utt_ids = self.tokenizer.convert_tokens_to_ids(cur_utt_tokens)
        context_ids = history_ids + [cur_utt_ids]
        context_ids_r = list(reversed(context_ids))
        dial_ids = []
        for i, utt_ids in enumerate(context_ids_r):
            if i % 2 == 0:
                dial_ids.append([self.usr_token_id] + utt_ids + [self.sep_token_id])
            else:
                dial_ids.append([self.sys_token_id] + utt_ids + [self.sep_token_id])

        state_labels = {
            "fixed_labels": {},
            "labels": {}
        }

        for domain in self.ontology.keys():
            for slot in self.ontology[domain]["slots"].keys():
                state_labels["labels"][domain + "-" + slot] = ""
                state_labels["fixed_labels"][domain + "-" + slot] = ""

        for domain in turn["state"].keys():
            for slot, value in turn["state"][domain].items():
                state_labels["labels"][domain + "-" + slot] = "".join(value.strip().split(" "))
                state_labels["fixed_labels"][domain + "-" + slot] = "".join(value.strip().split(" "))

        for domain in turn["fixed_state"].keys():
            for slot, value in turn["fixed_state"][domain].items():
                if value != "not found":
                    state_labels["fixed_labels"][domain + "-" + slot] = "".join(value.strip().split(" "))

        update_slot_ctg = {}
        for s in turn["state_update"]["categorical"]:
            update_slot_ctg[s["domain"] + "-" + s["slot"]] = {
                "value": s["value"]
            }
        
        update_slot_nctg = {}
        no_span = False
        for s in turn["state_update"]["non-categorical"]:
            if "utt_idx" in s:
                prev_ids_len = 0
                assert s["utt_idx"] < len(dial_ids)
                for k in range(len(dial_ids) - s["utt_idx"] - 1):
                    prev_ids_len += len(dial_ids[k])

                value_utt = context_utts[s["utt_idx"]]
                prev_value_len = len(self.tokenizer.tokenize(value_utt[:s["start"]])) + 1 # 1 for [USR] token
                value_len = len(self.tokenizer.tokenize(value_utt[s["start"]:s["end"]]))
                new_start = prev_ids_len + prev_value_len
                new_end = prev_ids_len + prev_value_len + value_len

                update_slot_nctg[s["domain"] + "-" + s["slot"]] = {
                    "value": s["fixed_value"] if "fixed_value" in s else s["value"],
                    "start": new_start,
                    "end": new_end,
                }
            else:
                assert "start" not in s
                assert "end" not in s
                update_slot_nctg[s["domain"] + "-" + s["slot"]] = {
                    "value": s["value"],
                }
                if s["value"] not in ["", "dontcare", "none", "not mentioned"]:
                    no_span = True

        dial_ids = [i for d in dial_ids for i in d]
        if no_span:
            self.no_span_L.append(turn)
            self.no_span_utt += 1
        self.utt_num += 1
        return dial_ids, context_ids, context_utts, update_slot_ctg, update_slot_nctg, state_labels
    
    def valid_start_cand(self, idx):
        b1 = self.tokenizer.convert_ids_to_tokens(idx).startswith("##")
        b2 = (idx == self.sep_token_id)
        b3 = (idx == self.usr_token_id)
        b4 = (idx == self.sys_token_id)
        return not (b1 or b2 or b3 or b4)

    def valid_end_cand(self, idx):
        b1 = self.tokenizer.convert_ids_to_tokens(idx).startswith("##")
        return not b1

    def preprocess(self, data):
        """
        [CLS] [domain] domain_name domain_desc [slot] slot_name slot_desc [USR] U_t [SEP] [SYS] S_t-1 [SEP] [USR] ..
        
        0: none
        1: clear
        2: dont care
        3: value
        
        """
        seq_instances = []
        int_instances = []
        no_gpu_instances = []
        did = 0
        offset = []
        print(self.ratio)
        if self.split == "train":
            random.shuffle(data)
        for dial in tqdm(data[:int(len(data) * self.ratio)], desc="Preprocessing", ncols=80):
            history_ids, history_utts = [], []
            seq_inst_dial = []
            int_inst_dial = []
            no_gpu_inst_dial = []
            uid = 0
            part_state_labels = {}
            for domain in self.ontology.keys():
                for slot in self.ontology[domain]["slots"].keys():
                    part_state_labels[domain + "-" + slot] = ""

            for turn in dial["turns"]:
                if turn["speaker"] == "system":
                    history_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["utterance"])))
                    history_utts.append(turn["utterance"])
                    continue

                dial_ids, history_ids, history_utts, update_slot_ctg, update_slot_nctg, state_labels = self.preprocess_one_turn(turn, history_ids, history_utts)
                
                dial_start_cands = []
                dial_end_cands = []
                for j in range(0, len(dial_ids)):
                    if self.valid_start_cand(dial_ids[j]):
                        dial_start_cands.append(1)
                    else:
                        dial_start_cands.append(0)
                    if self.valid_end_cand(dial_ids[j]):
                        dial_end_cands.append(1)
                    else:
                        dial_end_cands.append(0)
                # dial_start_cands = np.array(start_cands)
                # dial_end_cands = np.array(end_cands)

                seq_inst_slot = []
                int_inst_slot = []
                no_gpu_inst_slot = []
                prefix_ids = [self.cls_token_id]
                start_cands = [0 for _ in range(len(prefix_ids))] + dial_start_cands
                end_cands = [0 for _ in range(len(prefix_ids))] + dial_end_cands

                input_ids = prefix_ids + dial_ids
                if len(input_ids) > self.block_size:
                    input_ids = input_ids[:-(len(input_ids) - self.block_size) - 1] + input_ids[-1:]
                    start_cands = start_cands[:-(len(start_cands) - self.block_size) - 1] + start_cands[-1:]
                    end_cands = end_cands[:-(len(end_cands) - self.block_size) - 1] + end_cands[-1:]
                length = len(input_ids)

                turn_ids, role_ids, position_ids = [], [], []
                if self.transformer == "dialog-bert":
                    turn_ids, role_ids = self.tokenizer.create_token_type_ids_from_sequences(input_ids)
                    assert len(turn_ids) == length
                    assert len(role_ids) == length
                    position_ids = self.tokenizer.create_positional_ids_from_sequences(input_ids)
                    assert len(position_ids) == length
                        
                assert len(start_cands) == length
                assert len(end_cands) == length

                attn_mask = [1] * length

                utt_state_updates = []
                utt_ctg_labels = []
                utt_start_labels = []
                utt_end_labels = []

                for domain in self.ontology.keys():
                    for slot in self.ontology[domain]["slots"].keys():
                        state_update = 0
                        ctg_label = -1
                        start_label = -1
                        end_label = -1

                        name = domain + "-" + slot
                        # state_label = state_labels[name]
                        if name in update_slot_ctg:
                            if update_slot_ctg[name]["value"] == "dontcare":
                                state_update = 1
                                part_state_labels[name] = "dontcare"
                            elif update_slot_ctg[name]["value"] == "":
                                state_update = 2
                                part_state_labels[name] = ""
                            else:
                                state_update = 3
                                value = update_slot_ctg[name]["value"]
                                assert self.ontology[domain]['slots'][slot]['is_categorical']
                                if value in self.ctg_value_lookup[name]:
                                    part_state_labels[name] = value
                                    ctg_label = self.ctg_value_lookup[name][value]["id"]
                                    # ctg_label = [i for i, x in enumerate(value_mask) if x == 1][ctg_label]
                                else:
                                    print("ctg not found:", value)

                        if name in update_slot_nctg:
                            if update_slot_nctg[name]["value"] == "dontcare":
                                state_update = 1
                                part_state_labels[name] = "dontcare"
                            elif update_slot_nctg[name]["value"] == "":
                                state_update = 2
                                part_state_labels[name] = ""
                            else:
                                state_update = 3
                                if "start" in update_slot_nctg[name] and "end" in update_slot_nctg[name]:
                                    start_label = update_slot_nctg[name]["start"] + len(prefix_ids)
                                    end_label = update_slot_nctg[name]["end"] + len(prefix_ids)

                                    # check
                                    assert start_label < end_label
                                    if end_label >= self.block_size:
                                        start_label = -1
                                        end_label = -1
                                    else:
                                        value = "".join(update_slot_nctg[name]["value"].strip().split(" "))
                                        part_state_labels[name] = value
                                        v = input_ids[start_label:end_label]
                                        vv = "".join(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(v)).strip().split(" "))
                                        if value != vv:
                                            print(len(prefix_ids))
                                            print(value)
                                            print(self.tokenizer.convert_ids_to_tokens(v))
                                            print(self.tokenizer.convert_ids_to_tokens(input_ids))
                                            print(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(v)).strip())
                                            print(start_label, end_label)
                                            # ignore
                                            start_label = -1
                                            end_label = -1
                                        
                                            # exit(0)

                        # drop some samples with update id = 0 with probability of 1 - update_ratio
                        if state_update == 0 and np.random.ranf() >= self.update_ratio:
                            continue
                        
                        utt_state_updates.append(state_update)
                        utt_ctg_labels.append(ctg_label)
                        utt_start_labels.append(start_label)
                        utt_end_labels.append(end_label)

                seq_inst_dial.append({
                    "input_ids": input_ids,
                    "turn_ids": turn_ids,
                    "role_ids": role_ids,
                    "position_ids": position_ids,
                    "attention_mask": attn_mask,
                    "start_cand": start_cands,
                    "end_cand": end_cands,
                })

                int_inst_dial.append({
                    "dial_id": did,
                    "utt_id": uid,
                    "state_update": utt_state_updates,
                    "ctg_label": utt_ctg_labels,
                    "start_label": utt_start_labels,
                    "end_label": utt_end_labels,
                    "length": length,
                })

                no_gpu_inst_dial.append({
                    "part_state_label": part_state_labels,
                    "state_label": state_labels["labels"],
                    "fixed_state_label": state_labels["fixed_labels"]
                })

                offset.append((did, uid))

                uid += 1

            seq_instances.append(seq_inst_dial)
            int_instances.append(int_inst_dial)
            no_gpu_instances.append(no_gpu_inst_dial)
            did += 1

        return seq_instances, int_instances, no_gpu_instances, offset

    def __len__(self):
        if self.evaluate:
            return len(self.seq_instances)
        else:
            return len(self.offset)

    def __getitem__(self, item):
        if self.evaluate:
            return self.seq_instances[item], self.int_instances[item], self.no_gpu_instance[item]
        else:
            did, uid = self.offset[item]
            return self.seq_instances[did][uid], self.int_instances[did][uid], self.no_gpu_instance[did][uid]

    def collate(self, examples):
        insts = {}
        seq_inst = [e[0] for e in examples]
        int_inst = [e[1] for e in examples]
        no_gpu_inst = [e[2] for e in examples]
        max_length = max([x["length"] for x in int_inst])
        for key in seq_inst[0].keys():
            seq = [inst[key] + [self.pads[key]] * (max_length - len(inst[key])) for inst in seq_inst]
            insts[key] = torch.tensor(seq, dtype=torch.long)
        for key in int_inst[0].keys():
            insts[key] = torch.tensor([inst[key] for inst in int_inst], dtype=torch.long)

        if self.transformer == "bert":
            input_keys = ["input_ids", "attention_mask"]
        elif self.transformer == "dialog-bert":
            input_keys = ["input_ids", "attention_mask", "position_ids", "role_ids", "turn_ids"]
        else:
            raise NotImplementedError("Transformer Type Not Implemented")

        inputs = {key: insts[key] for key in input_keys}

        span_mask = {
            "start": insts["start_cand"],
            "end": insts["end_cand"]
        }

        labels = {
            "update": insts["state_update"],
            "value": insts["ctg_label"],
            "start": insts["start_label"],
            "end": insts["end_label"]
        }

        ids = (insts["dial_id"], insts["utt_id"])

        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        for k, v in span_mask.items():
            span_mask[k] = v.to(self.device)
        if labels is not None:
            for k, v in labels.items():
                labels[k] = v.to(self.device)

        return ids, inputs, span_mask, labels

    def collate_eval(self, examples):
        insts = {}
        seq_inst = [e[0] for e in examples]
        int_inst = [e[1] for e in examples]
        no_gpu_inst = [e[2] for e in examples]
        max_length = max([x["length"] for x in int_inst])
        for key in seq_inst[0].keys():
            seq = [inst[key] + [self.pads[key]] * (max_length - len(inst[key])) for inst in seq_inst]
            insts[key] = torch.tensor(seq, dtype=torch.long)
        for key in int_inst[0].keys():
            insts[key] = torch.tensor([inst[key] for inst in int_inst], dtype=torch.long)
        for key in no_gpu_inst[0].keys():
            insts[key] = [inst[key] for inst in no_gpu_inst]

        if self.transformer == "bert":
            input_keys = ["input_ids", "attention_mask"]
        elif self.transformer == "dialog-bert":
            input_keys = ["input_ids", "attention_mask", "position_ids", "role_ids", "turn_ids"]
        else:
            raise NotImplementedError("Transformer Type Not Implemented")

        # remove for split
        # domain_pos = insts["domain_pos"]
        # slot_pos = insts["slot_pos"]
        # value_mask = insts["value_mask"]

        inputs = {key: insts[key] for key in input_keys}

        span_mask = {
            "start": insts["start_cand"],
            "end": insts["end_cand"]
        }
        
        labels = {
            "update": insts["state_update"],
            "value": insts["ctg_label"],
            "start": insts["start_label"],
            "end": insts["end_label"]
        }

        # ids = (insts["dial_id"], insts["utt_id"], insts["slot_id"])
        ids = (insts["dial_id"], insts["utt_id"])

        state_labels = {
            "origin": insts["state_label"],
            "fixed": insts["fixed_state_label"]
            # "fixed": insts["part_state_label"]
        }
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        # value_mask = value_mask.to(self.device)
        for k, v in span_mask.items():
            span_mask[k] = v.to(self.device)
        if labels is not None:
            for k, v in labels.items():
                labels[k] = v.to(self.device)

        # return ids, inputs, domain_pos, slot_pos, value_mask, span_mask, state_labels, labels
        return ids, inputs, span_mask, state_labels, labels
