import os
import json
import random
import math
from convlab2.ptm.pretraining.dataloader import TaskProcessor, DialogBertSampler
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from convlab2.ptm.pretraining.schema_linking.processor import preprocess_schema
from pprint import pprint
from copy import deepcopy
from collections import Counter
from itertools import zip_longest
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import numpy as np
from tqdm import tqdm


class AugDialDataset(Dataset):
    def __init__(self, data, utt_pool, tokenizer, max_length, pos_aug_num=1, neg_aug_num=1, pick1utt_num=1,
                 clip_ori=False, clip_aug=True, keep_value=True):
        """load full dialogue"""
        self.utt_pool = utt_pool
        self.user_da_seq = list(self.utt_pool["user"].keys())
        self.user_da_cnt = [len(self.utt_pool["user"][x]) for x in self.user_da_seq]
        self.system_da_seq = list(self.utt_pool["system"].keys())
        self.system_da_cnt = [len(self.utt_pool["system"][x]) for x in self.system_da_seq]
        self.data, self.data_bucket_ids = [], []
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.pos_aug_num = pos_aug_num
        self.neg_aug_num = neg_aug_num
        self.pick1utt_num = pick1utt_num
        self.clip_ori = clip_ori
        self.clip_aug = clip_aug
        self.keep_value = keep_value
        for d in tqdm(data, desc="Loading dataset"):
            self.data.append(d)
            self.data_bucket_ids.append(0)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # clip the whole dial randomly [0, t]
        data_item = self.data[index]
        ori_dial = self.clip_dial(data_item, rand_start=False, rand_end=True)
        # TODO: mask a random utt
        outputs = [self.clip_dial(ori_dial, rand_start=True, rand_end=False) if self.clip_ori else ori_dial]
        for _ in range(self.pos_aug_num):
            if self.clip_aug:
                dial = self.clip_dial(ori_dial, rand_start=True, rand_end=False)
            else:
                dial = ori_dial
            outputs.append(self.augment(dial, positive=True))
        for _ in range(self.neg_aug_num):
            if self.clip_aug:
                dial = self.clip_dial(ori_dial, rand_start=True, rand_end=False)
            else:
                dial = ori_dial
            outputs.append(self.augment(dial, positive=False))
        for _ in range(self.pick1utt_num):
            outputs.append(self.augment(self.pick1utt(ori_dial), positive=True))

        return outputs  # [ori_dial, pos_dial1, ..., neg_dial1, ...]

    def clip_dial(self, dial, rand_start=False, rand_end=False):
        if rand_end:
            end = random.choice(range(1, dial["num_utt"] + 1, 2))  # select user turn to end
        else:
            end = dial["num_utt"]
        if rand_start:
            start = random.choice(range(0, end, 2))  # select user turn to start
        else:
            start = 0
        true_start = end
        total_len = 1  # CLS token
        # add as much context as possible
        while true_start > start and total_len + len(dial["dialogue"][true_start-1]) + 2 < self.max_length:
            true_start -= 1
            total_len += len(dial["dialogue"][true_start]) + 2  # [USR]/[SYS] + [SEP]

        clip_dial = {x: (y[true_start:end] if isinstance(y, list) else y) for x, y in dial.items()}
        clip_dial["num_utt"] = end - true_start
        # print('clip to [{}, {}), start: {}'.format(true_start, end, start))
        return clip_dial

    def pick1utt(self, dial):
        utt_idx = random.randrange(dial['num_utt'])
        clip_dial = {x: (y[utt_idx:utt_idx+1] if isinstance(y, list) else y) for x, y in dial.items()}
        clip_dial["num_utt"] = 1
        clip_dial["role"] = 'user' if utt_idx % 2 == 0 else 'system'
        assert len(dial['dialogue'][utt_idx]) + 3 <= self.max_length
        return clip_dial

    def augment(self, dial, positive):
        aug_dial = {
            "dataset": dial["dataset"],
            "num_utt": dial["num_utt"],
            "dialogue": [],
            "da_list": [],
            "spans": [],
            "intent": [],
            "role": dial.get("role", "user")
        }
        num_utt = dial["num_utt"]
        assert num_utt == len(dial["dialogue"])
        for utt, da, spans, intent in zip(dial["dialogue"], dial["da_list"], dial["spans"], dial["intent"]):
            role = da.split('-')[-1]
            template = random.choice(self.utt_pool[role][da])
            template_utt = template["utterance"]
            template_spans = template["da_spans"]
            template_intent = template["intent"]

            aug_dial["intent"].append(template_intent)
            aug_dial["da_list"].append(da)

            if self.keep_value and template_utt != utt:
                new_spans = []
                new_utt = []
                start = 0
                sorted_spans = sorted(template_spans, key=lambda x: x['start'])
                assert template_spans == sorted_spans
                copy_spans = deepcopy(spans)
                assert len(spans) == len(template_spans)
                for template_span in sorted_spans:
                    new_utt.extend(template_utt[start:template_span['start']])
                    intent, domain, slot = [template_span[x] for x in ['intent', 'domain', 'slot']]
                    for i, ori_span in enumerate(copy_spans):
                        if intent == ori_span['intent'] and domain == ori_span['domain'] and slot == ori_span['slot']:
                            ori_value = utt[ori_span['start']:ori_span['end']]
                            ori_span['start'] = len(new_utt)
                            new_utt.extend(ori_value)
                            ori_span['end'] = len(new_utt)
                            assert new_utt[ori_span['start']:ori_span['end']] == ori_value
                            start = template_span['end']
                            new_spans.append(ori_span)
                            copy_spans.pop(i)
                            break
                    else:
                        # all span in the paraphrase should match
                        assert 0, print(utt, template_utt, spans, template_spans, copy_spans, sep='\n')
                new_utt.extend(template_utt[start:])
                aug_dial["dialogue"].append(new_utt)
                aug_dial["spans"].append(sorted(new_spans, key=lambda x: x['start']))
            else:
                aug_dial["dialogue"].append(template_utt)
                aug_dial["spans"].append(template_spans)

        if positive:
            # just do the paraphrase
            pass
        else:
            # replace a continuous part of dialog with random utts
            # ratio = 0  # replace 1 utt
            ratio = 0.3  # replace x% utt randomly
            # ratio = random.uniform(0.2, 0.5)
            num_utt2replace = max(1, int(round(num_utt * ratio)))
            rand_start = random.randint(0, num_utt-num_utt2replace)
            all_utt_idx2replace = list(range(rand_start, rand_start+num_utt2replace))
            # print(num_utt, num_utt2replace, rand_start, all_utt_idx2replace)
            for utt_idx2replace in all_utt_idx2replace:
                ori_da_list = aug_dial["da_list"][utt_idx2replace]
                if utt_idx2replace % 2 == 0:
                    role = "user"
                    da_list = random.choices(self.user_da_seq, weights=self.user_da_cnt)[0]
                    while da_list == ori_da_list:
                        da_list = random.choices(self.user_da_seq, weights=self.user_da_cnt)[0]
                else:
                    role = "system"
                    da_list = random.choices(self.system_da_seq, weights=self.system_da_cnt)[0]
                    while da_list == ori_da_list:
                        da_list = random.choices(self.system_da_seq, weights=self.system_da_cnt)[0]
                template = random.choice(self.utt_pool[role][da_list])

                aug_dial["dialogue"][utt_idx2replace] = template["utterance"]
                aug_dial["da_list"][utt_idx2replace] = da_list
                aug_dial["spans"][utt_idx2replace] = template["da_spans"]
                aug_dial["intent"][utt_idx2replace] = template["intent"]
        return aug_dial


class AugDialProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer: DialogBertTokenizer, mlm_probability=0.15, mlm_ignore_idx=-100,
                 pos_aug_num=1, neg_aug_num=1, pick1utt_num=1,
                 clip_ori=False, clip_aug=True, keep_value=False, use_label=False,
                 nolabel4aug=True):
        '''

        :param datasets:
        :param tokenizer:
        :param mlm_probability:
        :param mlm_ignore_idx: loss for special tokens are excluded, labels for such tokens are set to this value
        '''
        self.task_name = 'augdial'
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/full_dialog')
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.pos_aug_num = pos_aug_num
        self.neg_aug_num = neg_aug_num
        self.pick1utt_num = pick1utt_num
        self.clip_ori = clip_ori
        self.clip_aug = clip_aug
        self.keep_value = keep_value
        self.use_label = use_label
        self.nolabel4aug = nolabel4aug

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, do_train=True):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.max_length = max_length
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        self.schema = {}
        total_samples = 0
        for dataset in self.datasets:
            assert dataset == 'schema'
            print('load {} dataset:'.format(dataset))
            with open(os.path.join(self.data_dir, dataset + '_utt_pool.json')) as f:
                utt_pool = json.load(f)
            dataset_ontology = {
                dataset: json.load(open(os.path.join(self.data_dir, dataset + '_ontology.json')))}
            schema = preprocess_schema(dataset_ontology, self.tokenizer)
            self.schema.update(schema)
            with open(os.path.join(self.data_dir, dataset + '_data_dev.json')) as f:
                dev_data = json.load(f)
            dev_dataset = AugDialDataset(dev_data, utt_pool, self.tokenizer, max_length,
                                         pos_aug_num=self.pos_aug_num, neg_aug_num=self.neg_aug_num, pick1utt_num=self.pick1utt_num,
                                         clip_ori=self.clip_ori, clip_aug=self.clip_aug, keep_value=self.keep_value)
            dev_sampler = DialogBertSampler(dev_dataset.data, dev_dataset.data_bucket_ids, dev_batch_size,
                                            drop_last=True, replacement=False)
            dev_dataloader = DataLoader(
                dev_dataset,
                batch_sampler=dev_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            self.dev_dataloaders[dataset] = dev_dataloader
            train_data = []
            if do_train:
                train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                train_dataset = AugDialDataset(train_data, utt_pool, self.tokenizer, max_length,
                                               pos_aug_num=self.pos_aug_num, neg_aug_num=self.neg_aug_num, pick1utt_num=self.pick1utt_num,
                                               clip_ori=self.clip_ori, clip_aug=self.clip_aug, keep_value=self.keep_value)
                train_sampler = DialogBertSampler(train_dataset.data, train_dataset.data_bucket_ids, train_batch_size,
                                                  drop_last=True, replacement=True)
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_sampler=train_sampler,
                    collate_fn=self.collate_fn,
                    num_workers=num_workers
                )
                self.train_dataloaders[dataset] = train_dataloader
                self.dataset_samples[dataset] = len(train_data)
                print('\t train: total samples {}.'.format(len(train_data)))
            print('\t dev: total samples {}.'.format(len(dev_data)))
            total_samples += len(train_data) + len(dev_data)

        print('dataset sample ratio')
        print(self.datasets)
        print(np.array(list(self.dataset_samples.values())) / np.sum(list(self.dataset_samples.values())))
        print('total train samples', total_samples)
        self.slot_dim = {dataset: self.schema[dataset]['slot_dim'] for dataset in self.schema}
        self.intent_dim = {dataset: self.schema[dataset]['intent_dim'] for dataset in self.schema}
        self.domain_dim = {dataset: self.schema[dataset]['domain_dim'] for dataset in self.schema}
        for dataset in self.schema:
            print('dataset', dataset)
            print('\tdomain_dim', self.domain_dim[dataset])
            print('\tslot_dim', self.slot_dim[dataset])
            print('\tintent_dim', self.intent_dim[dataset])

    def collate_fn(self, batch_data):
        """
        trans to pytorch tensor, pad batch
        :param batch_data: list of tuples(ori_dial, aug_dial1, ...)
        :return: A Dictionary of shape:: {'batches': [ori_dial_batch1, aug_pos_batch1, ..., aug_neg_batch1, ...]}
        each batch is a dictionary:
                {
                    attention_mask: torch.tensor: (batch_size, max_seq_len)
                    input_ids: torch.tensor: (batch_size, max_seq_len)
                    turn_ids: torch.tensor: (batch_size, max_seq_len)
                    role_ids: torch.tensor: (batch_size, max_seq_len)
                    position_ids: torch.tensor: (batch_size, max_seq_len)
                    (optional)
                        masked_lm_labels: torch.tensor: (batch_size, max_seq_len)
                        token_tag_ids: torch.tensor: (batch_size, max_seq_len, slot_dim*2+1)
                        intent_tag_ids: torch.tensor: (batch_size, max_seq_len, intent_dim)
                        domain_tag_ids: torch.tensor: (batch_size, max_seq_len, domain_dim)
                        slot_tag_ids: torch.tensor: (batch_size, max_seq_len, slot_dim)
                        sen_cls_mask: torch.tensor: (batch_size, max_seq_len)
                        cls_intent_tag_ids: torch.tensor: (batch_size, max_seq_len, intent_dim)
                        cls_domain_tag_ids: torch.tensor: (batch_size, max_seq_len, domain_dim)
                        cls_slot_tag_ids: torch.tensor: (batch_size, max_seq_len, slot_dim)
                }
        """
        # return batch_data[0]
        input_batches = []
        for i, dials in enumerate(zip(*batch_data)):  # ori_dials, aug_dials1, aug_dials2,...
            input_batch = []
            for dial in dials:
                if self.use_label and not (self.nolabel4aug and i > 0):  # create label
                    encoded_inputs = self.create_label(dial)
                else:
                    encoded_inputs = self.tokenizer.prepare_input_seq(dial['dialogue'], last_role=dial.get('role', 'user'),
                                                                      max_length=self.max_length, return_lengths=True)
                input_batch.append(encoded_inputs)
            input_batches.append(input_batch)

        assert len(input_batches) == (1 + self.pos_aug_num + self.neg_aug_num + self.pick1utt_num), print(len(batch_data))

        output_data = []

        for batchi, input_batch in enumerate(input_batches):
            batch_size = len(input_batch)
            max_seq_len = max([x['length'] for x in input_batch])
            attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            turn_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            role_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

            if self.mlm_probability > 0:
                masked_lm_labels = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx

            if self.use_label and not (self.nolabel4aug and batchi > 0):
                dataset = input_batch[0]['dataset']
                token_tag_ids = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx
                intent_tag_ids = torch.zeros((batch_size, max_seq_len, self.schema[dataset]['intent_dim']),
                                             dtype=torch.float)
                domain_tag_ids = torch.zeros((batch_size, max_seq_len, self.schema[dataset]['domain_dim']),
                                             dtype=torch.float)
                slot_tag_ids = torch.zeros((batch_size, max_seq_len, self.schema[dataset]['slot_dim']),
                                           dtype=torch.float)
                sen_cls_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
                cls_intent_tag_ids = torch.zeros((batch_size, self.schema[dataset]['intent_dim']),
                                                 dtype=torch.float)
                cls_domain_tag_ids = torch.zeros((batch_size, self.schema[dataset]['domain_dim']),
                                                 dtype=torch.float)
                cls_slot_tag_ids = torch.zeros((batch_size, self.schema[dataset]['slot_dim']),
                                               dtype=torch.float)

            for i in range(batch_size):
                sen_len = input_batch[i]['length']
                attention_mask[i, :sen_len] = 1
                turn_ids[i, :sen_len] = torch.LongTensor(input_batch[i]['turn_ids'])
                role_ids[i, :sen_len] = torch.LongTensor(input_batch[i]['role_ids'])
                position_ids[i, :sen_len] = torch.LongTensor(input_batch[i]['position_ids'])
                if self.mlm_probability > 0:
                    masked_input, label = self._wwm_tokens(input_batch[i]['input_ids'])
                    input_ids[i, :sen_len] = torch.LongTensor(masked_input)
                    masked_lm_labels[i, :sen_len] = torch.LongTensor(label)
                else:
                    input_ids[i, :sen_len] = torch.LongTensor(input_batch[i]['input_ids'])

                if self.use_label and not (self.nolabel4aug and batchi > 0):
                    token_tag_ids[i, :sen_len] = torch.LongTensor(input_batch[i]['token_tag_ids'])
                    sen_cls_mask[i, :sen_len] = torch.LongTensor(input_batch[i]['sen_cls_mask'])
                    k = 0
                    for j in range(sen_len):
                        if sen_cls_mask[i, j] > 0:
                            intent_tag_ids[i, j, :] = torch.LongTensor(input_batch[i]['intent_tag_ids'][k])
                            domain_tag_ids[i, j, :] = torch.LongTensor(input_batch[i]['domain_tag_ids'][k])
                            slot_tag_ids[i, j, :] = torch.LongTensor(input_batch[i]['slot_tag_ids'][k])
                            k += 1
                    cls_intent_tag_ids[i] = torch.LongTensor(input_batch[i]['cls_intent_tag_ids'])
                    cls_domain_tag_ids[i] = torch.LongTensor(input_batch[i]['cls_domain_tag_ids'])
                    cls_slot_tag_ids[i] = torch.LongTensor(input_batch[i]['cls_slot_tag_ids'])

            if self.use_label and not (self.nolabel4aug and batchi > 0):
                output_data.append({"attention_mask": attention_mask,
                                    "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids,
                                    "position_ids": position_ids,
                                    "token_tag_ids": token_tag_ids,
                                    "intent_tag_ids": intent_tag_ids,
                                    "domain_tag_ids": domain_tag_ids,
                                    "slot_tag_ids": slot_tag_ids,
                                    "sen_cls_mask": sen_cls_mask,
                                    "cls_intent_tag_ids": cls_intent_tag_ids,
                                    "cls_domain_tag_ids": cls_domain_tag_ids,
                                    "cls_slot_tag_ids": cls_slot_tag_ids
                                    })
            else:
                output_data.append({"attention_mask": attention_mask,
                                    "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids,
                                    "position_ids": position_ids,
                                    })

            if self.mlm_probability > 0 and not (self.nolabel4aug and batchi > 0):
                output_data[-1]["masked_lm_labels"] = masked_lm_labels

        return {"batches": output_data}

    def create_label(self, dial):
        """extract label from dial['spans'] and dial['intent']"""
        encoded_inputs = self.tokenizer.prepare_input_seq(dial['dialogue'], last_role=dial.get('role', 'user'),
                                                          max_length=self.max_length, return_lengths=True)
        length = encoded_inputs['length']
        intent_tags = []
        domain_tags = []
        slot_tags = []
        token_tags = [self.mlm_ignore_idx]
        sen_idx = 1
        utt_num = 0
        # spans_idx = []
        all_slots = []
        all_intents = []
        all_domains = []
        for tokens, spans, sen_intent in zip(dial['dialogue'][::-1], dial['spans'][::-1], dial['intent'][::-1]):
            if sen_idx + len(tokens) + 2 > length:
                break
            utt_num += 1
            intent2add = [0] * self.schema[dial['dataset']]['intent_dim']
            domain2add = [0] * self.schema[dial['dataset']]['domain_dim']
            slot2add = [0] * self.schema[dial['dataset']]['slot_dim']
            for intent, domain, slot in sen_intent:
                if intent in self.schema[dial['dataset']]['intent_set']:
                    intent2add[self.schema[dial['dataset']]['intent_set'].index(intent)] = 1
                    all_intents.append(self.schema[dial['dataset']]['intent_set'].index(intent))
                if domain in self.schema[dial['dataset']]['domain_set']:
                    domain2add[self.schema[dial['dataset']]['domain_set'].index(domain)] = 1
                    all_domains.append(self.schema[dial['dataset']]['domain_set'].index(domain))
                if slot in self.schema[dial['dataset']]['slot_set']:
                    all_slots.append(self.schema[dial['dataset']]['slot_set'].index(slot))
                    slot2add[self.schema[dial['dataset']]['slot_set'].index(slot)] = 1
            intent_tags.append(intent2add)
            domain_tags.append(domain2add)
            slot_tags.append(slot2add)
            sen_idx += 1
            tokenslot2add = [0] * len(tokens)
            idx_in_span_mask = [0] * (len(tokens) + 2)
            for span in spans:
                assert all([idx_in_span_mask[i] == 0 for i in range(span['start'], span['end'])]), print(spans, tokens)
                if all([idx_in_span_mask[i] == 0 for i in range(span['start'], span['end'])]):
                    # add 1. idx=0: non-value token
                    # B-slot
                    tokenslot2add[span['start']] = self.schema[dial['dataset']]['slot_set'].index(span['slot']) * 2 + 1
                    # I-slot
                    tokenslot2add[span['start']+1:span['end']] = [self.schema[dial['dataset']]['slot_set'].index(span['slot']) * 2 + 2] * (span['end'] - span['start'] - 1)
                    idx_in_span_mask[span['start']:span['end']] = [1] * (span['end'] - span['start'])
            token_tags.extend([self.mlm_ignore_idx] + tokenslot2add + [self.mlm_ignore_idx])
            sen_idx += len(tokens) + 1

        cls_slots = [0] * self.schema[dial['dataset']]['slot_dim']
        for slot in all_slots:
            cls_slots[slot] = 1
        cls_intents = [0] * self.schema[dial['dataset']]['intent_dim']
        for intent in all_intents:
            cls_intents[intent] = 1
        cls_domains = [0] * self.schema[dial['dataset']]['domain_dim']
        for domain in all_domains:
            cls_domains[domain] = 1

        sen_cls_mask = self.tokenizer.get_tokens_not_mask(encoded_inputs['input_ids'],
                                                          not_mask_tokens=['[USR]', '[SYS]'])

        encoded_inputs['utt_num'] = utt_num
        # encoded_inputs['dialogue'] = dial['dialogue']
        encoded_inputs['dataset'] = dial['dataset']
        encoded_inputs['token_tag_ids'] = token_tags
        encoded_inputs['intent_tag_ids'] = intent_tags
        encoded_inputs['domain_tag_ids'] = domain_tags
        encoded_inputs['slot_tag_ids'] = slot_tags
        encoded_inputs['sen_cls_mask'] = sen_cls_mask
        encoded_inputs['cls_intent_tag_ids'] = cls_intents
        encoded_inputs['cls_domain_tag_ids'] = cls_domains
        encoded_inputs['cls_slot_tag_ids'] = cls_slots
        return encoded_inputs

    def _wwm_tokens(self, input_ids):
        masked_input = list(input_ids)
        not_mask_tokens = self.tokenizer.all_special_tokens
        cand_indexes = []
        for (i, input_id) in enumerate(masked_input):
            token = self.tokenizer.convert_ids_to_tokens(input_id)
            if token in not_mask_tokens:
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        random.shuffle(cand_indexes)
        labels = [self.mlm_ignore_idx] * len(masked_input)
        for index_set in cand_indexes[:int(round(len(cand_indexes) * self.mlm_probability))]:
            r = random.random()
            if r < 0.8:
                # 80% of the time, replace with [MASK]
                masked_token = "[MASK]"
            elif 0.8 <= r < 0.9:
                # 10% of the time, keep original
                masked_token = "original"
            else:
                # 10% of the time, replace with random word
                masked_token = "replace"

            for index in index_set:
                if masked_token == "[MASK]":
                    new_id = self.tokenizer.convert_tokens_to_ids(masked_token)
                elif masked_token == "original":
                    new_id = masked_input[index]
                else:
                    # assert masked_token == "replace"
                    while True:
                        new_id = random.randint(0, self.tokenizer.vocab_size - 1)
                        if self.tokenizer.convert_ids_to_tokens(new_id) not in not_mask_tokens:
                            break
                labels[index] = masked_input[index]
                masked_input[index] = new_id
        return masked_input, labels

    def sample_batch(self, dataset, batch_size, data_key):
        assert data_key == 'train'
        assert batch_size == self.train_batch_size, print('batch size must be the same as loaded data')
        dataloader = self.train_dataloaders[dataset]
        for batch in dataloader:
            return batch

    def yield_batches(self, dataset, batch_size, data_key):
        # assert data_key == 'dev'
        # assert batch_size == self.dev_batch_size, print('batch size must be the same as loaded data')
        if data_key == 'dev':
            dataloader = self.dev_dataloaders[dataset]
        elif data_key == 'train':
            dataloader = self.train_dataloaders[dataset]
        else:
            raise RuntimeError("Data key must be chosen from train or dev")
        for batch in dataloader:
            yield batch

    def sample_dataset(self, data_key):
        assert data_key == 'train'
        return random.choices(list(self.dataset_samples.keys()), list(self.dataset_samples.values()))[0]


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/augdial/mlm_wwm_120k_0831_bert')
    for spc_token in tokenizer.all_special_tokens:
        print(spc_token, tokenizer.convert_tokens_to_ids(spc_token))
    print(tokenizer.vocab_size)
    processor = AugDialProcessor(['schema'], tokenizer, clip_ori=False, clip_aug=False, keep_value=False, use_label=True,
                                 pos_aug_num=1, neg_aug_num=0, pick1utt_num=1, mlm_probability=0, nolabel4aug=True)
    batch_size = 4
    processor.load_data(batch_size, batch_size, 256, do_train=False)
    dataset = 'schema'
    for batch in processor.yield_batches('schema', batch_size, 'dev'):
    # for batch in [processor.sample_batch('schema', batch_size, 'train')]:
        batches = batch["batches"]
    #     print(torch.max(batches[0]["turn_ids"], dim=-1))
        for augi, b in enumerate(batches):
            print("aug {}".format(augi) + '-'*100)
            print(b.keys())
            for samplei in range(batch_size):
                print(tokenizer.decode(b['input_ids'][samplei]))
                if not processor.use_label or 'token_tag_ids' not in b:
                    continue
                tokens = tokenizer.convert_ids_to_tokens(b['input_ids'][samplei])
                token_tags = [
                    ['B', 'I'][(x - 1) % 2] + '-' + processor.schema[dataset]['slot_set'][(x - 1) / 2] if x > 0 else
                    {0: 'O', -100: 'X'}[x.item()] for x in b['token_tag_ids'][samplei]]
                domain_tags = b['domain_tag_ids'][samplei]
                intent_tags = b['intent_tag_ids'][samplei]
                slot_tags = b['slot_tag_ids'][samplei]
                for i, (token, bio_tag, domain_tag, intent_tag, slot_tag) in enumerate(
                        zip(tokens, token_tags, domain_tags, intent_tags, slot_tags)):
                    if i == 0:
                        intents = []
                        for j, intent in enumerate(processor.schema[dataset]['intent_set']):
                            if 'cls_intent_tag_ids' in b and b['cls_intent_tag_ids'][samplei][j] > 0:
                                intents.append(intent)
                        slots = []
                        for j, slot in enumerate(processor.schema[dataset]['slot_set']):
                            if 'cls_slot_tag_ids' in b and b['cls_slot_tag_ids'][samplei][j] > 0:
                                slots.append(slot)
                        domains = []
                        for j, domain in enumerate(processor.schema[dataset]['domain_set']):
                            if 'cls_domain_tag_ids' in b and b['cls_domain_tag_ids'][samplei][j] > 0:
                                domains.append(domain)
                        print(token, bio_tag, domains, intents, slots)
                #     if b['sen_cls_mask'][samplei][i] > 0:
                #         intents = []
                #         for j, intent in enumerate(processor.schema[dataset]['intent_set']):
                #             if intent_tag[j] > 0:
                #                 intents.append(intent)
                #         slots = []
                #         for j, slot in enumerate(processor.schema[dataset]['slot_set']):
                #             if slot_tag[j] > 0:
                #                 slots.append(slot)
                #         domains = []
                #         for j, domain in enumerate(processor.schema[dataset]['domain_set']):
                #             if domain_tag[j] > 0:
                #                 domains.append(domain)
                #         print(token, bio_tag, domains, intents, slots)
                #     else:
                #         print(token, bio_tag)
                    # break
                # break
        break
