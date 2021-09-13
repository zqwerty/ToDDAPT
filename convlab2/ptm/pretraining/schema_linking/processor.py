import os
import json
import random
import math
from convlab2.ptm.pretraining.dataloader import TaskProcessor, DialogBertSampler
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from pprint import pprint
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict


def preprocess_schema(dataset_ontology, tokenizer: DialogBertTokenizer):
    schema = {}
    for dataset, ontology in dataset_ontology.items():
        schema[dataset] = {"domains": {}, "intents": {}, 'domain_set': [], 'slot_set': [], 'intent_set': []}
        for domain, d_ontology in ontology['domains'].items():
            schema[dataset]['domain_set'].append(domain)
            domain_des = d_ontology["description"]
            schema[dataset]["domains"][domain] = {"description": domain_des,
                                                  "input": tokenizer.prepare_input_seq([], prefix_dict=OrderedDict(
                                                      {"[DOMAIN]": domain_des})),
                                                  "slots": {}}
            for slot, s_ontology in d_ontology["slots"].items():
                # schema[dataset]['slot_set'].append('-'.join([domain, slot]))
                if slot not in schema[dataset]['slot_set']:
                    schema[dataset]['slot_set'].append(slot)
                slot_des = s_ontology["description"]
                is_categorical = s_ontology["is_categorical"]
                possible_values = s_ontology["possible_values"]
                schema[dataset]["domains"][domain]["slots"][slot] = {"description": slot_des,
                                                                     "is_categorical": is_categorical,
                                                                     "possible_values": possible_values}
                schema[dataset]["domains"][domain]["slots"][slot]["input"] = tokenizer.prepare_input_seq(
                    [], prefix_dict=OrderedDict({
                        "[DOMAIN]": domain_des,
                        "[SLOT]": slot_des,
                        "[VALUE]": possible_values
                    }))
        schema[dataset]['domain_dim'] = len(schema[dataset]['domain_set'])
        schema[dataset]['slot_dim'] = len(schema[dataset]['slot_set'])
        for intent, i_ontology in ontology["intents"].items():
            schema[dataset]['intent_set'].append(intent)
            intent_des = i_ontology["description"]
            schema[dataset]["intents"][intent] = {
                "description": intent_des,
                "input": tokenizer.prepare_input_seq([], prefix_dict=OrderedDict({
                    "[INTENT]": intent_des
                }))}
        schema[dataset]['intent_dim'] = len(schema[dataset]['intent_set'])
    return schema


class SchemaPretrainingDataset(Dataset):
    def __init__(self, data, schema, tokenizer: DialogBertTokenizer, max_length=512):
        """prepare input and label here

            self.data: A Dictionary of shape::
                {
                    input_ids: list[int], reverse order
                    turn_ids: list[int]
                    role_ids: list[int]
                    position_ids: list[int]

                    spans: list[list[int,int]]

                    length: int
                    utt_num: int
                }

            self.data_bucket_ids: utterance numbers of dialogues
        """
        self.schema = schema
        self.data, self.data_bucket_ids = [], []
        # data = [{"num_utt": int, "dialogue": list of tokens lists,
        # "spans": list of spans:{domain, slot, value, utt_idx, start, end}}]
        bio_tag2id = {'B': 1, 'I': 2, 'O': 0, 'X': -100}
        span_word_ratio = []
        for d in data:
            # print(d['dialogue'])
            # print(d)
            encoded_inputs = tokenizer.prepare_input_seq(d['dialogue'], last_role='user',
                                                         max_length=max_length, return_lengths=True)
            length = encoded_inputs['length']
            bio_tags = ['X']
            intent_tags = []
            domain_tags = []
            slot_tags = []
            sen_idx = 1
            utt_num = 0
            spans_idx = []
            idx_in_span_mask = [0]
            all_slots = []
            all_intents = []
            all_domains = []
            for tokens, spans, sen_bio, sen_intent in zip(d['dialogue'][::-1], d['spans'][::-1], d['bio_tag'][::-1], d['intent'][::-1]):
                tags2add = ['X'] + sen_bio + ['X']
                if sen_idx + len(tokens) + 2 > length:
                    break
                utt_num += 1
                bio_tags += tags2add
                intent2add = [0] * self.schema[d['dataset']]['intent_dim']
                domain2add = [0] * self.schema[d['dataset']]['domain_dim']
                slot2add = [0] * self.schema[d['dataset']]['slot_dim']
                for intent, domain, slot in sen_intent:
                    if intent in self.schema[d['dataset']]['intent_set']:
                        intent2add[self.schema[d['dataset']]['intent_set'].index(intent)] = 1
                        all_intents.append(self.schema[d['dataset']]['intent_set'].index(intent))
                    if domain in self.schema[d['dataset']]['domain_set']:
                        domain2add[self.schema[d['dataset']]['domain_set'].index(domain)] = 1
                        all_domains.append(self.schema[d['dataset']]['domain_set'].index(domain))
                    # slot = '-'.join([domain, slot])
                    if slot in self.schema[d['dataset']]['slot_set']:
                        all_slots.append(self.schema[d['dataset']]['slot_set'].index(slot))
                        slot2add[self.schema[d['dataset']]['slot_set'].index(slot)] = 1
                intent_tags.append(intent2add)
                domain_tags.append(domain2add)
                slot_tags.append(slot2add)
                sen_idx += 1
                idx_in_span_mask += [0] * (len(tokens) + 2)
                for span in spans:
                    # if span['end'] - span['start'] > span_len_th:
                    #     continue
                    span = deepcopy(span)
                    span['start'] += sen_idx
                    span['end'] += sen_idx
                    # print(span['slot'])
                    # print(self.schema[d['dataset']]['slot_set'])
                    # slot2add[self.schema[d['dataset']]['slot_set'].index('-'.join([span['domain'],span['slot']]))] = 1
                    # span['input'] = self.schema[d['dataset']]['domains'][span['domain']]['slots'][span['slot']]['input']
                    assert all([idx_in_span_mask[i] == 0 for i in range(span['start'], span['end'])]), print(spans, tokens)
                    if all([idx_in_span_mask[i] == 0 for i in range(span['start'], span['end'])]):
                        spans_idx.append(span)
                        idx_in_span_mask[span['start']:span['end']] = [1] * (span['end'] - span['start'])
                    # spans_idx.append(span)
                # slot_tags.append(slot2add)
                sen_idx += len(tokens) + 1
            # print(d['dialogue'])
            span_word_ratio.append(sum(idx_in_span_mask) / len(idx_in_span_mask))
            bio_tag_ids = [bio_tag2id[tag] for tag in bio_tags]
            # bio_mask = [0 if x == -1 else 1 for x in bio_tag_ids]
            cls_slots = [0] * self.schema[d['dataset']]['slot_dim']
            for slot in all_slots:
                cls_slots[slot] = 1
            slot_tags.insert(0, cls_slots)
            cls_intents = [0] * self.schema[d['dataset']]['intent_dim']
            for intent in all_intents:
                cls_intents[intent] = 1
            intent_tags.insert(0, cls_intents)
            cls_domains = [0] * self.schema[d['dataset']]['domain_dim']
            for domain in all_domains:
                cls_domains[domain] = 1
            domain_tags.insert(0, cls_domains)

            sen_cls_mask = tokenizer.get_tokens_not_mask(encoded_inputs['input_ids'], not_mask_tokens=['[USR]', '[SYS]', '[CLS]'])
            # sen_cls_mask = tokenizer.get_tokens_not_mask(encoded_inputs['input_ids'], not_mask_tokens=['[USR]', '[SYS]'])

            encoded_inputs['span_mask'] = idx_in_span_mask
            encoded_inputs['spans'] = spans_idx
            encoded_inputs['utt_num'] = utt_num
            encoded_inputs['dialogue'] = d['dialogue']
            encoded_inputs['dataset'] = d['dataset']
            encoded_inputs['bio_tag_ids'] = bio_tag_ids
            # encoded_inputs['bio_mask'] = bio_mask
            encoded_inputs['intent_tag_ids'] = intent_tags
            encoded_inputs['domain_tag_ids'] = domain_tags
            encoded_inputs['slot_tag_ids'] = slot_tags
            encoded_inputs['sen_cls_mask'] = sen_cls_mask
            assert length == len(encoded_inputs['input_ids'])
            assert length == len(encoded_inputs['turn_ids'])
            assert length == len(encoded_inputs['role_ids'])
            assert length == len(encoded_inputs['position_ids'])
            assert 0 < length <= max_length
            # if spans_idx: # if there is no span, ignore this sample
            self.data.append(encoded_inputs)
            self.data_bucket_ids.append(encoded_inputs["length"])
            # self.data_bucket_ids.append(utt_num)
        print('span words ratio: mean %.4f, max %.4f, min %.4f' % (np.mean(span_word_ratio), np.max(span_word_ratio), np.min(span_word_ratio)))
        print('samples that have spans', len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SchemaProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer,
                 mlm_probability=0.15, mlm_ignore_idx=-100,
                 one_side_mask=False, mask_user_probability=0.5, mix_batch=False):
        self.task_name = 'schema_linking'
        self.datasets = datasets
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/prefix_dialog_cut59')
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.one_side_mask = one_side_mask
        self.mask_user_probability = mask_user_probability
        self.mix_batch = mix_batch

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, use_multiwoz=False, do_train=True):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        self.schema = {}
        total_samples = 0
        for dataset in self.datasets:
            print('load {} dataset:'.format(dataset))
            dev_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_dev.json')))
            dataset_ontology = {
                dataset: json.load(open(os.path.join(self.data_dir, dataset + '_ontology.json')))}
            schema = preprocess_schema(dataset_ontology, self.tokenizer)
            self.schema.update(schema)
            dev_dataset = SchemaPretrainingDataset(dev_data, schema, self.tokenizer, max_length)
            dev_sampler = DialogBertSampler(dev_dataset.data, dev_dataset.data_bucket_ids, dev_batch_size,
                                            drop_last=False, replacement=False)
            dev_dataloader = DataLoader(
                dev_dataset,
                batch_sampler=dev_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            self.dev_dataloaders[dataset] = dev_dataloader
            train_data = []
            if do_train:
                if dataset != 'multiwoz25' or use_multiwoz:
                    train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                    train_dataset = SchemaPretrainingDataset(train_data, schema, self.tokenizer, max_length)
                    train_sampler = DialogBertSampler(train_dataset.data, train_dataset.data_bucket_ids, train_batch_size,
                                                      drop_last=False, replacement=True)
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
        print('total samples', total_samples)
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
        :param batch_data: see BIO_IntentPretrainingDataset
        :return: A Dictionary of shape::
                {
                    input_mask: torch.tensor: (batch_size, max_seq_len)
                    input_ids: torch.tensor: (batch_size, max_seq_len)
                    turn_ids: torch.tensor: (batch_size, max_seq_len)
                    role_ids: torch.tensor: (batch_size, max_seq_len)
                    position_ids: torch.tensor: (batch_size, max_seq_len)

                    TODO: masked_spans: list of batch size, ele: [{start, end, description}]
                }
        """
        # print(batch_data)
        # return batch_data
        batch_size = len(batch_data)
        max_seq_len = max([x['length'] for x in batch_data])
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        turn_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        role_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        masked_lm_labels = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx
        spans = []
        dataset = batch_data[0]['dataset']
        bio_tag_ids = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx
        intent_tag_ids = torch.zeros((batch_size, max_seq_len, self.schema[dataset]['intent_dim']), dtype=torch.float)
        domain_tag_ids = torch.zeros((batch_size, max_seq_len, self.schema[dataset]['domain_dim']), dtype=torch.float)
        slot_tag_ids = torch.zeros((batch_size, max_seq_len, self.schema[dataset]['slot_dim']), dtype=torch.float)
        sen_cls_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        mask_user_side = None
        if self.one_side_mask and not self.mix_batch:
            mask_user_side = random.random() < self.mask_user_probability

        for i in range(batch_size):
            sen_len = batch_data[i]['length']
            attention_mask[i, :sen_len] = 1
            if self.one_side_mask:
                if self.mix_batch:
                    mask_user_side = random.random() < self.mask_user_probability
                masked_input, label = self._one_side_wwm_tokens(batch_data[i]['input_ids'], batch_data[i]['role_ids'],
                                                                mask_user_side)
            else:
                masked_input, label = self._wwm_tokens(batch_data[i]['input_ids'])
            # masked_input, label = self._span_mask_token(batch_data[i]['input_ids'], batch_data[i]['bio_tag_ids'])
            # masked_input = batch_data[i]['input_ids']
            bio_tag_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['bio_tag_ids'])
            spans.append([])
            for span in batch_data[i]['spans']:
                # slot_label = self.schema[dataset]['slot_set'].index('-'.join([span['domain'], span['slot']]))
                slot_label = self.schema[dataset]['slot_set'].index(span['slot'])
                for idx in range(span['start'], span['end']):
                    assert 0 < bio_tag_ids[i, idx] < 3, print(span, idx, bio_tag_ids[i, idx],
                                                              batch_data[i]['bio_tag_ids'][idx],
                                                              batch_data[i]['bio_tag_ids'][span['start']:span['end']])
                    bio_tag_ids[i, idx] = bio_tag_ids[i, idx] + slot_label * 2
                spans[-1].append({
                    "start": span['start'],
                    "end": span['end'],
                    "label": slot_label,
                    'slot': span['slot']
                })
                assert 0 <= slot_label < self.slot_dim[dataset], print(self.schema[dataset]['slot_set'], dataset, span['domain'], span['slot'])

            input_ids[i, :sen_len] = torch.LongTensor(masked_input)
            masked_lm_labels[i, :sen_len] = torch.LongTensor(label)
            turn_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['turn_ids'])
            role_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['role_ids'])
            position_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['position_ids'])

            sen_cls_mask[i, :sen_len] = torch.LongTensor(batch_data[i]['sen_cls_mask'])
            k = 0
            # print(self.tokenizer.convert_ids_to_tokens(batch_data[i]['input_ids']))
            # print(batch_data[i]['slot_tag_ids'])
            # print(self.schema[dataset]['slot_set'])
            for j in range(sen_len):
                if sen_cls_mask[i, j] > 0:
                    intent_tag_ids[i, j, :] = torch.LongTensor(batch_data[i]['intent_tag_ids'][k])
                    domain_tag_ids[i, j, :] = torch.LongTensor(batch_data[i]['domain_tag_ids'][k])
                    slot_tag_ids[i, j, :] = torch.LongTensor(batch_data[i]['slot_tag_ids'][k])
                    k += 1

        return {"attention_mask": attention_mask,
                "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids, "position_ids": position_ids,
                "masked_lm_labels": masked_lm_labels,
                # "spans": spans,
                "bio_tag_ids": bio_tag_ids,
                "intent_tag_ids": intent_tag_ids,
                "domain_tag_ids": domain_tag_ids,
                "slot_tag_ids": slot_tag_ids,
                "sen_cls_mask": sen_cls_mask
                }

    def _prepare_tensor_inputs(self, input_ids, turn_ids, role_ids, position_ids, max_len2pad=None, unsqueeze=False):
        if unsqueeze:
            input_ids = [input_ids]
            turn_ids = [turn_ids]
            role_ids = [role_ids]
            position_ids = [position_ids]
        batch_size = len(input_ids)
        if not max_len2pad:
            max_len2pad = max([len(x) for x in input_ids])

        attention_mask_tensor = torch.zeros((batch_size, max_len2pad), dtype=torch.long)
        input_ids_tensor = torch.zeros((batch_size, max_len2pad), dtype=torch.long)
        turn_ids_tensor = torch.zeros((batch_size, max_len2pad), dtype=torch.long)
        role_ids_tensor = torch.zeros((batch_size, max_len2pad), dtype=torch.long)
        position_ids_tensor = torch.zeros((batch_size, max_len2pad), dtype=torch.long)

        for i in range(batch_size):
            seq_len = len(input_ids[i])
            attention_mask_tensor[0, :seq_len] = 1
            input_ids_tensor[0, :seq_len] = torch.LongTensor(input_ids[i])
            turn_ids_tensor[0, :seq_len] = torch.LongTensor(turn_ids[i])
            role_ids_tensor[0, :seq_len] = torch.LongTensor(role_ids[i])
            position_ids_tensor[0, :seq_len] = torch.LongTensor(position_ids[i])
        return input_ids_tensor, turn_ids_tensor, role_ids_tensor, position_ids_tensor, attention_mask_tensor

    def _one_side_wwm_tokens(self, input_ids, role_ids, mask_user_side):
        masked_input = list(input_ids)
        not_mask_tokens = self.tokenizer.all_special_tokens
        cand_indexes = []
        for i, (input_id, role_id) in enumerate(zip(input_ids, role_ids)):
            if mask_user_side and role_id == 2:
                # mask user, skip sys utt
                continue
            elif not mask_user_side and role_id == 1:
                # mask sys, skip user utt
                continue
            token = self.tokenizer.convert_ids_to_tokens(input_id)
            if token in not_mask_tokens:
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        random.shuffle(cand_indexes)
        labels = [self.mlm_ignore_idx] * len(masked_input)
        for index_set in cand_indexes[:max(1, int(round(len(cand_indexes) * self.mlm_probability)))]:
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
        for index_set in cand_indexes[:max(1, int(round(len(cand_indexes) * self.mlm_probability)))]:
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

    def _span_mask_token(self, input_ids, bio_tag_ids):
        """

        :param input_ids:
        :param spans: list of span, from small to large
        :return:
        """
        masked_input = list(input_ids)
        not_mask_tokens = self.tokenizer.all_special_tokens
        # print(not_mask_tokens)
        cand_indexes = []
        is_span = []
        in_span = False
        for i, (input_id, bio_tag_id) in enumerate(zip(input_ids, bio_tag_ids)):
            token = self.tokenizer.convert_ids_to_tokens(input_id)
            if token in not_mask_tokens:
                in_span = False
                continue
            if len(cand_indexes) >= 1 and (token.startswith("##") or (in_span and bio_tag_id == 2)):
                # not first sub token, whole word mask or BI span mask
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
                is_span.append(bio_tag_id > 0)
            if bio_tag_id == 1:  # 'B' label
                in_span = True
            elif bio_tag_id == 0:  # 'O' label
                in_span = False

        # print(self.tokenizer.convert_ids_to_tokens(input_ids))
        # print(len(cand_indexes),cand_indexes)
        # print(len(is_span),is_span)
        cand_indexes = list(zip(cand_indexes, is_span))

        random.shuffle(cand_indexes)
        labels = [self.mlm_ignore_idx] * len(masked_input)
        for index_set, is_span in cand_indexes:
            if not is_span:
                continue

            for index in index_set:
                new_id = self.tokenizer.convert_tokens_to_ids("[VALUE]")
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
        assert data_key == 'dev'
        assert batch_size == self.dev_batch_size, print('batch size must be the same as loaded data')
        dataloader = self.dev_dataloaders[dataset]
        for batch in dataloader:
            yield batch

    def sample_dataset(self, data_key):
        assert data_key == 'train'
        return random.choices(list(self.dataset_samples.keys()), list(self.dataset_samples.values()))[0]


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(43)
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_12k_batch64_lr1e-4_block256_1031')
    for spc_token in tokenizer.unique_added_tokens_encoder:
        print(spc_token, tokenizer.convert_tokens_to_ids(spc_token))
    print(len(tokenizer))
    print(tokenizer.all_special_tokens)
    datasets = ["schema"]
    processor = SchemaProcessor(datasets, tokenizer)
    processor.load_data(2, 2, 256, do_train=False)
    print(processor.schema['schema']['domain_set'])
    print(processor.schema['schema']['intent_set'])
    print(processor.schema['schema']['slot_set'])
    dataset = "schema"
    for batch in processor.yield_batches(dataset, 2, 'dev'):
        print(tokenizer.decode(batch['input_ids'][0]))
        tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
        print(tokens)
        bio_tags = [
            ['B', 'I'][(x - 1) % 2] + '-' + processor.schema[dataset]['slot_set'][(x - 1) / 2] if x > 0 else
            {0: 'O', -100: 'X'}[x.item()] for x in batch['bio_tag_ids'][0]]
        print(bio_tags)
        label = list(zip(tokens, bio_tags))
        domain_tags = batch['domain_tag_ids'][0]
        intent_tags = batch['intent_tag_ids'][0]
        slot_tags = batch['slot_tag_ids'][0]
        for i, (token, bio_tag, domain_tag, intent_tag, slot_tag) in enumerate(
                zip(tokens, bio_tags, domain_tags, intent_tags, slot_tags)):
            if batch['sen_cls_mask'][0][i] > 0:
                intents = []
                for j, intent in enumerate(processor.schema[dataset]['intent_set']):
                    if intent_tag[j] > 0:
                        intents.append(intent)
                slots = []
                for j, slot in enumerate(processor.schema[dataset]['slot_set']):
                    if slot_tag[j] > 0:
                        slots.append(slot)
                domains = []
                for j, domain in enumerate(processor.schema[dataset]['domain_set']):
                    if domain_tag[j] > 0:
                        domains.append(domain)
                print(token, bio_tag, domains, intents, slots)
            else:
                print(token, bio_tag)
        break
    # for i in range(2):
    #     batch = processor.sample_batch('camrest', 2, 'train')
    #     print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
