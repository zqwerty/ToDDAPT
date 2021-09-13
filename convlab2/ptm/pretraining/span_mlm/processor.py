import os
import json
import random
import math
from convlab2.ptm.pretraining.dataloader import TaskProcessor, DialogBertSampler
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from convlab2.ptm.pretraining.bio.processor import BIODataset
from pprint import pprint
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict


class SpanMLMDataset(Dataset):
    def __init__(self, data, tokenizer: DialogBertTokenizer, max_length):
        """prepare input and label here

            self.data: A Dictionary of shape::
                {
                    input_ids: list[int], reverse order
                    turn_ids: list[int]
                    role_ids: list[int]
                    position_ids: list[int]

                    bio_tag_ids: list[int]
                    bio_mask: list[int]

                    length: int
                    utt_num: int
                }

            self.data_bucket_ids: utterance numbers of dialogues
        """
        self.data, self.data_bucket_ids = [], []
        # data = [{"num_utt": int, "dialogue": list of token lists, "bio_tag": list of tag lists
        # "spans": list of spans:{domain, slot, value, utt_idx, start, end}}]
        bio_tag2id = {'B': 1, 'I': 2, 'O': 0, 'X': -100}
        for d in data:
            # print(d['dialogue'])
            # print(d)
            encoded_inputs = tokenizer.prepare_input_seq(d['dialogue'], last_role='user',
                                                         max_length=max_length, return_lengths=True)
            length = encoded_inputs['length']
            bio_tags = ['X']
            sen_idx = 1
            dial_tags = d['pseudo_bio_tag'] if 'pseudo_bio_tag' in d else d['bio_tag']
            # if 'pseudo_bio_tag' in d:
            #     print('pseudo_bio_tag')
            for tokens, sen_bio in zip(d['dialogue'][::-1], dial_tags[::-1]):
                tags2add = ['X'] + sen_bio + ['X']
                if sen_idx + len(tokens) + 2 > length:
                    break
                bio_tags += tags2add
                sen_idx += len(tokens) + 2
            # print(d['dialogue'])
            bio_tag_ids = [bio_tag2id[tag] for tag in bio_tags]
            # encoded_inputs['dialogue'] = d['dialogue']
            encoded_inputs['dataset'] = d['dataset']
            encoded_inputs['bio_tag_ids'] = bio_tag_ids
            assert length == len(encoded_inputs['input_ids'])
            assert length == len(encoded_inputs['turn_ids'])
            assert length == len(encoded_inputs['role_ids'])
            assert length == len(encoded_inputs['position_ids'])
            assert 0 < length <= max_length
            self.data.append(encoded_inputs)
            self.data_bucket_ids.append(length)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SpanMLMProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer: DialogBertTokenizer, mlm_probability=0.15, mlm_ignore_idx=-100,
                 one_side_mask=False, mask_user_probability=0.5, mix_batch=False,
                 span_mask_probability=0.5, span_len_th=5):
        self.task_name = 'span_mlm'
        self.datasets = datasets
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/prefix_dialog')
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.one_side_mask = one_side_mask
        self.mask_user_probability = mask_user_probability
        self.mix_batch = mix_batch
        self.span_mask_probability = span_mask_probability
        self.span_len_th = span_len_th

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, use_multiwoz=False, do_train=True):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        total_samples = 0
        true_bio_dataset = ["schema", "multiwoz25", "taskmaster", "camrest", "frames", "m2m"]
        for dataset in self.datasets:
            print('load {} dataset:'.format(dataset))
            if dataset in true_bio_dataset:
                dev_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_dev.json')))
            else:
                print(dataset, 'pseudo_bio_tag')
                dev_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_dev_pseudo_bio.json')))
            dev_dataset = SpanMLMDataset(dev_data, self.tokenizer, max_length)
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
                    if dataset in true_bio_dataset:
                        train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                    else:
                        train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train_pseudo_bio.json')))
                    train_dataset = SpanMLMDataset(train_data, self.tokenizer, max_length)
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
            total_samples += len(train_data)

        print('dataset sample ratio')
        print(self.datasets)
        print(np.array(list(self.dataset_samples.values())) / np.sum(list(self.dataset_samples.values())))
        print('total train samples', total_samples)

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

                    masked_lm_labels: torch.tensor: (batch_size, max_seq_len)
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
        mask_user_side = None
        if self.one_side_mask and not self.mix_batch:
            mask_user_side = random.random() < self.mask_user_probability

        for i in range(batch_size):
            sen_len = batch_data[i]['length']
            attention_mask[i, :sen_len] = 1
            if self.one_side_mask and self.mix_batch:
                mask_user_side = random.random() < self.mask_user_probability
            masked_input, label = self._span_mask_token(batch_data[i]['input_ids'],
                                                        batch_data[i]['role_ids'],
                                                        batch_data[i]['bio_tag_ids'],
                                                        mask_user_side)
            input_ids[i, :sen_len] = torch.LongTensor(masked_input)
            masked_lm_labels[i, :sen_len] = torch.LongTensor(label)
            # input_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['input_ids'])
            turn_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['turn_ids'])
            role_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['role_ids'])
            position_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['position_ids'])

        return {"attention_mask": attention_mask,
                "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids, "position_ids": position_ids,
                "masked_lm_labels": masked_lm_labels,
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

    def _span_mask_token(self, input_ids, role_ids, bio_tag_ids, mask_user_side):
        """
        regard a whole span as a unit to mask
        """
        masked_input = list(input_ids)
        not_mask_tokens = self.tokenizer.all_special_tokens
        # print(not_mask_tokens)
        cand_indexes = []
        is_span = []
        in_span = False
        for i, (input_id, role_id, bio_tag_id) in enumerate(zip(input_ids, role_ids, bio_tag_ids)):
            if self.one_side_mask:
                if mask_user_side and role_id == 2:
                    # mask user, skip sys utt
                    continue
                elif not mask_user_side and role_id == 1:
                    # mask sys, skip user utt
                    continue
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
        all_spans = [x for x in cand_indexes if x[1] and len(x[0])<self.span_len_th]
        not_spans = [x for x in cand_indexes if not x[1] or len(x[0])>=self.span_len_th]
        cand_indexes = all_spans + not_spans
        # print(len(cand_indexes),cand_indexes)

        num2predict = max(1, int(round(sum(map(len, cand_indexes))) * self.mlm_probability))
        labels = [self.mlm_ignore_idx] * len(masked_input)
        masked_lms = []
        for index_set, is_span in cand_indexes:
            if len(masked_lms) >= num2predict:
                break
            if len(masked_lms) + len(index_set) > num2predict:
                continue
            if is_span and random.random() > self.span_mask_probability:
                continue
            r = random.random()
            if r < 0.8:
                masked_token = "[MASK]" if not is_span else "[VALUE]"
            elif 0.8 <= r < 0.9:
                masked_token = "original"
            else:
                masked_token = "replace"

            for index in index_set:
                if masked_token == "[MASK]":
                    new_id = self.tokenizer.convert_tokens_to_ids(masked_token)
                elif masked_token == "[VALUE]":
                    new_id = self.tokenizer.convert_tokens_to_ids(masked_token)
                elif masked_token == "original":
                    new_id = masked_input[index]
                elif masked_token == "replace":
                    # assert masked_token == "replace"
                    while True:
                        new_id = random.randint(0, self.tokenizer.vocab_size - 1)
                        if self.tokenizer.convert_ids_to_tokens(new_id) not in not_mask_tokens:
                            break
                labels[index] = masked_input[index]
                masked_input[index] = new_id
                masked_lms.append([index, new_id])

        # for index_set, is_span in cand_indexes[:max(1, int(round(len(cand_indexes) * self.mlm_probability)))]:
        #     if is_span:
        #         r = random.random()
        #         if r < 0.8:
        #             masked_token = "[MASK]"
        #         elif 0.8 <= r < 0.9:
        #             masked_token = "original"
        #         else:
        #             masked_token = "replace"
        #         # masked_token = "original"
        #     else:
        #         r = random.random()
        #         if r < 0.8:
        #             # 80% of the time, replace with [MASK]
        #             masked_token = "[MASK]"
        #         elif 0.8 <= r < 0.9:
        #             # 10% of the time, keep original
        #             masked_token = "original"
        #         else:
        #             # 10% of the time, replace with random word
        #             masked_token = "replace"
        #
        #     for index in index_set:
        #         if masked_token == "[MASK]":
        #             new_id = self.tokenizer.convert_tokens_to_ids(masked_token)
        #         elif masked_token == "[VALUE]":
        #             new_id = self.tokenizer.convert_tokens_to_ids(masked_token)
        #         elif masked_token == "original":
        #             new_id = masked_input[index]
        #         elif masked_token == "replace":
        #             # assert masked_token == "replace"
        #             while True:
        #                 new_id = random.randint(0, self.tokenizer.vocab_size - 1)
        #                 if self.tokenizer.convert_ids_to_tokens(new_id) not in not_mask_tokens:
        #                     break
        #         labels[index] = masked_input[index]
        #         masked_input[index] = new_id
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
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_120k_0831_bert')
    print(tokenizer.all_special_tokens)
    for spc_token in tokenizer.unique_added_tokens_encoder:
        print(spc_token, tokenizer.convert_tokens_to_ids(spc_token))
    # print(tokenizer.vocab_size)
    # print(tokenizer.tokenize("id love a Iced Caffè Mocha. Nope. Thatś all I need. Thank you."))
    # processor = SpanMLMProcessor(["schema", "multiwoz25", "taskmaster", "camrest", "frames", "m2m"], tokenizer)
    processor = SpanMLMProcessor(["multiwoz25", "schema"], tokenizer, mlm_probability=0.15, mlm_ignore_idx=-100,
                                 one_side_mask=True, mask_user_probability=0.5, mix_batch=False,
                                 span_mask_probability=0.5, span_len_th=5)
    processor.load_data(4, 4, 256, do_train=False)
    t = 0
    for batch in processor.yield_batches('schema', 4, 'dev'):
        # batch = processor.sample_batch('schema', 4, 'train')
        print(tokenizer.decode(batch['input_ids'][0]))
        print(tokenizer.convert_ids_to_tokens(batch['masked_lm_labels'][0]))
        print(tokenizer.decode(batch['input_ids'][1]))
        print(tokenizer.convert_ids_to_tokens(batch['masked_lm_labels'][1]))
        print()
        t+=1
        if t>3:
            break
    # tokenizer = DialogBertTokenizer.from_pretrained('/home/data/zhuqi/pre-trained-models/dialogbert/mlm/output_all_oneturn_25epoch_6.23')
    # print(tokenizer.convert_tokens_to_ids(['[USR]', '[SYS]', '[UNK]', '[DOMAIN]', '[SLOT]', '[VALUE]']))
    # processor = BIO_IntentProcessor(datasets=['schema', 'multiwoz25', 'taskmaster', 'camrest', 'frames', 'm2m'],
    #                                 tokenizer=tokenizer)
    # processor.load_data(train_batch_size=16, dev_batch_size=64)
    # processor = SSLProcessor(['multiwoz25'], tokenizer)
    # processor.load_data(2, 2, 512)
    # pprint(processor.sample_batch('multiwoz25', 2, 'train'))
    # for i in range(2):
    #     # dataset = processor.sample_dataset('train')
    #     dataset = 'multiwoz25'
    #     print('dataset:', dataset)
    #     batch = processor.sample_batch(dataset, 2, 'train')
    #     pprint(batch)
    #     # pprint(list(zip(batch['turn_ids'], batch['role_ids'], batch['position_ids'], batch['bio_tag_ids'], batch['bio_mask'], batch['intent_mask'])))
    #     # print(batch['dialogue'])
    #     # print(batch['intent_tag_ids'])
    #     print('='*100)
    # # pprint(processor.sample_batch('schema', 3, 'dev'))
    #
    # for i, batch in enumerate(processor.yield_batches('multiwoz25', 4, 'dev')):
    #     print(i, '='*100)
    #     print(batch)
    #     if i>1:
    #         break
