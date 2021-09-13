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


class SSLDataset(Dataset):
    def __init__(self, data, tokenizer: DialogBertTokenizer, max_length):
        """prepare input and label here

            self.data: A Dictionary of shape::
                {
                    input_ids: list[int], reverse order
                    turn_ids: list[int]
                    role_ids: list[int]
                    position_ids: list[int]

                    bio_tag_ids: list[int]

                    tf_idf: list[float]

                    token_mask: list[int]

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
            tf_idf = [0.]
            sen_idx = 1
            for tokens, sen_bio, sen_tf_idf in zip(d['dialogue'][::-1], d['bio_tag'][::-1], d['tf_idf'][::-1]):
                tags2add = ['X'] + sen_bio + ['X']
                if sen_idx + len(tokens) + 2 > length:
                    break
                bio_tags += tags2add
                tf_idf += [0.] + sen_tf_idf + [0.]
                sen_idx += len(tokens) + 2
            # print(d['dialogue'])
            bio_tag_ids = [bio_tag2id[tag] for tag in bio_tags]
            # encoded_inputs['dialogue'] = d['dialogue']
            token_mask = tokenizer.get_tokens_mask(encoded_inputs['input_ids'], mask_tokens=tokenizer.all_special_tokens)
            encoded_inputs['dataset'] = d['dataset']
            encoded_inputs['bio_tag_ids'] = bio_tag_ids
            encoded_inputs['token_mask'] = token_mask
            encoded_inputs['tf_idf'] = tf_idf
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


class SSLProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer,
                 mlm_probability=0.15, mlm_ignore_idx=-100, pseudo_bio=True, tf_idf=True):
        self.task_name = 'ssl'
        self.datasets = datasets
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/prefix_dialog')
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.pseudo_bio = pseudo_bio
        self.tf_idf = tf_idf
        self.is_train = True

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, use_multiwoz=False, do_train=True):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        total_samples = 0
        for dataset in self.datasets:
            print('load {} dataset:'.format(dataset))
            if self.pseudo_bio:
                dev_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_dev_pseudo_bio.json')))
            elif self.tf_idf:
                dev_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_dev_tf_allidf.json')))
            else:
                dev_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_dev.json')))
            dev_dataset = SSLDataset(dev_data, self.tokenizer, max_length)
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
                    if self.pseudo_bio:
                        train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train_pseudo_bio.json')))
                    elif self.tf_idf:
                        train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train_tf_allidf.json')))
                    else:
                        train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                    train_dataset = SSLDataset(train_data, self.tokenizer, max_length)
                    train_sampler = DialogBertSampler(train_dataset.data, train_dataset.data_bucket_ids, train_batch_size,
                                                      drop_last=False, replacement=False)
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
        :param batch_data: see SSLDataset
        :return: A Dictionary of shape::
                {
                    input_mask: torch.tensor: (batch_size, max_seq_len)
                    input_ids: torch.tensor: (batch_size, max_seq_len)
                    turn_ids: torch.tensor: (batch_size, max_seq_len)
                    role_ids: torch.tensor: (batch_size, max_seq_len)
                    position_ids: torch.tensor: (batch_size, max_seq_len)

                    bio_tag_ids: torch.tensor: (batch_size, max_seq_len)

                    tf_idf: torch.tensor: (batch_size, max_seq_len)

                    token_mask: torch.tensor: (batch_size, max_seq_len)
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
        bio_tag_ids = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx
        token_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tf_idf = torch.zeros((batch_size, max_seq_len), dtype=torch.float)

        for i in range(batch_size):
            sen_len = batch_data[i]['length']
            attention_mask[i, :sen_len] = 1
            masked_input, label = self._wwm_tokens(batch_data[i]['input_ids'])
            input_ids[i, :sen_len] = torch.LongTensor(masked_input)
            masked_lm_labels[i, :sen_len] = torch.LongTensor(label)
            turn_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['turn_ids'])
            role_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['role_ids'])
            position_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['position_ids'])

            bio_tag_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['bio_tag_ids'])

            token_mask[i, :sen_len] = torch.LongTensor(batch_data[i]['token_mask'])
            tf_idf[i, :sen_len] = torch.FloatTensor(batch_data[i]['tf_idf'])

        # predict bio on only masked locations?
        # bio_tag_ids[masked_lm_labels==-100] = -100

        # if batch_data[0]['dataset'] not in ['taskmaster', 'schema']:
        #     return {"attention_mask": attention_mask,
        #             "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids, "position_ids": position_ids,
        #             "masked_lm_labels": masked_lm_labels,
        #             # "bio_tag_ids": bio_tag_ids
        #             }

        return {"attention_mask": attention_mask,
                "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids, "position_ids": position_ids,
                "masked_lm_labels": masked_lm_labels,
                "bio_tag_ids": bio_tag_ids,
                "tf_idf": tf_idf,
                "token_mask": token_mask
                }

    def _wwm_tokens(self, input_ids):
        masked_input = list(input_ids)
        not_mask_tokens = self.tokenizer.all_special_tokens
        cand_indexes = []
        for (i, input_id) in enumerate(masked_input):
            token = self.tokenizer._convert_id_to_token(input_id)
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
                        if self.tokenizer._convert_id_to_token(new_id) not in not_mask_tokens:
                            break
                labels[index] = masked_input[index]
                masked_input[index] = new_id
        return masked_input, labels

    def _wwm_with_span_tokens(self, input_ids):
        masked_input = list(input_ids)
        not_mask_tokens = self.tokenizer.all_special_tokens
        cand_indexes = []
        for (i, input_id) in enumerate(masked_input):
            token = self.tokenizer._convert_id_to_token(input_id)
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
                        if self.tokenizer._convert_id_to_token(new_id) not in not_mask_tokens:
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
        assert data_key == 'dev'
        assert batch_size == self.dev_batch_size, print('batch size must be the same as loaded data')
        dataloader = self.dev_dataloaders[dataset]
        for batch in dataloader:
            yield batch

    def sample_dataset(self, data_key):
        assert data_key == 'train'
        return random.choices(list(self.dataset_samples.keys()), list(self.dataset_samples.values()))[0]


if __name__ == '__main__':
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(42)
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_120k_0831_bert')
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[INTENT]']})
    for spc_token in tokenizer.unique_added_tokens_encoder:
        print(spc_token, tokenizer.convert_tokens_to_ids(spc_token))
    print(tokenizer.all_special_tokens)
    print(tokenizer.unique_added_tokens_encoder)
    # print(len(tokenizer))
    processor = SSLProcessor(["camrest"], tokenizer, mlm_probability=0.15, pseudo_bio=False, tf_idf=True)
    processor.load_data(2, 2, 256)
    for batch in processor.yield_batches('camrest', 2, 'dev'):
        print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
        print(batch['masked_lm_labels'][0])
        print(batch['bio_tag_ids'][0])
        print(batch['tf_idf'][0])
        print(batch['token_mask'][0])
        break
