import os
import json
import random
import math
from convlab2.ptm.pretraining.dataloader import TaskProcessor
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from pprint import pprint
from copy import deepcopy
from collections import Counter
from itertools import zip_longest
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler

import numpy as np
from tqdm import tqdm


class AugDialSSLDataset(Dataset):
    def __init__(self, data, max_length, pos_aug_num=0, neg_aug_num=0, pick1utt_num=1):
        """load full dialogue"""
        self.data = []
        for d in data:
            for t in d['dialogue']:
                if len(t) > max_length - 3:
                    break
            else:
                self.data.append(d)
        self.max_length = max_length
        self.pos_aug_num = pos_aug_num
        self.neg_aug_num = neg_aug_num
        self.pick1utt_num = pick1utt_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # clip the whole dial randomly
        data_item = self.data[index]
        # ori_dial = self.clip_dial(data_item, rand_start=False, rand_end=True)
        ori_dial = data_item
        outputs = [ori_dial]
        for _ in range(self.pos_aug_num):
            outputs.append(self.unsupervised_augment(ori_dial, positive=True))
        for _ in range(self.neg_aug_num):
            outputs.append(self.unsupervised_augment(ori_dial, positive=False))
        for _ in range(self.pick1utt_num):
            outputs.append(self.unsupervised_augment(self.pick1utt(ori_dial), positive=True))

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
        while true_start > start and total_len + len(dial["dialogue"][true_start - 1]) + 2 < self.max_length:
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
        # replace ori sen with mask
        # dial['dialogue'][utt_idx] = ['[MASK]']
        assert len(dial['dialogue'][utt_idx]) + 3 <= self.max_length
        return clip_dial

    def unsupervised_augment(self, dial, positive):
        if positive:
            return dial
        else:
            num_utt = dial["num_utt"]
            all_utt_idx2replace = list(range(num_utt))
            random.shuffle(all_utt_idx2replace)
            # replace 50% utt randomly
            # ratio = random.uniform(0.2, 0.8)
            ratio = 0
            # print(int(round(num_utt * ratio)))
            all_utt_idx2replace = all_utt_idx2replace[:max(1, int(round(num_utt * ratio)))]
            for utt_idx2replace in all_utt_idx2replace:
                if utt_idx2replace % 2 == 0:
                    role = "user"
                    da_list = random.choices(self.user_da_seq, weights=self.user_da_cnt)[0]
                else:
                    role = "system"
                    da_list = random.choices(self.system_da_seq, weights=self.system_da_cnt)[0]
                template = random.choice(self.utt_pool[role][da_list])

                dial["dialogue"][utt_idx2replace] = template["utterance"]
            return dial


class AugDialProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer: DialogBertTokenizer, mlm_probability=0.15, mlm_ignore_idx=-100,
                 pos_aug_num=1, neg_aug_num=1, pick1utt_num=1, train_ratio=1.0):
        '''

        :param datasets:
        :param tokenizer:
        :param mlm_probability:
        :param mlm_ignore_idx: loss for special tokens are excluded, labels for such tokens are set to this value
        '''
        self.task_name = 'augdial'
        # self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/prefix_dialog')
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/full_dialog')
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.pos_aug_num = pos_aug_num
        self.neg_aug_num = neg_aug_num
        self.pick1utt_num = pick1utt_num
        self.train_ratio = train_ratio

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, do_train=True):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.max_length = max_length
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        total_samples = 0
        for dataset in self.datasets:
            print('load {} dataset:'.format(dataset))
            with open(os.path.join(self.data_dir, dataset + '_data_dev.json')) as f:
                dev_data = json.load(f)
            dev_dataset = AugDialSSLDataset(dev_data, max_length, pos_aug_num=self.pos_aug_num,
                                            neg_aug_num=self.neg_aug_num, pick1utt_num=self.pick1utt_num)
            dev_sampler = RandomSampler(dev_dataset)
            dev_dataloader = DataLoader(
                dev_dataset,
                sampler=dev_sampler,
                batch_size=dev_batch_size,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            self.dev_dataloaders[dataset] = dev_dataloader
            train_data = []
            if do_train:
                train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                train_dataset = AugDialSSLDataset(train_data, max_length, pos_aug_num=self.pos_aug_num,
                                                  neg_aug_num=self.neg_aug_num, pick1utt_num=self.pick1utt_num)
                train_sampler = RandomSampler(train_dataset)
                train_dataloader = DataLoader(
                    train_dataset,
                    sampler=train_sampler,
                    batch_size=train_batch_size,
                    collate_fn=self.collate_fn,
                    num_workers=num_workers
                )
                self.train_dataloaders[dataset] = train_dataloader
                self.dataset_samples[dataset] = len(train_data)
                print('\t train: total samples {}.'.format(len(train_dataset)))
                total_samples += len(train_dataset)
            print('\t dev: total samples {}.'.format(len(dev_dataset)))
            total_samples += len(dev_dataset)

        print('dataset sample ratio')
        print(self.datasets)
        print(np.array(list(self.dataset_samples.values())) / np.sum(list(self.dataset_samples.values())))
        print('total train samples', total_samples)

    def collate_fn(self, batch_data):
        """
        trans to pytorch tensor, pad batch
        :param batch_data: list of tuples(ori_dial, aug_dial1, ...)
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
        # return batch_data[0]
        input_batches = []
        for i, dials in enumerate(zip(*batch_data)):  # ori_dials, aug_dials1, aug_dials2,...
            input_batch = []
            for dial in dials:
                encoded_inputs = self.tokenizer.prepare_input_seq(dial['dialogue'], last_role=dial.get('role', 'user'),
                                                                  max_length=self.max_length, return_lengths=True)
                input_batch.append(encoded_inputs)
            input_batches.append(input_batch)

        assert len(input_batches) == (1 + self.pos_aug_num + self.neg_aug_num + self.pick1utt_num), print(len(input_batches))

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

            output_data.append({"attention_mask": attention_mask,
                                "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids,
                                "position_ids": position_ids,
                                })

            if self.mlm_probability > 0 and batchi==0:
                output_data[-1]["masked_lm_labels"] = masked_lm_labels

        return {"batches": output_data}

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
    datasets = ['taskmaster']
    processor = AugDialProcessor(datasets, tokenizer, pos_aug_num=0, neg_aug_num=0, pick1utt_num=1)
    batch_size = 4
    processor.load_data(batch_size, batch_size, 256, do_train=False)
    dataset = 'taskmaster'
    for batch in processor.yield_batches(dataset, batch_size, 'dev'):
    # for batch in [processor.sample_batch(dataset, batch_size, 'train')]:
        batches = batch["batches"]
    #     print(torch.max(batches[0]["turn_ids"], dim=-1))
        for augi, b in enumerate(batches):
            print("aug {}".format(augi) + '-'*100)
            print(b.keys())
            for samplei in range(batch_size):
                print(tokenizer.decode(b['input_ids'][samplei]))
                tokens = tokenizer.convert_ids_to_tokens(b['input_ids'][samplei])
                    # break
                # break
        break
