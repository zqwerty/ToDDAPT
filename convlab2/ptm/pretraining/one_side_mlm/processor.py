import os
import json
import random
import math
from convlab2.ptm.pretraining.dataloader import TaskProcessor, DialogBertSampler
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from convlab2.ptm.pretraining.mlm.processor import MLMDataset
from pprint import pprint
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np


class OneSideMLMProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer: DialogBertTokenizer, mlm_probability=0.15, mlm_ignore_idx=-100,
                 mask_user_probability=0.5, mix_batch=False):
        '''

        :param datasets:
        :param tokenizer:
        :param mlm_probability:
        :param mlm_ignore_idx: loss for special tokens are excluded, labels for such tokens are set to this value
        '''
        self.task_name = 'one_side_mlm'
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/prefix_dialog_cut59')
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.mask_user_probability = mask_user_probability
        self.mix_batch = mix_batch

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, use_multiwoz=False):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        total_samples = 0
        for dataset in self.datasets:
            print('load {} dataset:'.format(dataset))
            dev_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_dev.json')))
            dev_dataset = MLMDataset(dev_data, self.tokenizer, max_length)
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
            if dataset != 'multiwoz25' or use_multiwoz:
                train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                train_dataset = MLMDataset(train_data, self.tokenizer, max_length)
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
        :param batch_data: list of
                {
                    input_ids: list[int], reverse order
                    turn_ids: list[int]
                    role_ids: list[int]
                    position_ids: list[int]
                    length: int
                }
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
        # pprint(batch_data)
        batch_size = len(batch_data)
        max_seq_len = max([x['length'] for x in batch_data])
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        turn_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        role_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        masked_lm_labels = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx
        if not self.mix_batch:
            mask_user_side = random.random() < self.mask_user_probability

        for i in range(batch_size):
            sen_len = batch_data[i]['length']
            attention_mask[i, :sen_len] = torch.LongTensor([1] * sen_len)
            if self.mix_batch:
                mask_user_side = random.random() < self.mask_user_probability
            masked_input, label = self._one_side_wwm_tokens(batch_data[i]['input_ids'], batch_data[i]['role_ids'], mask_user_side)
            input_ids[i, :sen_len] = torch.LongTensor(masked_input)
            masked_lm_labels[i, :sen_len] = torch.LongTensor(label)
            turn_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['turn_ids'])
            role_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['role_ids'])
            position_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['position_ids'])

        return {"attention_mask": attention_mask,
                "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids, "position_ids": position_ids,
                "masked_lm_labels": masked_lm_labels}

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
    for spc_token in tokenizer.unique_added_tokens_encoder:
        print(spc_token, tokenizer.convert_tokens_to_ids(spc_token))
    print(len(tokenizer))
    print(tokenizer.tokenize("id love a Iced Caff?? Mocha. Nope. That?? all I need. Thank you."))
    processor = OneSideMLMProcessor(['camrest'], tokenizer, mlm_probability=1.0)
    processor.load_data(2, 2, 256)
    # print(processor.sample_batch('camrest', 2, 'train'))
    # print(processor.sample_batch('camrest', 2, 'train'))
    for batch in processor.yield_batches('camrest',2,'dev'):
        print(tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))
        print(tokenizer.convert_ids_to_tokens(batch["input_ids"][1]))
