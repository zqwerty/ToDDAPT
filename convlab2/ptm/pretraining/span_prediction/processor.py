import os
import json
import random
import math
import sys
sys.path.append('../../../../')
from convlab2.ptm.pretraining.dataloader import TaskProcessor, DialogBertSampler
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from pprint import pprint
import torch
import numpy as np

CLS_STR_TO_LABEL = {'none': 0, 'dontcare':1, 'delete':2, 'has_value':3}


class StateUpdatePretrainingDataset(Dataset):
    def __init__(self, dataset_name, slot_type, data, tokenizer: DialogBertTokenizer, max_length=512):
        """prepare input and label here

            self.data: A Dictionary of shape::
                {
                    input_mask: torch.tensor: (batch_size, max_seq_len)
                    input_ids: torch.tensor: (batch_size, max_seq_len)
                    turn_ids: torch.tensor: (batch_size, max_seq_len)
                    role_ids: torch.tensor: (batch_size, max_seq_len)
                    position_ids: torch.tensor: (batch_size, max_seq_len)

                    cls_label:  (batch_size)
                    value_token_mask: (batch_size, max_seq_len), optional
                    value_labels : (batch_size), optional
                    start: (batch_size), optional,
                    end: (batch_size), optional
                }

            self.data_bucket_ids: token length of dialogues
        """
        self.dataset = dataset_name
        self.data, self.data_bucket_ids = [], []
        self.slot_type = slot_type
        assert slot_type in ['categorical', 'non-categorical']

        if self.slot_type == 'categorical':
            for d in data:
                length = len(d['input_ids'])
                self.data_bucket_ids.append(len(d['input_ids']))
                d['length'] = length
                d['cls_label'] = CLS_STR_TO_LABEL[d['cls_label']]

                assert length == len(d['input_ids'])
                assert length == len(d['turn_ids'])
                assert length == len(d['role_ids'])
                assert length == len(d['position_ids'])
                assert 0 < length <= max_length, print(length, d['input_ids'])
                self.data.append(d)
        else:
            for d in data:
                length = len(d['input_ids'])
                self.data_bucket_ids.append(len(d['input_ids']))
                d['length'] = length
                d['cls_label'] = CLS_STR_TO_LABEL[d['cls_label']]
                if 'start' not in d:
                    d['start'] = -100  # ignored_index
                    d['end'] = -100

                assert length == len(d['input_ids'])
                assert length == len(d['turn_ids'])
                assert length == len(d['role_ids'])
                assert length == len(d['position_ids'])
                assert 0 < length <= max_length
                self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class StateUpdateProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer):
        self.task_name = 'state_update'
        self.datasets = datasets
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
        self.tokenizer = tokenizer


    def load_data(self, train_batch_size, dev_batch_size, max_length=512, num_workers=0):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.dataloaders = {}
        self.dataset_sample_weight = []
        self.slot_type_sample_weight = {}
        total_cat_samples = 0
        total_noncat_samples = 0
        for dataset in self.datasets:
            cat_train_data = json.load(open(os.path.join(self.data_dir, dataset + 'cat' + '_stateupdate_data_train.json')))
            cat_dev_data = json.load(open(os.path.join(self.data_dir, dataset + 'cat' + '_stateupdate_data_dev.json')))
            noncat_train_data = json.load(open(os.path.join(self.data_dir, dataset + 'noncat' + '_stateupdate_data_train.json')))
            noncat_dev_data = json.load(open(os.path.join(self.data_dir, dataset + 'noncat' + '_stateupdate_data_dev.json')))

            cat_train_dataset = StateUpdatePretrainingDataset(dataset, 'categorical', cat_train_data, self.tokenizer, max_length)
            cat_dev_dataset = StateUpdatePretrainingDataset(dataset, 'categorical', cat_dev_data, self.tokenizer, max_length)
            noncat_train_dataset = StateUpdatePretrainingDataset(dataset, 'non-categorical', noncat_train_data, self.tokenizer, max_length)
            noncat_dev_dataset = StateUpdatePretrainingDataset(dataset, 'non-categorical', noncat_dev_data, self.tokenizer, max_length)

            cat_train_sampler = DialogBertSampler(cat_train_dataset.data, cat_train_dataset.data_bucket_ids, train_batch_size, drop_last=False, replacement=True)
            cat_dev_sampler = DialogBertSampler(cat_dev_dataset.data, cat_dev_dataset.data_bucket_ids, dev_batch_size, drop_last=False, replacement=False)
            noncat_train_sampler = DialogBertSampler(noncat_train_dataset.data, noncat_train_dataset.data_bucket_ids,
                                                  train_batch_size, drop_last=False, replacement=True)
            noncat_dev_sampler = DialogBertSampler(noncat_dev_dataset.data, noncat_dev_dataset.data_bucket_ids, dev_batch_size,
                                                drop_last=False, replacement=False)

            cat_train_dataloader = DataLoader(
                cat_train_dataset,
                batch_sampler=cat_train_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            cat_dev_dataloader = DataLoader(
                cat_dev_dataset,
                batch_sampler=cat_dev_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            noncat_train_dataloader = DataLoader(
                noncat_train_dataset,
                batch_sampler=noncat_train_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            noncat_dev_dataloader = DataLoader(
                noncat_dev_dataset,
                batch_sampler=noncat_dev_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            print('load {} dataset:'.format(dataset))
            total_cat_samples += len(cat_train_data) + len(cat_dev_data)
            total_noncat_samples += len(noncat_train_data) + len(noncat_dev_data)
            self.dataloaders[dataset] = {'train': {'cat': cat_train_dataloader, 'noncat': noncat_train_dataloader},
                                         'dev': {'cat': cat_dev_dataloader, 'noncat': noncat_dev_dataloader}}
            self.dataset_sample_weight.append(len(cat_train_data) + len(noncat_train_data))
            self.slot_type_sample_weight[dataset] = [len(cat_train_data), len(noncat_train_data)]
        print('dataset sample ratio')
        print(self.datasets)
        print(np.array(self.dataset_sample_weight)/np.sum(self.dataset_sample_weight))
        print('total samples', total_cat_samples+total_noncat_samples) # total:535344 taskmaster:282483 multiwoz25:56168

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

                    cls_labels:  (batch_size)
                    value_token_masks: (batch_size, max_seq_len), optional
                    value_labels : (batch_size), optional
                    starts: (batch_size), optional
                    ends: (batch_size), optional
                }
        """
        batch_size = len(batch_data)
        max_seq_len = max([x['length'] for x in batch_data])
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        turn_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        role_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        cls_labels = torch.zeros((batch_size), dtype=torch.long)
        slot_token_indexes = torch.zeros((batch_size), dtype=torch.long)
        value_token_masks = None
        value_labels = None
        starts = None
        ends = None
        if 'start' not in batch_data[0]:
            # categorical slots
            value_token_masks = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
            value_labels = torch.zeros((batch_size), dtype=torch.long)
        else:
            starts = torch.zeros((batch_size), dtype=torch.long)
            ends = torch.zeros((batch_size), dtype=torch.long)

        for i in range(batch_size):
            sen_len = batch_data[i]['length']
            attention_mask[i, :sen_len] = torch.LongTensor([1] * sen_len)
            input_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['input_ids'])
            turn_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['turn_ids'])
            role_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['role_ids'])
            position_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['position_ids'])
            cls_labels[i] = batch_data[i]['cls_label']
            slot_token_indexes[i] = batch_data[i]['slot_token_idx']

            if 'start' not in batch_data[0]:
                value_token_masks[i, :sen_len] = torch.LongTensor(batch_data[i]['value_token_mask'])
                value_labels[i] = batch_data[i]['value_label']
            else:
                starts[i] = batch_data[i]['start']
                ends[i] = batch_data[i]['end']

        return {"attention_mask": attention_mask,
                "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids, "position_ids": position_ids,
                "cls_labels": cls_labels, "value_token_masks": value_token_masks, 'value_labels': value_labels,
                "starts": starts, "ends": ends, 'slot_token_index': slot_token_indexes}

    def sample_batch(self, dataset, batch_size, data_key):
        assert data_key == 'train'
        assert batch_size == self.train_batch_size, print('batch size must be the same as loaded data')
        sampled_slot_type = random.choices(list(self.dataloaders[dataset][data_key].keys()), self.slot_type_sample_weight[dataset])[0]
        dataloader = self.dataloaders[dataset][data_key][sampled_slot_type]
        for batch in dataloader:
            return batch

    def yield_batches(self, dataset, batch_size, data_key):
        assert data_key == 'dev'
        assert batch_size == self.dev_batch_size, print('batch size must be the same as loaded data')
        dataloaders = self.dataloaders[dataset][data_key].values()
        for dataloader in dataloaders:
            yield from dataloader

    def sample_dataset(self, data_key):
        return random.choices(self.datasets, self.dataset_sample_weight)[0]


if __name__ == '__main__':
    tokenizer = DialogBertTokenizer.from_pretrained('/home/data/zhuqi/pre-trained-models/dialogbert/mlm/output_all_oneturn_25epoch_6.23')
    # print(tokenizer.convert_tokens_to_ids(['[USR]', '[SYS]', '[UNK]', '[DOMAIN]', '[SLOT]', '[VALUE]']))
    # processor = BIO_IntentProcessor(datasets=['schema', 'multiwoz25', 'taskmaster', 'camrest', 'frames', 'm2m'],
    #                                 tokenizer=tokenizer)
    processor = StateUpdateProcessor(['multiwoz25'], tokenizer)
    processor.load_data(train_batch_size=16, dev_batch_size=2)
    pprint(processor.sample_batch('multiwoz25', 16, 'train'))
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
