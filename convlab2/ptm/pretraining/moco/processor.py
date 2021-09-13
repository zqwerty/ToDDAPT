import os
import json
import random
import math
from convlab2.ptm.pretraining.dataloader import TaskProcessor, DialogBertSampler
from convlab2.ptm.pretraining.model import DialogBertTokenizer
from pprint import pprint
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import numpy as np
from tqdm import tqdm


class MOCODataset(Dataset):
    def __init__(self, data, utt_pool, tokenizer, max_length, keep_value=False):
        """prepare input and label here

            self.data: A Dictionary of shape::
                {
                    input_ids: list[int], reverse order
                    turn_ids: list[int]
                    role_ids: list[int]
                    position_ids: list[int]
                    length: int
                }
        """
        self.utt_pool = utt_pool
        self.data, self.data_bucket_ids = [], []
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.keep_value = keep_value
        for d in tqdm(data, desc="Loading dataset"):
            # encoded_inputs = tokenizer.prepare_input_seq(d['dialogue'], last_role='user', max_length=max_length, return_lengths=True)
            self.data.append({
                "utt_ids": d["utt_ids"],
                "da_list": d["da_list"],
                "num_utt": d["num_utt"]
            })
            self.data_bucket_ids.append(d["num_utt"])

    def __getitem__(self, index):
        data_item = self.data[index]
        a = random.randint(0, data_item["num_utt"] - 1)
        b = a
        total_len = 0
        while b < data_item["num_utt"] and total_len < self.max_length - 1: # -1 for cls token
            total_len += len(data_item["utt_ids"][b])
            b += 1
        dial_Q = data_item["utt_ids"][a:b]
        da_Q = data_item["da_list"][a:b]
        
        aa = random.randint(0, len(dial_Q) - 1)
        bb = random.randint(aa, len(dial_Q) - 1)

        dial_K = dial_Q[aa:bb]
        da_K = da_Q[aa:bb]

        # dial_K = dial_Q
        # da_K = da_Q

        dial_Q = self.augment(dial_Q, da_Q)
        dial_K = self.augment(dial_K, da_K)

        dial_Q = self.tokenizer.convert_utt_ids_to_input(dial_Q)
        dial_K = self.tokenizer.convert_utt_ids_to_input(dial_K)

        # assert len(dial_Q) < self.max_length, print("Len dial Q: {}".format(len(dial_Q)))
        # assert len(dial_K) < self.max_length, print("Len dial K: {}".format(len(dial_K)))

        return dial_Q, dial_K

    def __len__(self):
        return len(self.data)

    def augment(self, dials, das):
        aug_dial = []
        for dial, da in zip(dials, das):
            choice = random.choice(self.utt_pool[da])
            if self.keep_value:
                start = 0
                aug_utt = []
                for da_name, da_span in choice["da_spans"].items():
                    aug_utt.extend(choice["token_ids"][start:da_span["start"]])
                    if da_name in dial["da_spans"]:
                        aug_utt.extend(dial["token_ids"][dial["da_spans"][da_name]["start"]:dial["da_spans"][da_name]["end"]])
                    else:
                        aug_utt.extend(choice["token_ids"][da_span["start"]:da_span["end"]])
                    start = da_span["end"]
                aug_utt.extend(choice["token_ids"][start:])
                aug_dial.append(aug_utt)
            else:
                aug_dial.append(choice["token_ids"])

        return aug_dial

class MOCOProcessor(TaskProcessor):
    def __init__(self, datasets, tokenizer: DialogBertTokenizer, mlm_probability=0.15, mlm_ignore_idx=-100, keep_value=False):
        '''

        :param datasets:
        :param tokenizer:
        :param mlm_probability:
        :param mlm_ignore_idx: loss for special tokens are excluded, labels for such tokens are set to this value
        '''
        self.task_name = 'moco'
        # self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/prefix_dialog')
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/dialog_with_da')
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.keep_value = keep_value

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, use_multiwoz=False):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        total_samples = 0
        for dataset in self.datasets:
            print('load {} dataset:'.format(dataset))
            with open(os.path.join(self.data_dir, dataset + '_utt_pool.json')) as f:
                utt_pool = json.load(f)
            with open(os.path.join(self.data_dir, dataset + '_data_dev.json')) as f:
                dev_data = json.load(f)
            dev_dataset = MOCODataset(dev_data, utt_pool, self.tokenizer, max_length)
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
            if dataset != 'multiwoz25' or use_multiwoz:
                train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                train_dataset = MOCODataset(train_data, utt_pool, self.tokenizer, max_length, keep_value=self.keep_value)
                train_sampler = DialogBertSampler(train_dataset.data, train_dataset.data_bucket_ids, train_batch_size,
                                                  drop_last=True, replacement=False)
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_sampler=train_sampler,
                    collate_fn=self.collate_fn,
                    num_workers=num_workers
                )
                self.train_dataloaders[dataset] = train_dataloader
                self.dataset_samples[dataset] = len(train_data)
                print('\t train: total samples {}.'.format(len(train_dataset)))
            print('\t dev: total samples {}.'.format(len(dev_dataset)))
            total_samples += len(train_dataset)

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
        dial_Q = [x[0] for x in batch_data]
        dial_K = [x[1] for x in batch_data]
        batch_size = len(dial_Q)
        assert len(dial_K) == batch_size, print(len(dial_Q, len(dial_K)))

        output_data = {}

        for (k, data) in [("dial_q", dial_Q), ("dial_k", dial_K)]:
            batch_size = len(data)
            max_seq_len = max([x['length'] for x in data])
            attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            turn_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            role_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            # masked_lm_labels = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx

            for i in range(batch_size):
                sen_len = data[i]['length']
                attention_mask[i, :sen_len] = torch.LongTensor([1] * sen_len)
                # masked_input, label = self._wwm_tokens(data[i]['input_ids'])
                # input_ids[i, :sen_len] = torch.LongTensor(masked_input)
                # masked_lm_labels[i, :sen_len] = torch.LongTensor(label)
                input_ids[i, :sen_len] = torch.LongTensor(data[i]["input_ids"])
                turn_ids[i, :sen_len] = torch.LongTensor(data[i]['turn_ids'])
                role_ids[i, :sen_len] = torch.LongTensor(data[i]['role_ids'])
                position_ids[i, :sen_len] = torch.LongTensor(data[i]['position_ids'])
            
            output_data[k] = {
                "attention_mask": attention_mask,
                "input_ids": input_ids, 
                "turn_ids": turn_ids, 
                "role_ids": role_ids, 
                "position_ids": position_ids}

        return output_data

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
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_120k_0831_bert')
    for spc_token in tokenizer.all_special_tokens:
        print(spc_token, tokenizer.convert_tokens_to_ids(spc_token))
    # print(len(tokenizer))
    print(tokenizer.vocab_size)
    # print(tokenizer.all_special_tokens)
    # print(tokenizer.tokenize("id love a Iced Caffè Mocha. Nope. Thatś all I need. Thank you."))
    processor = MOCOProcessor(['schema'], tokenizer, mlm_probability=1.0)
    processor.load_data(2, 2, 256)
    # print(processor.sample_batch('camrest', 2, 'train')['input_ids'][0])
    # print(processor.sample_batch('camrest', 2, 'train')['input_ids'][0])
    for batch in processor.yield_batches('schema', 2, 'dev'):
        print(tokenizer.convert_ids_to_tokens(batch["Q"]['input_ids'][0]))
        print(tokenizer.convert_ids_to_tokens(batch["K"]['input_ids'][0]))
        exit(0)
