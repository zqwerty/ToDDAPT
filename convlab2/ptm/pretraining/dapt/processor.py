import os
import json
import random
from convlab2.ptm.pretraining.dataloader import TaskProcessor
from convlab2.ptm.pretraining.model import DialogBertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np


def preprocess_schema(dataset_ontology):
    schema = {}
    for dataset, ontology in dataset_ontology.items():
        schema[dataset] = {"domains": {}, "intents": {}, 'domain_set': [], 'slot_set': [], 'intent_set': []}
        for domain, d_ontology in ontology['domains'].items():
            schema[dataset]['domain_set'].append(domain)
            domain_des = d_ontology["description"]
            schema[dataset]["domains"][domain] = {"description": domain_des,
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
        schema[dataset]['domain_dim'] = len(schema[dataset]['domain_set'])
        schema[dataset]['slot_dim'] = len(schema[dataset]['slot_set'])
        for intent, i_ontology in ontology["intents"].items():
            schema[dataset]['intent_set'].append(intent)
            intent_des = i_ontology["description"]
            schema[dataset]["intents"][intent] = {"description": intent_des}
        schema[dataset]['intent_dim'] = len(schema[dataset]['intent_set'])
    return schema


class DAPTDataset(Dataset):
    def __init__(self, data, max_length, tokenizer, mlm_ignore_idx, mlm_probability, is_train=True):
        """load full dialogue"""
        self.data = []
        more_than_maxlen = 0
        self.total_words = 0
        self.total_dials = 0
        for d in data:
            for t in d['dialogue']:
                if len(t) > max_length - 3:
                    break
            else:
                self.data.append(d)
                self.total_words += sum(map(len, d['dialogue']))
                if sum(map(len, d['dialogue'])) > max_length:
                    more_than_maxlen += 1
        self.max_length = max_length
        self.total_dials = len(self.data)
        self.random_gen = random.Random(42)
        self.tokenizer = tokenizer
        self.mlm_ignore_idx = mlm_ignore_idx
        self.mlm_probability = mlm_probability
        self.is_train = is_train
        # print('{}/{} dials exceed max length {}'.format(more_than_maxlen, self.total_dials, max_length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # clip the whole dial randomly
        dial = self.clip_dial(self.data[index], rand_start=False, rand_end=True)
        encoded_inputs = self.process_dial(dial)
        if self.mlm_probability > 0:
            masked_input, label = self._wwm_tokens(encoded_inputs['input_ids'])
            encoded_inputs['masked_input'] = masked_input
            encoded_inputs['label'] = label
        return encoded_inputs

    def reset_rand(self):
        # reset random seed for dev set
        if not self.is_train:
            self.random_gen = random.Random(42)

    def clip_dial(self, dial, rand_start=False, rand_end=False):
        if rand_end:
            end = self.random_gen.choice(range(1, dial["num_utt"] + 1, 2))  # select user turn to end
        else:
            end = dial["num_utt"]
        if rand_start:
            start = self.random_gen.choice(range(0, end, 2))  # select user turn to start
        else:
            start = 0
        true_start = end
        total_len = 1  # CLS token
        # add as much context as possible
        while true_start > start and total_len + len(dial["dialogue"][true_start - 1]) + 2 < self.max_length:
            true_start -= 1
            total_len += len(dial["dialogue"][true_start]) + 2  # [USR]/[SYS] + [SEP]

        # clip_dial = {x: (y[true_start:end] if isinstance(y, list) else y) for x, y in dial.items()}
        clip_dial = {"dialogue": dial["dialogue"][true_start:end], "num_utt": end - true_start, "dataset": dial["dataset"]}
        # print('clip to [{}, {}), start: {}'.format(true_start, end, start))
        return clip_dial

    def process_dial(self, dial):
        """extract label from dial['spans'] and dial['intent']"""
        encoded_inputs = self.tokenizer.prepare_input_seq(dial['dialogue'], last_role=dial.get('role', 'user'),
                                                          max_length=self.max_length, return_lengths=True)
        encoded_inputs['dataset'] = dial['dataset']
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
        self.random_gen.shuffle(cand_indexes)
        labels = [self.mlm_ignore_idx] * len(masked_input)
        for index_set in cand_indexes[:max(1, int(round(len(cand_indexes) * self.mlm_probability)))]:
            r = self.random_gen.random()
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
                        new_id = self.random_gen.randint(0, self.tokenizer.vocab_size - 1)
                        if self.tokenizer.convert_ids_to_tokens(new_id) not in not_mask_tokens:
                            break
                labels[index] = masked_input[index]
                masked_input[index] = new_id
        return masked_input, labels


class DAPTProcessor(TaskProcessor):
    def __init__(self, train_datasets, dev_datasets,
                 tokenizer: DialogBertTokenizer, mlm_probability=0.15, mlm_ignore_idx=-100,
                 train_ratio=1.0):
        """
        :param datasets: list of domain adaptive pretraining dataset
        :param tokenizer: instance of DialogBertTokenizer
        :param mlm_probability: default 0.15
        :param mlm_ignore_idx: loss for special tokens are excluded, labels for such tokens are set to this value
        :param biotagging: BIO-slot tagging
        :param sencls: sentence domain/intent/slot classification
        :param dialcls: dialogue domain/intent/slot classification
        """
        self.task_name = 'dapt'
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prepare_data/full_dialog_mlm')
        self.train_datasets = train_datasets
        self.datasets = dev_datasets
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_ignore_idx = mlm_ignore_idx
        self.train_ratio = train_ratio

    def load_data(self, train_batch_size, dev_batch_size, max_length=256, num_workers=0, do_train=True):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.max_length = max_length
        self.dev_dataloaders = {}
        self.train_dataloaders = {}
        self.dataset_samples = {}
        total_samples = 0
        total_words = 0
        for dataset in self.datasets:
            print('load {} dataset:'.format(dataset))
            with open(os.path.join(self.data_dir, dataset + '_data_dev.json')) as f:
                dev_data = json.load(f)
            dev_dataset = DAPTDataset(dev_data, max_length, self.tokenizer, self.mlm_ignore_idx, self.mlm_probability,
                                      is_train=False)
            dev_sampler = SequentialSampler(dev_dataset)
            dev_dataloader = DataLoader(
                dev_dataset,
                sampler=dev_sampler,
                batch_size=dev_batch_size,
                collate_fn=self.collate_fn,
                num_workers=num_workers
            )
            self.dev_dataloaders[dataset] = dev_dataloader
            print('\t dev: total samples {}.'.format(len(dev_dataset)))

        if do_train:
            for dataset in self.train_datasets:
                print('load {} dataset:'.format(dataset))
                train_data = json.load(open(os.path.join(self.data_dir, dataset + '_data_train.json')))
                train_data = train_data[:int(round(len(train_data) * self.train_ratio))]
                train_dataset = DAPTDataset(train_data, max_length, self.tokenizer, self.mlm_ignore_idx, self.mlm_probability)
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
                total_samples += train_dataset.total_dials
                total_words += train_dataset.total_words

        print('dataset sample ratio')
        print(self.train_datasets)
        print(np.array(list(self.dataset_samples.values())) / np.sum(list(self.dataset_samples.values())))
        print('total train dials', total_samples)
        print('total train words', total_words)

    def collate_fn(self, batch_data):
        """
        trans to pytorch tensor, pad batch
        :param batch_data: list of raw dials
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
        output_data = {}
        batch_size = len(batch_data)
        max_seq_len = max([x['length'] for x in batch_data])

        input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

        if self.mlm_probability > 0:
            masked_lm_labels = torch.ones((batch_size, max_seq_len), dtype=torch.long) * self.mlm_ignore_idx

        for i in range(batch_size):
            sen_len = batch_data[i]['length']
            if self.mlm_probability > 0:
                masked_input, label = batch_data[i]['masked_input'], batch_data[i]['label']
                input_ids[i, :sen_len] = torch.LongTensor(masked_input)
                masked_lm_labels[i, :sen_len] = torch.LongTensor(label)
            else:
                input_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['input_ids'])

        output_data.update({"attention_mask": input_ids > 0, "input_ids": input_ids})

        if self.mlm_probability > 0:
            output_data["masked_lm_labels"] = masked_lm_labels

        return output_data

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
        dataloader.dataset.reset_rand()

    def sample_dataset(self, data_key):
        assert data_key == 'train'
        return random.choices(list(self.dataset_samples.keys()), list(self.dataset_samples.values()))[0]


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(43)
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/augdial/mlm_wwm_120k_0831_bert')
    for spc_token in tokenizer.all_special_tokens:
        print(spc_token, tokenizer.convert_tokens_to_ids(spc_token))
    print(tokenizer.vocab_size)
    print(len(tokenizer.added_tokens_encoder))
    print(len(tokenizer))
    # train_datasets = ["schema", "taskmaster", "metalwoz", "woz", "camrest", "frames", "mdc", "kvret",
    #                   "banking", "oos", "hwu", "restaurant8k", "top", "m2m", "multiwoz21"]
    train_datasets = ["camrest"]
    # train_datasets = ["multiwoz25", "dstc2", "m2m", "oos", "hwu", "clinc", "banking", "restaurant8k", "top"]
    # dev_datasets = ["schema", "taskmaster", "metalwoz", "woz", "camrest", "frames", "mdc", "kvret",
    #                 "banking", "oos", "hwu", "restaurant8k", "top", "m2m", "multiwoz21"]
    dev_datasets = ["camrest"]
    processor = DAPTProcessor(train_datasets, dev_datasets, tokenizer, mlm_probability=0.15, mlm_ignore_idx=-100,
                              train_ratio=0.01)
    batch_size = 2
    processor.load_data(batch_size, batch_size, 256, do_train=True)
    dataset = 'camrest'
    show = True
    for batch in processor.yield_batches(dataset, batch_size, 'dev'):
        if show:
            for samplei in range(len(batch['input_ids'])):
                print(tokenizer.decode(batch['input_ids'][samplei]))
            show = False
    show = True
    for batch in processor.yield_batches(dataset, batch_size, 'dev'):
        if show:
            for samplei in range(len(batch['input_ids'])):
                print(tokenizer.decode(batch['input_ids'][samplei]))
            show = False
    show = True
    for batch in [processor.sample_batch(dataset, batch_size, 'train')]:
        if show:
            for samplei in range(len(batch['input_ids'])):
                print(tokenizer.decode(batch['input_ids'][samplei]))
            show = False
    show = True
    for batch in [processor.sample_batch(dataset, batch_size, 'train')]:
        if show:
            for samplei in range(len(batch['input_ids'])):
                print(tokenizer.decode(batch['input_ids'][samplei]))
            show = False
    # tb = [b for b in processor.yield_batches(dataset, batch_size, 'dev')]
    for batch in [processor.sample_batch(dataset, batch_size, 'train')]:
        # tb = [b for b in processor.yield_batches(dataset, batch_size, 'dev')]
        print(batch.keys())
        print(batch)
        for samplei in range(len(batch['input_ids'])):
            print(tokenizer.decode(batch['input_ids'][samplei]))
            tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][samplei])

            # token_tags = [
            #     ['B', 'I'][(x - 1) % 2] + '-' + processor.schema[dataset]['slot_set'][(x - 1) / 2] if x > 0 else
            #     {0: 'O', -100: 'X'}[x.item()] for x in batch['token_tag_ids'][samplei]]
            # domain_tags = batch['domain_tag_ids'][samplei]
            # intent_tags = batch['intent_tag_ids'][samplei]
            # slot_tags = batch['slot_tag_ids'][samplei]
            # for i, (token, bio_tag, domain_tag, intent_tag, slot_tag) in enumerate(
            #         zip(tokens, token_tags, domain_tags, intent_tags, slot_tags)):
            #     if i == 0:
            #         intents = []
            #         for j, intent in enumerate(processor.schema[dataset]['intent_set']):
            #             if 'cls_intent_tag_ids' in batch and batch['cls_intent_tag_ids'][samplei][j] > 0:
            #                 intents.append(intent)
            #         slots = []
            #         for j, slot in enumerate(processor.schema[dataset]['slot_set']):
            #             if 'cls_slot_tag_ids' in batch and batch['cls_slot_tag_ids'][samplei][j] > 0:
            #                 slots.append(slot)
            #         domains = []
            #         for j, domain in enumerate(processor.schema[dataset]['domain_set']):
            #             if 'cls_domain_tag_ids' in batch and batch['cls_domain_tag_ids'][samplei][j] > 0:
            #                 domains.append(domain)
            #         print(token, bio_tag, domains, intents, slots)
            #     if batch['sen_cls_mask'][samplei][i] > 0:
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
