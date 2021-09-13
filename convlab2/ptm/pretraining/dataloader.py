"""
Dataloaders for different pre-training task and datasets
"""
import random
import math
from typing import List
import itertools
import torch
from torch.utils.data.dataset import Dataset
from convlab2.ptm.pretraining.model.tokenization_dialog_bert import DialogBertTokenizer
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler, SequentialSampler


class MetaDataloader:
    def __init__(self, task_processors):
        """
        wrapper for task_processors
        :param task_processors: a dict of task_processor {task_name: task_processor, ...}.
        """
        self.processors = task_processors
        self.tasks = list(task_processors.keys())

    def sample_batch(self, task, dataset, batch_size, data_key):
        """
        sample a batch using the task processor
        :param task: task name, e.g. 'mlm'
        :param dataset: dataset name, e.g. 'schema'
        :param batch_size:
        :param data_key: train/val/test
        :return:
        """
        batch_data = self.processors[task].sample_batch(dataset, batch_size, data_key)
        return batch_data

    def yield_batches(self, task, dataset, batch_size, data_key):
        """
        yield a batch each time for evaluation
        """
        for batch_data in self.processors[task].yield_batches(dataset, batch_size, data_key):
            yield batch_data

    def sample_dataset(self, task, data_key):
        return self.processors[task].sample_dataset(data_key)


class TaskProcessor:
    """
    Base class for different task processors
    """
    def load_data(self, **kwargs):
        """
        load preprocessed data
        """
        pass

    def sample_batch(self, dataset, batch_size, data_key):
        """
        sample a batch from a dataset[data_key]
        :param dataset: string that indicate the dataset
        :param batch_size:
        :param data_key: train/val/test
        :return:
        """
        raise NotImplementedError

    def yield_batches(self, dataset, batch_size, data_key):
        """
        yield a batch from a dataset[data_key] each time
        """
        raise NotImplementedError

    def sample_dataset(self, data_key):
        """
        sample a dataset from all datasets
        """
        raise NotImplementedError


class DummyTaskProcessor(TaskProcessor):
    def __init__(self):
        self.task_name = 'dummy'

    def load_data(self):
        dummy_data1 = {
            'train': [i for i in range(1000)],
            'val': [i for i in range(1000)],
            'test': [i for i in range(1000)]
        }
        dummy_data2 = {
            'train': [-i for i in range(100)],
            'val': [-i for i in range(100)],
            'test': [-i for i in range(100)]
        }
        self.data = {
            'dataset1': dummy_data1,
            'dataset2': dummy_data2
        }

    def sample_batch(self, dataset, batch_size, data_key):
        batch_data = random.choices(self.data[dataset][data_key], k=batch_size)
        return batch_data

    def yield_batches(self, dataset, batch_size, data_key):
        batch_num = math.ceil(len(self.data[dataset][data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[dataset][data_key][i * batch_size:(i + 1) * batch_size]
            yield batch_data


class DialogBertSampler(Sampler):
    """
    dialogs contains same turn should have similar length.
    group dialogs by their turns, sample turn_length and then sample dialogs.

    Arguments:
        data_source (PretrainingDataset): dataset to sample from
        data_source_bucket_ids: List[int], data_source[i] should be put to data_source_bucket_ids[i] th bucket
        drop_last: whether drop last examples in each bucket
    """

    def __init__(self, data_source, data_source_bucket_ids, batch_size, drop_last, replacement):
        super().__init__(data_source)
        self.data_source = data_source
        self.data_source_bucket_ids = data_source_bucket_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.replacement = replacement
        assert len(data_source) == len(data_source_bucket_ids)
        # sort examples by bucket ids
        self.buckets_to_idx = {}
        for data_idx, bucket_idx in enumerate(data_source_bucket_ids):
            self.buckets_to_idx.setdefault(bucket_idx, [])
            self.buckets_to_idx[bucket_idx].append(data_idx)
        self.sorted_bucket_ids = sorted(list(self.buckets_to_idx.keys()))
        self.first_batch = True

        if self.replacement:
            for k in self.buckets_to_idx:
                random.shuffle(self.buckets_to_idx[k])
            self.sorted_examples = list(itertools.chain(*[self.buckets_to_idx[bid] for bid in self.sorted_bucket_ids]))
        else:
            batched_examples = []
            batch = []
            for i in range(len(self.data_source)):
                batch.append(i)
                if len(batch) == self.batch_size:
                    batched_examples.append(batch)
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                batched_examples.append(batch)
            self.batched_examples = batched_examples
            self.sorted_examples = list(itertools.chain(*[self.buckets_to_idx[bid] for bid in self.sorted_bucket_ids]))

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        if self.replacement:
            start_idx = random.randint(0, len(self.sorted_examples)-self.batch_size)
            bid = self.data_source_bucket_ids[self.sorted_examples[start_idx]]
            random.shuffle(self.buckets_to_idx[bid])
            self.sorted_examples = list(itertools.chain(*[self.buckets_to_idx[bid] for bid in self.sorted_bucket_ids]))
            if self.first_batch:
                self.first_batch = False
                yield self.sorted_examples[-self.batch_size:]
            else:
                yield self.sorted_examples[start_idx:start_idx+self.batch_size]
        else:
            for batch in SubsetRandomSampler(self.batched_examples):
                yield [self.sorted_examples[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    # dummy_processor1 = DummyTaskProcessor()
    # dummy_processor2 = DummyTaskProcessor()
    # dummy_processor2.task_name = 'dummy2'
    # dummy_processor1.load_data()
    # dummy_processor2.load_data()
    # task_processors = {x.task_name:x for x in [dummy_processor1, dummy_processor2]}
    # tasks = list(task_processors.keys())
    # datasets = ['dataset1', 'dataset2']
    # metadataloader = MetaDataloader(task_processors)
    # max_steps = 10
    # for i in range(10):
    #     for task in tasks:
    #         datasets_ratio = [len(task_processors[task].data[x]['train']) for x in datasets]
    #         dataset = random.choices(datasets, datasets_ratio)[0]
    #         batch_data = metadataloader.sample_batch(task, dataset, 10, 'train')
    #         # train batch
    #         # print(datasets_ratio)
    #         print(task)
    #         print('\t',dataset)
    #         print('\t',batch_data)
    #
    # for task in tasks:
    #     for dataset in datasets:
    #         for batch_data in metadataloader.yield_batches(task, dataset, 300, 'test'):
    #             # eval batch
    #             print(task, dataset, len(batch_data))

    data = list(range(20))
    bucket = [i // 5 for i in data]
    # data = [1, 123, 342,1213, 65646, 5321,344]
    # bucket = [0, 1, 0, 2, 0, 1, 0]
    sampler = DialogBertSampler(data, bucket, 2, False, True)
    for b in sampler:
        print(b)
        break
    print(sampler.first_batch)
    for b in sampler:
        print(b)
