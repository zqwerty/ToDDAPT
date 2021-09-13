# sampler for dialogbert pretraining
# first sample turn using weighted sampling, then sample a batch

import torch
import random
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler, SequentialSampler
import torch.distributed as dist
from collections import Counter
import itertools
import math


class DialogBertSampler(Sampler):
    r"""
    dialogs contains same turn should have similar length.
    group dialogs by their turns, sample turn_length and then sample dialogs.

    Arguments:
        data_source (Dataset): dataset to sample from
        data_source_bucket_ids: List[int], data_source[i] should be put to data_source_bucket_ids[i] th bucket
        drop_last: whether drop last examples in each bucket
    """

    def __init__(self, data_source, data_source_bucket_ids, batch_size, drop_last):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        assert len(data_source) == len(data_source_bucket_ids)
        # sort examples by bucket ids
        self.bucket_count = Counter(data_source_bucket_ids)
        self.buckets_to_idx = {k: [] for k in self.bucket_count}
        for data_idx, bucket_idx in enumerate(data_source_bucket_ids):
            self.buckets_to_idx[bucket_idx].append(data_idx)
        self.sorted_bucket_ids = sorted([k for k in self.bucket_count.keys()])

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        for k in self.buckets_to_idx:
            random.shuffle(self.buckets_to_idx[k])
        # print(self.buckets_to_idx)
        sorted_examples = list(itertools.chain(*[self.buckets_to_idx[bid] for bid in self.sorted_bucket_ids]))
        for batch in SubsetRandomSampler(
                list(BatchSampler(SequentialSampler(sorted_examples), self.batch_size, self.drop_last))):
            yield [sorted_examples[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class DistributedDialogBertSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, data_source, data_source_bucket_ids, batch_size, num_replicas=None, rank=None):
        super().__init__(data_source)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.data_source) * 1.0 / self.num_replicas))
        self.batch_size = batch_size
        # batches are distributed to multi-gpus, pad batches to min{total} s.t. total % (num_replicas * batch_size) == 0
        # first pad examples s.t # example % batch_size == 0
        # then pad batches s.t # batch % num_replica == 0
        self.total_batch = (len(self.data_source) + self.batch_size - 1) // self.batch_size
        self.num_pad_examples = self.total_batch * self.batch_size - len(self.data_source)
        self.num_samples = int(math.ceil(self.total_batch * 1.0 / self.num_replicas))
        self.num_pad_batches = self.num_replicas * self.num_samples - self.total_batch

        # for bucket sampling
        assert len(data_source) == len(data_source_bucket_ids)
        # sort examples by bucket ids
        self.bucket_count = Counter(data_source_bucket_ids)
        self.buckets_to_idx = {k: [] for k in self.bucket_count}
        for data_idx, bucket_idx in enumerate(data_source_bucket_ids):
            self.buckets_to_idx[bucket_idx].append(data_idx)
        self.sorted_bucket_ids = sorted([k for k in self.bucket_count.keys()])

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.random.seed(self.epoch)
        random.seed(self.epoch)

        for k in self.buckets_to_idx:
            random.shuffle(self.buckets_to_idx[k])

        sorted_examples = list(itertools.chain(*[self.buckets_to_idx[bid] for bid in self.sorted_bucket_ids]))
        sorted_examples += sorted_examples[:self.num_pad_examples]

        batch_idxes = list(SubsetRandomSampler(list(BatchSampler(SequentialSampler(sorted_examples), self.batch_size, drop_last=True))))
        batch_idxes += batch_idxes[:self.num_pad_batches]

        indices = batch_idxes[self.rank:self.total_batch:self.num_replicas]
        for _indices in indices:
            yield [sorted_examples[i] for i in _indices]

    def __len__(self):
        return self.total_batch

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == '__main__':
    data = list(range(20))
    bucket = [i // 3 for i in data]
    # data = [1, 123, 342,1213, 65646, 5321,344]
    # bucket = [0, 1, 0, 2, 0, 1, 0]
    # sampler = DialogBertSampler(data, bucket, 2, False)
    sampler = DistributedDialogBertSampler(data, bucket, 2)
    for b in sampler:
        print(b)

