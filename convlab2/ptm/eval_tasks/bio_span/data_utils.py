import random
from collections import Counter
import math
import numpy as np
import torch
from tqdm import trange


class Dataloader:
    def __init__(self, processor, use_weighted_tag_loss=False):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param prepare_input_fn: prepare input before training, including append context
        :param batch_fn: function that prepare batch input when training
        """

        self.processor = processor
        self.data = {}
        self.id2tag = dict([(i, x) for i, x in enumerate(processor.labels_map['id_to_label'])])
        self.tag2id = dict([(x, i) for i, x in enumerate(processor.labels_map['id_to_label'])])
        self.tag_dim = len(self.id2tag)
        self.tag_weight = [1] * len(self.id2tag)
        self.use_weighted_tag_loss = use_weighted_tag_loss

    def load_data(self, data, data_key):
        """

        :param data_key: train/val/test
        :param data: preprocessed data
        :return:
        """
        self.data[data_key] = data
        sen_len = [len(seq['input_ids']) for seq in data]

        if data_key == 'train' and self.use_weighted_tag_loss:
            for d in self.data[data_key]:
                for tag_id in d['bio_labels']:
                    self.tag_weight[tag_id] += 1

            total_weight = sum(self.tag_weight)

            for tag_id in range(len(self.tag_weight)):
                self.tag_weight[tag_id] = np.log10(total_weight / self.tag_weight[tag_id])
            self.tag_weight = torch.tensor(self.tag_weight)
        # print(sorted(Counter(sen_len).items()))

    def seq_tag2id(self, tags):
        return [self.tag2id[x] for x in tags]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def pad_batch(self, batch_data):
        return self.processor.batch_fn(batch_data)

    def get_train_batch(self, batch_size):
        # d = ('user'/'sys', tokens, tags, intents, dialog_act, context,
        # tokenized_tokens, tokenized_tags, tags_id, intents_id)
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in trange(batch_num, desc='eval'):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
