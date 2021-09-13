from abc import ABC
from typing import *

import torch
from torch.utils import data

from convlab2.ptm.pretraining.model import DialogBertTokenizer


class DatasetBase(data.Dataset, ABC):
    def __init__(self, args, tokenizer: DialogBertTokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args
        self.device = torch.device(args['device'])

    def preprocess(self, sequence: List[str], last_role):
        tokens = [self.tokenizer.tokenize(s) for s in sequence]
        encoding = self.tokenizer.prepare_input_seq(tokens, max_length=self.max_length, last_role=last_role)
        return encoding

    # inputs must not be empty
    def merge_inputs(self, inputs_list):
        ret = {}
        max_len = max(len(encoding['input_ids']) for encoding in inputs_list)
        for inputs in inputs_list:
            for k, v in inputs.items():
                ret.setdefault(k, []).append(v + [self.tokenizer.pad_token_id] * (max_len - len(v)))
        for k, v in ret.items():
            ret[k] = torch.tensor(v).to(self.device)
        return ret
