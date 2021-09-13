import torch
import torch.utils.data as data
# from .config import *
from .utils_function import merge, merge_multi_response, merge_sent_and_word
from convlab2.ptm.pretraining.model import DialogBertTokenizer

from .dataloader_base import DatasetBase


class Dataset_nlu(DatasetBase):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, tokenizer: DialogBertTokenizer, args, unified_meta, mode, max_length=512):
        super().__init__(args, tokenizer, max_length)
        """Reads source and target sequences from txt files."""
        self.data = data_info
        self.tokenizer = tokenizer
        self.num_total_seqs = len(data_info["ID"])
        self.usr_token = args["usr_token"]
        self.sys_token = args["sys_token"]
        self.max_length = max_length
        self.unified_meta = unified_meta

        if "bert" in self.args["model_type"] or "electra" in self.args["model_type"]:
            self.start_token = self.tokenizer.cls_token
            self.sep_token = self.tokenizer.sep_token
        else:
            self.start_token = self.tokenizer.bos_token
            self.sep_token = self.tokenizer.eos_token

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""

        if self.args["example_type"] == "turn":
            context = [self.data["turn_usr"][index]]
            encoding = self.preprocess(context, 'user')
            intent_plain = self.data["intent"][index]
            try:
                intent_idx = self.unified_meta["intent"][intent_plain]
            except:
                intent_idx = -100

            try:
                domain_idx = self.unified_meta["turn_domain"][self.data["turn_domain"][index]]
            except:
                domain_idx = -100

            try:
                turn_slot_one_hot = [0] * len(self.unified_meta["turn_slot"])
                for ts in self.data["turn_slot"][index]:
                    turn_slot_one_hot[self.unified_meta["turn_slot"][ts]] = 1
            except:
                turn_slot_one_hot = -100
        else:
            raise NotImplementedError(f"Not Implemented {self.args['example_type']} for nlu yet...")

        item_info = {
            "ID": self.data["ID"][index],
            "context": torch.tensor(encoding['input_ids']),
            "turn_id": torch.tensor(encoding["turn_ids"]),
            "role_id": torch.tensor(encoding["role_ids"]),
            "position_id": torch.tensor(encoding["position_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "intent": intent_idx,
            "intent_plain": intent_plain,
        }

        return item_info

    def __len__(self):
        return self.num_total_seqs

    def collate_fn(self, data):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['context']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        intent = torch.tensor(item_info["intent"])
        inputs = {
            'input_ids': merge(item_info['context'])[0].to(self.device),
            'turn_ids': merge(item_info['turn_id'])[0].to(self.device),
            'role_ids': merge(item_info['role_id'])[0].to(self.device),
            'position_ids': merge(item_info['position_id'])[0].to(self.device),
            'attention_mask': merge(item_info['attention_mask'])[0].to(self.device),
        }

        intent = intent.to(self.device)
        return inputs, intent


def collate_fn_nlu_dial(data):
    # TODO
    return
