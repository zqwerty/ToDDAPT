import torch
import torch.utils.data as data
# from .config import *
from .utils_function import to_cuda, merge, merge_multi_response, merge_sent_and_word
from .dataloader_base import DatasetBase


class Dataset_dm(DatasetBase):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, tokenizer, args, unified_meta, mode, max_length=512):
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
            dialog_history = self.data["dialog_history"][index]
            if len(dialog_history) > 0 and len(dialog_history[0]) == 0:
                dialog_history = dialog_history[1:]
            context = dialog_history + [self.data["turn_sys"][index], self.data["turn_usr"][index]]
            act_plain = self.data["sys_act"][index]
            act_one_hot = [0] * len(self.unified_meta["sysact"])
            for act in act_plain:
                act_one_hot[self.unified_meta["sysact"][act]] = 1
        else:
            print(f"Not Implemented {self.args['example_type']} for nlu yet...")
            raise ValueError

        item_info = {
            "context": self.preprocess(context, last_role='user'),
            "sysact": act_one_hot,
            "sysact_plain": act_plain,
            "turn_sys": self.preprocess(self.data["turn_sys"][index], last_role='system')
        }

        return item_info

    def __len__(self):
        return self.num_total_seqs

    def collate_fn(self, data):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        # data.sort(key=lambda x: len(x['context']), reverse=True)

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # # merge sequences
        # src_seqs, src_lengths = merge(item_info['context'])
        # turn_sys, _ = merge(item_info["turn_sys"])
        # sysact = torch.tensor(item_info["sysact"]).float()


        # item_info["context"] = to_cuda(src_seqs)
        # item_info["context_len"] = src_lengths
        # item_info["sysact"] = to_cuda(sysact)
        # item_info["turn_sys"] = to_cuda(turn_sys)

        # return item_info
        return self.merge_inputs(item_info['context']), self.merge_inputs(item_info['turn_sys']), torch.tensor(item_info["sysact"]).float().to(self.device)
