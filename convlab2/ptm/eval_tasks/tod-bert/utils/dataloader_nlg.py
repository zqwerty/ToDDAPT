import torch
import torch.utils.data as data
import random
from .dataloader_base import DatasetBase

from .utils_function import to_cuda, merge

from convlab2.ptm.pretraining.model import DialogBertTokenizer
# from .config import *


class Dataset_nlg(DatasetBase):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, tokenizer: DialogBertTokenizer, args, unified_meta, mode, max_length=512, max_sys_resp_len=50):
        super().__init__(args, tokenizer, max_length)
        """Reads source and target sequences from txt files."""
        self.data = data_info
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_total_seqs = len(data_info["ID"])
        self.usr_token = args["usr_token"]
        self.sys_token = args["sys_token"]
        self.unified_meta = unified_meta
        self.mode = mode

        if "bert" in self.args["model_type"] or "electra" in self.args["model_type"]:
            self.start_token = self.tokenizer.cls_token
            self.sep_token = self.tokenizer.sep_token
        else:
            self.start_token = self.tokenizer.bos_token
            self.sep_token = self.tokenizer.eos_token

        self.resp_cand_trn = list(self.unified_meta["resp_cand_trn"])
        random.shuffle(self.resp_cand_trn)
        self.max_sys_resp_len = max_sys_resp_len
        self.others = unified_meta["others"]

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""

        if self.args["example_type"] == "turn":
            dialog_history = self.data["dialog_history"][index]
            if len(dialog_history) > 0 and len(dialog_history[0]) == 0:
                dialog_history = dialog_history[1:]
            context = dialog_history
            # context_plain = " ".join(context)
            # reversed_context = list(reversed(context))
            # reverse_context_plain = " ".join(reversed_context)
            context_plain = self.get_concat_context(self.data["dialog_history"][index])
            # context = self.preprocess(context_plain)
            response_plain = self.data["turn_sys"][index]
        else:
            raise NotImplementedError
        item_info = {
            "context": self.preprocess(dialog_history, 'user'),
            "context_plain": context_plain,
            "response": self.preprocess(response_plain, 'system'),
            "response_plain": response_plain,
        }

        '''
        Add additional negative samples per training samples to make the selection harder, 
        we found that by adding this we can slightly improve the response selection performance
        '''
        if self.args["nb_neg_sample_rs"] != 0 and self.mode == "train":

            if self.args["sample_negative_by_kmeans"]:
                try:
                    cur_cluster = self.others["ToD_BERT_SYS_UTTR_KMEANS"][self.data["turn_sys"][index]]
                    candidates = self.others["KMEANS_to_SENTS"][cur_cluster]
                    nb_selected = min(self.args["nb_neg_sample_rs"], len(candidates))
                    try:
                        start_pos = random.randint(0, len(candidates) - nb_selected - 1)
                    except:
                        start_pos = 0
                    sampled_neg_resps = candidates[start_pos:start_pos + nb_selected]

                except:
                    start_pos = random.randint(0, len(self.resp_cand_trn) - self.args["nb_neg_sample_rs"] - 1)
                    sampled_neg_resps = self.resp_cand_trn[start_pos:start_pos + self.args["nb_neg_sample_rs"]]
            else:
                start_pos = random.randint(0, len(self.resp_cand_trn) - self.args["nb_neg_sample_rs"] - 1)
                sampled_neg_resps = self.resp_cand_trn[start_pos:start_pos + self.args["nb_neg_sample_rs"]]

            neg_resp_arr, neg_resp_idx_arr = [], []
            for neg_resp in sampled_neg_resps:
                neg_resp_plain = "{} ".format(self.sys_token) + neg_resp
                neg_resp_idx = self.preprocess(neg_resp_plain)[:self.max_sys_resp_len]
                neg_resp_idx_arr.append(neg_resp_idx)
                neg_resp_arr.append(neg_resp_plain)

            item_info["neg_resp_idx_arr"] = neg_resp_idx_arr
            item_info["neg_resp_arr"] = neg_resp_arr

        return item_info

    def __len__(self):
        return self.num_total_seqs

    def get_concat_context(self, dialog_history):
        dialog_history_str = ""
        for ui, uttr in enumerate(dialog_history):
            if ui % 2 == 0:
                dialog_history_str += "{} {} ".format(self.sys_token, uttr)
            else:
                dialog_history_str += "{} {} ".format(self.usr_token, uttr)
        dialog_history_str = dialog_history_str.strip()
        return dialog_history_str

    def collate_fn(self, data):
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        # data.sort(key=lambda x: len(x['context']), reverse=True)

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # augment negative samples
        if "neg_resp_idx_arr" in item_info.keys():
            neg_resp_idx_arr = []
            for arr in item_info['neg_resp_idx_arr']:
                neg_resp_idx_arr += arr

            # remove neg samples that are the same as one of the gold responses
            # print('item_info["response"]', item_info["response"])
            # print('neg_resp_idx_arr', neg_resp_idx_arr)

            for bi, arr in enumerate(item_info['neg_resp_arr']):
                for ri, neg_resp in enumerate(arr):
                    if neg_resp not in item_info["response_plain"]:
                        item_info["response"] += [item_info['neg_resp_idx_arr'][bi][ri]]

        return self.merge_inputs(item_info['context']), self.merge_inputs(item_info['response'])
