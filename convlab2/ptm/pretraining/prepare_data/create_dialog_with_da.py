import random
import os
import sys
import json
import logging
from itertools import chain
from tqdm import tqdm
from collections import defaultdict
from convlab2.util.file_util import read_zipped_json, write_zipped_json
from convlab2.ptm.pretraining.model import DialogBertTokenizer

def parent_dir(path, time=1):
    for _ in range(time):
        path = os.path.dirname(path)
    return path

self_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = parent_dir(self_dir, 4)

sys.path.append(root_dir)
print(sys.path[-1])


random.seed(1234)
dev_ratio = 0.01  # ratio of data used to evaluate

# datasets used in pretraining
# dev/test set of multiwoz and schema are excluded automatically
dataset_names = [
    'schema',
]

data_dir = os.path.join(root_dir, 'data_ptm')
processed_data_dir = os.path.join(self_dir, 'dialog_with_da')

os.makedirs(processed_data_dir, exist_ok=True)

tokenizer = DialogBertTokenizer.from_pretrained('/home/guyuxian/pretrain-models/bert-base-uncased')

special_token_dict = {'additional_special_tokens': ['[USR]', '[SYS]', '[INTENT]', '[DOMAIN]', '[SLOT]', '[VALUE]',
                                                    '[STATE]', '[DIALOG_ACT]',
                                                    '[NEXT_SENTENCE]']}
tokenizer.add_special_tokens(special_token_dict)

usr_id = tokenizer.convert_tokens_to_ids("[USR]")
sys_id = tokenizer.convert_tokens_to_ids("[SYS]")
sep_id = tokenizer.convert_tokens_to_ids("[SEP]")

for _ds in dataset_names:
    da_utt_dict = defaultdict(list)
    data = {
        "train": [],
        "val": [],
        "test": []
    }

    logging.info('preprocessing {} dataset'.format(_ds))
    _dataset_dir = os.path.join(data_dir, _ds)
    _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
    if not os.path.exists(_data_zipfile):
        raise FileNotFoundError('Data Not Found!', _data_zipfile)
    else:
        _dialogs = read_zipped_json(_data_zipfile, 'data.json')
        _ontology = json.load(open(os.path.join(_dataset_dir, 'ontology.json')))

    for dial in tqdm(_dialogs, desc="Processing dataset {}".format(_ds)):
        turns = dial["turns"]
        split = dial["data_split"]
        data[split].append({
            "utt_ids": [],
            "da_list": [],
            "num_utt": 0
        })
        for utt in turns:
            utt_da = utt["dialogue_act"]
            da = [utt["speaker"]]
            da_spans = {}
            for x in chain(utt_da["binary"], utt_da["categorical"], utt_da["non-categorical"]):
                tmp_da = str((x["intent"], x["domain"], x["slot"]))
                da.append(tmp_da)
                if "start" in x and "end" in x:
                    da_spans[tmp_da] = (x["start"], x["end"])

            da_spans = sorted(da_spans.items(), key=lambda x: x[1][0])
            da_token_spans = {}
            start = 0
            if utt["speaker"] == "user":
                token_ids = [usr_id]
            else:
                token_ids = [sys_id]

            for span in da_spans:
                da_token_spans[span[0]] = {}
                token_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt["utterance"][start:span[1][0]])))
                da_token_spans[span[0]]["start"] = len(token_ids)
                token_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt["utterance"][span[1][0]:span[1][1]])))
                da_token_spans[span[0]]["end"] = len(token_ids)
                start = span[1][1]
            token_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt["utterance"][start:])))

            da = "-".join(sorted(da))

            token_ids += [sep_id]
            
            da_utt_dict[da].append({
                "token_ids": token_ids,
                "da_spans": da_token_spans
            })

            data[split][-1]["utt_ids"].append({
                "token_ids": token_ids,
                "da_spans": da_token_spans
            })
            data[split][-1]["da_list"].append(da)
            data[split][-1]["num_utt"] += 1

    for split in data:
        with open(os.path.join(processed_data_dir, "{}_data_{}.json".format(_ds, "dev" if split == "val" else split)), "w") as f:
            json.dump(data[split], f, indent=4)
    
    with open(os.path.join(processed_data_dir, "{}_utt_pool.json".format(_ds)), "w") as f:
        json.dump(da_utt_dict, f, indent=4)
