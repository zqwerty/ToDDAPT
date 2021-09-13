import logging
import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
import random
import numpy as np
import sys
sys.path.append('../../../../')
from convlab2.ptm.pretraining.model import (
    DialogBertConfig,
    DialogBertTokenizer,
    DialogBertForPretraining,
)


logger = logging.getLogger(__name__)


class BIODataset(Dataset):
    def __init__(self, data, tokenizer: DialogBertTokenizer, max_length):
        """prepare input and label here

            self.data: A Dictionary of shape::
                {
                    input_ids: list[int], reverse order
                    turn_ids: list[int]
                    role_ids: list[int]
                    position_ids: list[int]

                    bio_tag_ids: list[int]
                    bio_mask: list[int]

                    length: int
                    utt_num: int
                }

            self.data_bucket_ids: utterance numbers of dialogues
        """
        self.data = []
        # data = [{"num_utt": int, "dialogue": list of token lists, "bio_tag": list of tag lists
        # "spans": list of spans:{domain, slot, value, utt_idx, start, end}}]
        bio_tag2id = {'B': 1, 'I': 2, 'O': 0, 'X': -100}
        for d in data:
            # print(d['dialogue'])
            # print(d)
            encoded_inputs = tokenizer.prepare_input_seq(d['dialogue'], last_role='user',
                                                         max_length=max_length, return_lengths=True)
            length = encoded_inputs['length']
            bio_tags = ['X']
            sen_idx = 1
            for tokens, sen_bio in zip(d['dialogue'][::-1], d['bio_tag'][::-1]):
                tags2add = ['X'] + sen_bio + ['X']
                if sen_idx + len(tokens) + 2 > length:
                    print('exceed max length {}'.format(length))
                    break
                bio_tags += tags2add
                sen_idx += len(tokens) + 2
            # print(d['dialogue'])
            bio_tag_ids = [bio_tag2id[tag] for tag in bio_tags]
            encoded_inputs['dialogue'] = d['dialogue']
            encoded_inputs['bio_tag'] = d['bio_tag']
            encoded_inputs['dataset'] = d['dataset']
            encoded_inputs['bio_tag_ids'] = bio_tag_ids
            assert length == len(encoded_inputs['input_ids'])
            assert length == len(encoded_inputs['turn_ids'])
            assert length == len(encoded_inputs['role_ids'])
            assert length == len(encoded_inputs['position_ids'])
            assert 0 < length <= max_length
            self.data.append(encoded_inputs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    batch_size = len(batch_data)
    max_seq_len = max([x['length'] for x in batch_data])
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    turn_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    role_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    bio_tag_ids = torch.ones((batch_size, max_seq_len), dtype=torch.long) * -100

    for i in range(batch_size):
        sen_len = batch_data[i]['length']
        attention_mask[i, :sen_len] = 1
        input_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['input_ids'])
        turn_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['turn_ids'])
        role_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['role_ids'])
        position_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['position_ids'])
        bio_tag_ids[i, :sen_len] = torch.LongTensor(batch_data[i]['bio_tag_ids'])

    return {"attention_mask": attention_mask,
            "input_ids": input_ids, "turn_ids": turn_ids, "role_ids": role_ids, "position_ids": position_ids,
            "bio_tag_ids": bio_tag_ids,
            # "dialogue": [x['dialogue'] for x in batch_data],
            # "bio_tag": [x['bio_tag'] for x in batch_data],
            }


def move_input_to_device(inputs, device):
    if isinstance(inputs, list):
        return [move_input_to_device(v, device) for v in inputs]
    elif isinstance(inputs, dict):
        return {k: move_input_to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    else:
        return inputs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_bio_tag(bio_tags):
    new_tags = []
    i = 0
    while i < len(bio_tags):
        tag = bio_tags[i]
        i += 1
        if tag == 'B':
            new_tags.append(tag)
            j = i
            while j < len(bio_tags):
                if bio_tags[j] in ['I', 'X']:
                    new_tags.append(bio_tags[j])
                    i += 1
                    j += 1
                else:
                    break
        elif tag == 'I':
            new_tags.append('O')
        else:
            new_tags.append(tag)
    return new_tags


def merge_bio_tag(ori_tags, pred_tags):
    merged_tags = []
    for ori_tag, pred_tag in zip(ori_tags, pred_tags):
        if ori_tag == 'O':
            merged_tags.append(pred_tag)
        else:
            merged_tags.append(ori_tag)
    return merged_tags


def merge_bio_pred(dialogue, ori_bio_tag, pred_bio_tag, input_ids, tokenizer):
    new_bio_tag, merged_bio_tag = [], []
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for token, tag in zip(input_tokens[1:], pred_bio_tag[1:]):
        # ignore CLS
        if token == '[SEP]':
            continue
        if token in ['[USR]', '[SYS]']:
            # reverse order input
            new_bio_tag.insert(0, [])
            continue
        if token == '[PAD]':
            break
        new_bio_tag[0].append(tag)

    # if len(new_bio_tag) != len(ori_bio_tag):
    #     print('exceed max length {} > {}'.format(len(ori_bio_tag), len(new_bio_tag)))
    for sen, new_sen_tag, ori_sen_tag in zip(dialogue[-len(new_bio_tag):], new_bio_tag, ori_bio_tag[-len(new_bio_tag):]):
        valid_tag = normalize_bio_tag(new_sen_tag)
        merged_bio_tag.append(valid_tag)
        assert len(new_sen_tag) == len(ori_sen_tag)
        # merged_tag = merge_bio_tag(ori_sen_tag, new_sen_tag)
        # merged_tag = normalize_bio_tag(merged_tag)
        # merged_bio_tag.append(merged_tag)

        # if ori_sen_tag != merged_tag:
        #     print(sen)
        #     print(ori_sen_tag)
        #     print(new_sen_tag)
        #     print(merged_tag)
        #     print()

    if len(merged_bio_tag) < len(ori_bio_tag):
        # cut tag seq, need to pad to ori len, copy from ori
        # notice: when use pseudo bio tag, should set the max length to the one that this procedure use
        merged_bio_tag = ori_bio_tag[:len(ori_bio_tag)-len(merged_bio_tag)] + merged_bio_tag
    assert len(merged_bio_tag) == len(ori_bio_tag)
    assert list(map(len, merged_bio_tag)) == list(map(len, ori_bio_tag))
    return merged_bio_tag


if __name__ == "__main__":
    dataset_names = [
        'camrest',
        'woz',

        'kvret',
        'dstc2',
        'mdc',
        'metalwoz',

        'frames',
        'm2m',
        'schema',
        'taskmaster',

        'multiwoz25',
    ]
    set_seed(42)
    data_dir = 'prefix_dialog'
    model_path = '/home/data/zhuqi/pre-trained-models/dialogbert/bio_mlm/mlm_masked_bio_12k_batch64_lr1e-4_block256_1031'
    tokenizer = DialogBertTokenizer.from_pretrained(model_path)
    config = DialogBertConfig.from_pretrained(model_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--bio', type=bool, default=True)
    parser.add_argument('--ssl', type=bool, default=False)
    parser.add_argument('--schema_linking', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    model = DialogBertForPretraining.from_pretrained(
        model_path,
        config=config,
        training_args=None,
        model_args=None,
        data_args=args,
        metadataloader=None
    )
    model.to(args.device)
    model.eval()
    max_length = 512
    batch_size = 128
    tag_map = {1: 'B', 2: 'I', 0: 'O'}
    for dataset in dataset_names:
        print('load {} dataset:'.format(dataset))
        for phase in ['dev', 'train']:
            data = json.load(open(os.path.join(data_dir, dataset + '_data_{}.json'.format(phase))))
            bio_dataset = BIODataset(data, tokenizer, max_length)
            sampler = SequentialSampler(bio_dataset)
            bio_dataloader = DataLoader(bio_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
            sample_idx = 0
            for batch_data in tqdm(bio_dataloader, desc="Evaluating {}".format(dataset)):
                # print(batch_data)
                # print()
                # dialogue = batch_data.pop('dialogue')
                # bio_tag = batch_data.pop('bio_tag')
                batch_data = move_input_to_device(batch_data, args.device)
                with torch.no_grad():
                    outputs = model(**batch_data, task='bio', dataset=dataset, evaluate=True)
                    bio_logits = outputs["bio_logits"]
                for i in range(len(batch_data['input_ids'])):
                    _, prediction_tags = torch.max(bio_logits[i], dim=-1)
                    prediction_tags = [tag_map[tag.item()] if tag.item() in tag_map else 'X' for tag in prediction_tags]
                    # print(dialogue)
                    # print(bio_tag)
                    # print(tokenizer.convert_ids_to_tokens(batch_data['input_ids'][i]))
                    # print(prediction_tags)
                    # assert dialogue[i] == data[sample_idx]['dialogue']
                    # assert bio_tag[i] == data[sample_idx]['bio_tag']
                    data[sample_idx]['pseudo_bio_tag'] = merge_bio_pred(data[sample_idx]['dialogue'],
                                                                        data[sample_idx]['bio_tag'],
                                                                        prediction_tags,
                                                                        batch_data['input_ids'][i],
                                                                        tokenizer)
                    sample_idx += 1
            json.dump(data, open(os.path.join(data_dir, dataset + '_data_{}_pseudo_bio.json'.format(phase)), 'w'), indent=2)


