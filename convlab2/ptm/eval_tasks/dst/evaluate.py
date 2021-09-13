"""
Evaluate NLU models on specified dataset
Usage: python evaluate.py [MultiWOZ|CrossWOZ] [TRADE|mdbt|sumbt|rule]
"""
import random
import numpy
import torch
import sys
from tqdm import tqdm
import copy
import jieba
import json
import os
from torch.utils.data import DataLoader, SequentialSampler

from convlab2.ptm.eval_tasks.dst.data_utils.MultiwozDataset import MultiwozDataset
from convlab2.ptm.eval_tasks.dst.model import DstModel, parser, get_transformer_settings
from convlab2.ptm import DATA_PTM_PATH


def evaluate_metrics(all_labels, all_preds, slot_num):
    slot_acc, joint_acc, slot_acc_total, joint_acc_total = 0, 0, 0, 0
    for labels, preds in zip(all_labels, all_preds):
        joint = 0
        for slot_id in range(slot_num):
            # NOTE: not exact, remove empty chars?
            pred = preds[slot_id]
            label = labels[slot_id]
            if pred == label:
                slot_acc += 1
                joint += 1
            slot_acc_total += 1

        if joint == slot_num:
            joint_acc += 1

        joint_acc_total += 1

    joint_acc = joint_acc / joint_acc_total
    slot_acc = slot_acc / slot_acc_total

    return joint_acc, slot_acc


if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    args = parser.parse_args()
    data_dir = os.path.join(DATA_PTM_PATH, args.dataset)
    ontology = json.load(open(os.path.join(data_dir, 'ontology.json')))
    tokenizer_cls, model_cls, config_cls = get_transformer_settings(args.transformer)
    tokenizer = tokenizer_cls.from_pretrained(args.transformer_path)

    dataset_name = "MultiWOZ"

    if dataset_name == 'MultiWOZ':
        model = DstModel(args, ontology).to(args.device)
        slot_num = len(model.slot2id)
        model.eval()
        ## load data
        test_set = MultiwozDataset(args, tokenizer, data_dir, 'test', evaluate=True)
        eval_sampler = SequentialSampler(test_set)
        # only one dialog at a time
        eval_loader = DataLoader(test_set, batch_size=1, sampler=eval_sampler, collate_fn=test_set.collate_eval)

        all_preds = []
        all_labels = []
        for inputs, candidates, labels in tqdm(eval_loader, "Evaluating"):
            dial_preds = []
            dial_labels = []
            inputs, candidates, labels = inputs[0], candidates[0], labels[0]
            # send one utt
            for utt_inputs, utt_cands, utt_labels in zip(inputs, candidates, labels):
                # utt_inputs: slot_num * token_num
                utt_pred, _ = model.update(utt_inputs, utt_cands)
                dial_preds.append(utt_pred)
                dial_labels.append(utt_labels)
            all_preds.append(dial_preds)
            all_labels.append(dial_labels)

        joint_acc, slot_acc = evaluate_metrics(all_labels, all_preds, slot_num)
        evaluation_metrics = {"Joint Acc": joint_acc, "Slot Acc": slot_acc}
        print(evaluation_metrics)
