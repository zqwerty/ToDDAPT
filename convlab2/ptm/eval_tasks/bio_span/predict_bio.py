import copy

from tensorboardX import SummaryWriter
from tqdm import tqdm

from bert.model import BertForBIOTag
from dialogbert.model import DialogBertForBIOTag
from bert.processor import BertProcessor
from dialogbert.processor import DialogBertProcessor
from data_utils import Dataloader
import argparse
import json
import os
import random
import numpy as np
import torch
import math
from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', '-c',
                    help='path to config file')
parser.add_argument('--datasplit',
                    type=str,
                    help='')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_joint_acc(pred, golden):
    for domain in pred:
        if domain not in golden:
            return 0
        for slot in pred[domain]:
            if slot not in golden[domain] or pred[domain][slot] != golden[domain][slot]:
                return 0
    for domain in golden:
        if domain not in pred:
            return 0
        for slot in golden[domain]:
            if slot not in pred[domain] or pred[domain][slot] != golden[domain][slot]:
                return 0
    return 1


def update_state(state, update):
    for service, service_update in update.items():
        if service not in state:
            state[service] = copy.deepcopy(service_update)
        else:
            state[service].update(update[service])


def calculate_metrics(pred_goldens):
    joint_acc = 0
    slot_accs = {}
    for pred_golden in pred_goldens:
        joint_acc += pred_golden['joint_acc']
        pred = pred_golden['predict']
        golden = pred_golden['golden']
        for domain in pred:
            if domain not in slot_accs:
                slot_accs[domain] = {}
            for slot in pred[domain]:
                if slot not in slot_accs[domain]:
                    slot_accs[domain][slot] = {'total_pred': 0, 'total_label': 0, 'pred_right': 0, 'covered': 0}
                slot_accs[domain][slot]['total_pred'] += 1
                if (domain in golden) and (slot in golden[domain]) and (golden[domain][slot] == pred[domain][slot]):
                    slot_accs[domain][slot]['pred_right'] += 1
        for domain in golden:
            if domain not in slot_accs:
                slot_accs[domain] = {}
            for slot in golden[domain]:
                if slot not in slot_accs[domain]:
                    slot_accs[domain][slot] = {'total_pred': 0, 'total_label': 0, 'pred_right': 0, 'covered': 0}
                slot_accs[domain][slot]['total_label'] += 1
                if (domain in pred) and (slot in pred[domain]) and (golden[domain][slot] == pred[domain][slot]):
                    slot_accs[domain][slot]['covered'] += 1
    for domain in slot_accs:
        for slot in slot_accs[domain]:
            slot_accs[domain][slot]['accuracy'] = slot_accs[domain][slot]['pred_right'] / slot_accs[domain][slot]['total_pred'] if slot_accs[domain][slot]['total_pred'] > 0 else float(slot_accs[domain][slot]['pred_right'] == 0)
            slot_accs[domain][slot]['recall'] = slot_accs[domain][slot]['covered'] / slot_accs[domain][slot]['total_label'] if slot_accs[domain][slot]['total_label'] > 0  else float(slot_accs[domain][slot]['covered'] == 0)
    return slot_accs, joint_acc / len(pred_goldens)


def same_dict(dict_1, dict_2):
    # input: dict of {k: v}
    for key in dict_1:
        if key not in dict_2:
            return False
        # if dict_1[key] != dict_2[key]:
        elif dict_1[key] == dict_2[key]:
            continue
        else:
            return False

    for key in dict_2:
        if key not in dict_1:
            return False
        elif dict_1[key] == dict_2[key]:
            continue
        else:
            return False
    return True


def same_state(pred_bs, target_bs):
    same = True
    for _service_name in target_bs:
        if len(target_bs[_service_name]) == 0:
            continue
        else:
            if _service_name not in pred_bs:
                same = False
                break
            else:
                if not same_dict(target_bs[_service_name], pred_bs[_service_name]):
                    same = False
                    break
    if same:
        for _service_name in pred_bs:
            if len(pred_bs[_service_name]) == 0:
                continue
            else:
                if _service_name not in target_bs:
                    same = False
                    break
                else:
                    if not same_dict(target_bs[_service_name], pred_bs[_service_name]):
                        same = False
                        break
    return same


def calculate_accum_joint_acc(pred_dict):
    joint_acc = 0
    total = 0
    for dial_id in pred_dict:
        all_turn_ids = sorted(pred_dict[dial_id].keys())
        cur_pred_state = {}
        cur_golden_state = {}
        for _tid in all_turn_ids:
            total += 1
            _pred = pred_dict[dial_id][_tid]['predict']
            _golden = pred_dict[dial_id][_tid]['golden']
            update_state(cur_pred_state, _pred)
            update_state(cur_golden_state, _golden)
            if same_state(cur_pred_state, cur_golden_state):
                joint_acc += 1
    return joint_acc / total


def to_device(value, device):
    if not isinstance(value, list):
        return value.to(device)
    else:
        return [v.to(device) for v in value]


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']
    DEVICE = config['DEVICE']
    dataset = config['dataset']

    print(args)

    if 'multiwoz' in data_dir:
        print('-'*20 + 'dataset:multiwoz' + '-'*20)

    if config['basemodel'] == 'bert':
        labels_map = json.load(open(os.path.join(data_dir, dataset + '_bert' + '_' +'labels_map.json')))
        slot_ontology = json.load(open(os.path.join(data_dir, dataset + '_bert' + '_' +'slot_ontology.json')))
        train_data = json.load(open(os.path.join(data_dir, dataset + '_bert_bio_train.json')))
        val_data = json.load(open(os.path.join(data_dir, dataset + '_bert_bio_val.json')))
        test_data = json.load(open(os.path.join(data_dir, dataset + '_bert_bio_test.json')))
        processor = BertProcessor(pretrained_weights=config['model']['pretrained_weights'], labels_map=labels_map)
    elif config['basemodel'] == 'dialogbert':
        labels_map = json.load(open(os.path.join(data_dir, dataset + '_dialogbert' + '_' + 'labels_map.json')))
        slot_ontology = json.load(open(os.path.join(data_dir, dataset + '_dialogbert' + '_' +'slot_ontology.json')))
        train_data = json.load(open(os.path.join(data_dir, dataset + '_dialogbert_bio_train.json')))
        val_data = json.load(open(os.path.join(data_dir, dataset + '_dialogbert_bio_val.json')))
        test_data = json.load(open(os.path.join(data_dir, dataset + '_dialogbert_bio_test.json')))
        processor = DialogBertProcessor(config['model']['pretrained_weights'], labels_map=labels_map)
    else:
        raise ValueError('basemodel in config must be in `bert, dialogbert`')

    dataloader = Dataloader(processor, use_weighted_tag_loss=True)
    print('tag num:', len(labels_map['id_to_label']))
    for data_key in ['train', 'val', 'test']:
        dataloader.load_data(eval(data_key + '_data'), data_key)
        if 'train_ratio' in config and config['train_ratio'] < 1 and data_key == 'train':
            set_seed(10086)
            dataloader.data[data_key] = random.sample(dataloader.data[data_key],
                                                      math.ceil(len(dataloader.data[data_key]) * config['train_ratio']))
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if config['basemodel'] == 'bert':
        # if config['model']['use_tag_weight']:
        #     model = BertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=dataloader.tag_weight,
        #                           slot_ontology=slot_ontology)
        # else:
        model = BertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=None,
                                  slot_ontology=slot_ontology)
        model.to(DEVICE)
    elif config['basemodel'] == 'dialogbert':
        # if config['model']['use_tag_weight']:
        #     model = DialogBertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=dataloader.tag_weight,
        #                                 slot_ontology=slot_ontology)
        # else:
        model = DialogBertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=None,
                                    slot_ontology=slot_ontology)
        model.to(DEVICE)
    else:
        raise ValueError('config[\'basemodel\'] illegal value')

    cate_slot_list = slot_ontology['categorical_slot_list']
    noncate_slot_list = slot_ontology['non-categorical_slot_list']
    cate_slot_ontology = slot_ontology['categorical_slot_ontology']

    if 'resume_checkpoint' in config['model']:
        print('loading checkpoint from {}'.format(config['model']['resume_checkpoint']))
        model.load_state_dict(torch.load(config['model']['resume_checkpoint']))

    batch_size = config['model']['predict_batch_size']

    predict_golden = []
    predict_dict = {}
    model.eval()
    for pad_batch_info, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key=args.datasplit):
        pad_batch, dial_ids, turn_ids, golden_state_update = pad_batch_info
        pad_batch = {k:to_device(v, DEVICE) for k, v in pad_batch.items()}

        with torch.no_grad():
            _val_tag_logits, _val_tag_loss, _val_slot_logits, _val_cate_loss, _val_noncate_loss = model.forward(
                **pad_batch)

        _true_labels = pad_batch['tag_seq_tensor'].detach().cpu().tolist()
        _pred_labels = torch.argmax(_val_tag_logits, -1).detach().cpu().tolist()
        _input_ids = pad_batch['word_seq_tensor'].detach().cpu().tolist()

        _val_slot_labels = [torch.argmax(_logits, -1).detach().cpu().tolist() for _logits in _val_slot_logits]

        for j in range(real_batch_size):
            predict = processor.recover_dialogue_act(_pred_labels[j], _input_ids[j])
            true_label = processor.recover_dialogue_act(_true_labels[j], _input_ids[j])

            for slot_type in golden_state_update[j]:
                for dsv in golden_state_update[j][slot_type]:
                    d = dsv['domain']
                    s = dsv['slot']
                    v = dsv['fixed_value'] if ('fixed_value' in dsv and dsv['fixed_value'] != 'not found') else dsv['value']
                    if d not in true_label:
                        true_label[d] = {}
                    if not s in true_label[d]:
                        true_label[d][s] = v

            for i in range(model.num_cate_slot):
                _pred_slot_label = _val_slot_labels[i][j]
                __domain, __slot = cate_slot_list[i]
                __value = cate_slot_ontology[__domain][__slot][_pred_slot_label]
                if __value == 'none':
                    continue
                if __domain not in predict:
                    predict[__domain] = {}
                predict[__domain][__slot] = __value

            for i in range(model.num_cate_slot, model.num_cate_slot + model.num_noncate_slot):
                _pred_slot_label = _val_slot_labels[i][j]
                __domain, __slot = noncate_slot_list[i - model.num_cate_slot]

                if _pred_slot_label == 0:
                    continue
                if __domain not in predict:
                    predict[__domain] = {}
                predict[__domain][__slot] = 'dontcare'

            predict_golden.append({
                'dial_id': dial_ids[j],
                'turn_id': turn_ids[j],
                'predict': predict,
                'golden': true_label,
                'joint_acc': calculate_joint_acc(predict, true_label),
                'context': processor.tokenizer.decode(_input_ids[j], skip_special_tokens=True)
            })

            if dial_ids[j] not in predict_dict:
                predict_dict[dial_ids[j]] = {}
            assert int(turn_ids[j]) not in predict_dict[dial_ids[j]]
            predict_dict[dial_ids[j]][int(turn_ids[j])] = {
                'predict': predict,
                'golden': true_label,
            }

    slot_acc, joint_acc = calculate_metrics(predict_golden)
    accum_joint_acc = calculate_accum_joint_acc(predict_dict)
    predict_dir = os.path.join(output_dir, 'predict_result')
    os.makedirs(predict_dir, exist_ok=True)
    json.dump(slot_acc, open(os.path.join(predict_dir, 'slot_accuracy.json'), 'w'), indent=4)
    json.dump(predict_golden, open(os.path.join(predict_dir, 'predictions.json'), 'w'), indent=4)
    print('dataset {} joint acc: {}'.format(args.datasplit, joint_acc))
    print('dataset {} accumulate joint acc: {}'.format(args.datasplit, accum_joint_acc))
