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
parser.add_argument('-r',
                    type=int,
                    help='random seed, default set by config file')
parser.add_argument('-train_ratio',
                    type=float,
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
    if args.r:
        random_seed = args.r
        output_dir += '_seed_{}'.format(args.r)
        log_dir += '_seed_{}'.format(args.r)
    else:
        random_seed = config['seed']
    if args.train_ratio:
        config['train_ratio'] = args.train_ratio
        output_dir += '_{}data'.format(args.train_ratio)
        log_dir += '_{}data'.format(args.train_ratio)

    print(args)

    if os.path.isdir(output_dir):
        print('output directory {} exists!'.format(output_dir))
        exit()

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

    writer = SummaryWriter(log_dir)

    set_seed(random_seed)

    if config['basemodel'] == 'bert':
        if config['model']['use_tag_weight']:
            model = BertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=dataloader.tag_weight, slot_ontology=slot_ontology)
        else:
            model = BertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=None, slot_ontology=slot_ontology)
        model.to(DEVICE)
    elif config['basemodel'] == 'dialogbert':
        if config['model']['use_tag_weight']:
            model = DialogBertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=dataloader.tag_weight, slot_ontology=slot_ontology)
        else:
            model = DialogBertForBIOTag(config['model'], DEVICE, dataloader.tag_dim, tag_weight=None, slot_ontology=slot_ontology)
        model.to(DEVICE)
    else:
        raise ValueError('config[\'basemodel\'] illegal value')

    max_step = config['model']['max_step']
    warmup_steps = config['model']['warmup_steps']
    check_step = config['model']['check_step']

    if 'train_ratio' in config:
        scale = np.sqrt(config['train_ratio'])
        print('train_ratio', config['train_ratio'])
        print('scale', scale)
        max_step = math.ceil(max_step*scale)
        warmup_steps = math.ceil(warmup_steps*scale)
        check_step = math.ceil(check_step*scale)
        print('max_step', max_step)
        print('warmup_steps', warmup_steps)
        print('check_step', check_step)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': config['model']['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['model']['learning_rate'],
                      eps=config['model']['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=max_step)

    for name, param in model.named_parameters():
        print(name, param.shape, param.device, param.requires_grad)

    batch_size = config['model']['batch_size']
    model.zero_grad()
    train_loss = 0
    train_tag_loss = 0
    train_cate_loss = 0
    train_noncate_loss = 0
    best_val_joint_acc = 0.
    best_val_loss = np.inf

    cate_slot_list = slot_ontology['categorical_slot_list']
    noncate_slot_list = slot_ontology['non-categorical_slot_list']
    cate_slot_ontology = slot_ontology['categorical_slot_ontology']

    writer.add_text('config', json.dumps(config))

    for step in tqdm(range(1, max_step + 1)):
        model.train()
        batched_data_info = dataloader.get_train_batch(batch_size)
        batched_data, _, _, _ = batched_data_info
        batched_data = {k:to_device(v, DEVICE) for k, v in batched_data.items()}
        tag_logits, tag_loss, all_slot_logits, cate_loss, noncate_loss = model.forward(**batched_data)
        train_loss += tag_loss.item() + cate_loss.item() + noncate_loss.item()
        train_tag_loss += tag_loss.item()
        train_cate_loss += cate_loss.item()
        train_noncate_loss += noncate_loss.item()
        (tag_loss + cate_loss + noncate_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        if step % check_step == 0 or step == max_step:
            train_loss = train_loss / check_step
            train_cate_loss /= check_step
            train_noncate_loss /= check_step
            train_tag_loss /= check_step

            predict_golden = []

            val_loss = 0
            val_tag_loss = 0
            val_cate_loss = 0
            val_noncate_loss = 0
            model.eval()

            # check
            # flag = 0
            for pad_batch_info, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key='val'):
                # if flag > 9:
                #     print(json.dumps(predict_golden, indent=4))
                #     exit()
                # flag += 1

                pad_batch, dial_ids, turn_ids, golden_state_update = pad_batch_info
                pad_batch = {k:to_device(v, DEVICE) for k, v in pad_batch.items()}

                with torch.no_grad():
                    _val_tag_logits, _val_tag_loss, _val_slot_logits, _val_cate_loss, _val_noncate_loss = model.forward(**pad_batch)
                val_loss += (_val_tag_loss + _val_cate_loss + _val_noncate_loss).item() * real_batch_size
                val_tag_loss += _val_tag_loss.item()
                val_cate_loss += _val_cate_loss.item()
                val_noncate_loss += _val_noncate_loss.item()

                _true_labels = pad_batch['tag_seq_tensor'].detach().cpu().tolist()
                _pred_labels = torch.argmax(_val_tag_logits, -1).detach().cpu().tolist()
                _input_ids = pad_batch['word_seq_tensor'].detach().cpu().tolist()

                _val_slot_labels = [torch.argmax(_logits, -1).detach().cpu().tolist() for _logits in _val_slot_logits]

                for j in range(real_batch_size):
                    # check
                    predict = processor.recover_dialogue_act(_pred_labels[j], _input_ids[j])
                    # predict = processor.recover_dialogue_act(_true_labels[j], _input_ids[j])
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
                        'predict': predict,
                        'golden': true_label,
                        'joint_acc': calculate_joint_acc(predict, true_label)
                    })

            for j in range(10):
                writer.add_text('val_sample_{}'.format(j),
                                json.dumps(predict_golden[j], indent=2, ensure_ascii=False),
                                global_step=step)

            total = len(dataloader.data['val'])
            val_loss /= total
            val_tag_loss /= total
            val_cate_loss /= total
            val_noncate_loss /= total

            writer.add_scalar('loss/train', train_loss, global_step=step)
            writer.add_scalar('loss/val', val_loss, global_step=step)
            writer.add_scalar('tag_loss/train', train_tag_loss, global_step=step)
            writer.add_scalar('tag_loss/val', val_tag_loss, global_step=step)
            writer.add_scalar('cate_loss/train', train_cate_loss, global_step=step)
            writer.add_scalar('cate_loss/val', val_cate_loss, global_step=step)
            writer.add_scalar('noncate_loss/train', train_noncate_loss, global_step=step)
            writer.add_scalar('noncate_loss/val', val_noncate_loss, global_step=step)

            slot_acc, joint_acc = calculate_metrics(predict_golden)
            writer.add_scalar('val_metrics/joint_acc', joint_acc, global_step=step)


            if joint_acc > best_val_joint_acc:
                best_val_joint_acc = joint_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                print('best joint acc %.4f' % best_val_joint_acc)
                print('save on', output_dir)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            #     print('best val loss %f' % val_loss)
            #     print('save on', output_dir)

            train_loss = 0
            train_tag_loss = 0
            train_cate_loss = 0
            train_noncate_loss = 0

    writer.add_text('best val joint acc', '%.2f' % (100 * best_val_joint_acc))
    writer.close()