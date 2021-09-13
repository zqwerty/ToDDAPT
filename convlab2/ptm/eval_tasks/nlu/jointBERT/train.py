import argparse
import json
import os
import random
import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from convlab2.ptm.eval_tasks.nlu.jointBERT.dataloader import Dataloader
from convlab2.ptm.eval_tasks.nlu.jointBERT.bert.model import BertForNLU
from convlab2.ptm.eval_tasks.nlu.jointBERT.bert.processor import BertProcessor
from convlab2.ptm.eval_tasks.nlu.jointBERT.todbert.model import ToDBertForNLU
from convlab2.ptm.eval_tasks.nlu.jointBERT.todbert.processor import ToDBertProcessor
from convlab2.ptm.eval_tasks.nlu.jointBERT.dialogbert.model import DialogBertForNLU
from convlab2.ptm.eval_tasks.nlu.jointBERT.dialogbert.processor import DialogBertProcessor
from convlab2.ptm.eval_tasks.nlu.jointBERT.multiwoz.postprocess import is_slot_da, calculateF1, calculateAcc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="", allow_abbrev=False)
parser.add_argument('--seeds',
                    nargs="+",
                    type=int,
                    default=[42],
                    help='random seed')
parser.add_argument('--train_ratios',
                    nargs="*",
                    type=float,
                    default=[1.0],
                    help='')
parser.add_argument('--evaluate_during_training',
                    action='store_true',
                    help='')
parser.add_argument('--do_train',
                    action='store_true',
                    help='')
parser.add_argument('--do_eval',
                    action='store_true',
                    help='')
parser.add_argument('--probing',
                    action='store_true',
                    help='')
parser.add_argument('--data_dir',
                    type=str,
                    default="multiwoz/multiwoz25/all_data",
                    help='')
parser.add_argument('--pretrained_weights',
                    type=str,
                    default="/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_120k_0831_bert",
                    help='')
parser.add_argument('--basemodel',
                    type=str,
                    default="dialogbert",
                    help='')
parser.add_argument('--output_dir',
                    type=str,
                    default="/home/data/zhuqi/pre-trained-models/dialogbert/eval_nlu/test",
                    help='')
parser.add_argument('--device',
                    type=str,
                    default="cuda",
                    help='')
parser.add_argument('--cut_sen_len',
                    type=int,
                    default=256,
                    help='')
parser.add_argument('--context_size',
                    type=int,
                    default=3,
                    help='')
parser.add_argument('--check_step',
                    type=int,
                    default=1000,
                    help='')
parser.add_argument('--max_step',
                    type=int,
                    default=10000,
                    help='')
parser.add_argument('--batch_size',
                    type=int,
                    default=40,
                    help='')
parser.add_argument('--eval_batch_size',
                    type=int,
                    default=200,
                    help='')
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-4,
                    help='')
parser.add_argument('--adam_epsilon',
                    type=float,
                    default=1e-8,
                    help='')
parser.add_argument('--warmup_steps',
                    type=int,
                    default=0,
                    help='')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='')
parser.add_argument('--dropout',
                    type=float,
                    default=0.1,
                    help='')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    data_dir = args.data_dir
    DEVICE = args.device
    random_seeds = args.seeds
    train_ratios = args.train_ratios

    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))

    print('-' * 20 + 'data_path:{}'.format(data_dir) + '-' * 20)
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))

    if args.basemodel == 'bert':
        processor = BertProcessor(args.pretrained_weights, intent_dim=len(intent_vocab))
    elif args.basemodel == 'dialogbert':
        processor = DialogBertProcessor(args.pretrained_weights, intent_dim=len(intent_vocab))
    elif args.basemodel == 'tod-bert':
        processor = ToDBertProcessor(args.pretrained_weights, intent_dim=len(intent_vocab))
    else:
        raise Exception("basemodel not found")
    dataloader = Dataloader(intent_vocab, tag_vocab, processor)

    if args.evaluate_during_training:
        print('load val data')
        data_key = 'val'
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))),
                             data_key,
                             cut_sen_len=args.cut_sen_len,
                             context_size=args.context_size)
    if args.do_eval:
        print('load test data')
        data_key = 'test'
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))),
                             data_key,
                             cut_sen_len=args.cut_sen_len,
                             context_size=args.context_size)
        eval_res = {}

    if args.do_train:
        print('load train data')
        data_key = 'train'
        train_data = json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key))))

    for train_ratio in train_ratios:
        if args.do_train:
            data_key = 'train'
            dataloader.load_data(train_data[:math.ceil(len(train_data) * train_ratio)],
                                 data_key,
                                 cut_sen_len=args.cut_sen_len,
                                 context_size=args.context_size)

        for random_seed in random_seeds:
            print('-' * 50)
            print('train ratio:', train_ratio)
            if args.do_train:
                print('train samples:', len(dataloader.data['train']))
            print('seed:', random_seed)
            print('-' * 50)
            output_dir = os.path.join(args.output_dir, '{}data'.format(train_ratio))
            output_dir = os.path.join(output_dir, 'seed_{}'.format(random_seed))
            log_dir = os.path.join(output_dir, 'log')

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            writer = SummaryWriter(log_dir)
            set_seed(random_seed)

            if args.basemodel == 'bert':
                model = BertForNLU(args, DEVICE, dataloader.tag_dim, dataloader.intent_dim, dataloader.intent_weight)
            elif args.basemodel == 'dialogbert':
                model = DialogBertForNLU(args, DEVICE, dataloader.tag_dim, dataloader.intent_dim, dataloader.intent_weight)
            elif args.basemodel == 'tod-bert':
                model = ToDBertForNLU(args, DEVICE, dataloader.tag_dim, dataloader.intent_dim, dataloader.intent_weight)
            else:
                raise Exception("basemodel not found")
            model.to(DEVICE)

            if args.do_train:
                max_step = args.max_step
                warmup_steps = args.warmup_steps
                check_step = args.check_step
                scale = np.sqrt(train_ratio)
                print('scale', scale)
                max_step = math.ceil(max_step * scale)
                warmup_steps = math.ceil(warmup_steps * scale)
                check_step = math.ceil(check_step * scale)
                print('max_step', max_step)
                print('warmup_steps', warmup_steps)
                print('check_step', check_step)

                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if
                                not any(nd in n for nd in no_decay) and p.requires_grad],
                     'weight_decay': args.weight_decay},
                    {'params': [p for n, p in model.named_parameters() if
                                any(nd in n for nd in no_decay) and p.requires_grad],
                     'weight_decay': 0.0}
                ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                                  eps=args.adam_epsilon)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=max_step)

                batch_size = args.batch_size
                model.zero_grad()
                train_slot_loss, train_intent_loss = 0, 0
                train_steps = 0
                best_val_f1 = 0.
                best_val_loss = np.inf

                config = vars(args)
                config['train_ratio'] = train_ratio
                config['seed'] = random_seed
                writer.add_text('config', json.dumps(config))

                for step in tqdm(range(1, max_step + 1)):
                    model.train()
                    batched_data = dataloader.get_train_batch(batch_size)
                    batched_data = tuple(t.to(DEVICE) for t in batched_data)
                    _, _, slot_loss, intent_loss = model.forward(*batched_data)
                    train_steps += 1
                    train_slot_loss += slot_loss.item()
                    train_intent_loss += intent_loss.item()
                    loss = slot_loss + intent_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    if step % check_step == 0 or step == max_step:
                        train_slot_loss = train_slot_loss / train_steps
                        train_intent_loss = train_intent_loss / train_steps
                        writer.add_scalar('slot_loss/train', train_slot_loss, global_step=step)
                        writer.add_scalar('intent_loss/train', train_intent_loss, global_step=step)

                        if args.evaluate_during_training:

                            predict_golden = {'intent': [], 'slot': [], 'overall': []}

                            val_slot_loss, val_intent_loss = 0, 0
                            model.eval()
                            for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size,
                                                                                                  data_key='val'):
                                pad_batch = tuple(t.to(DEVICE) for t in pad_batch)

                                with torch.no_grad():
                                    slot_logits, intent_logits, slot_loss, intent_loss = model.forward(*pad_batch)
                                slot_logits, intent_logits, slot_loss, intent_loss = slot_logits.cpu(), intent_logits.cpu(), slot_loss.cpu(), intent_loss.cpu()
                                val_slot_loss += slot_loss.item() * real_batch_size
                                val_intent_loss += intent_loss.item() * real_batch_size
                                for j in range(real_batch_size):
                                    predicts = processor.recover_dialogue_act(dataloader,
                                                                              slot_logits[j], intent_logits[j],
                                                                              pad_batch[-2][j], ori_batch[j][6],
                                                                              ori_batch[j][7], ori_batch[j][8])
                                    labels = ori_batch[j][4]

                                    predict_golden['overall'].append({
                                        'predict': predicts,
                                        'golden': labels
                                    })
                                    predict_golden['slot'].append({
                                        'predict': [x for x in predicts if is_slot_da(x)],
                                        'golden': [x for x in labels if is_slot_da(x)]
                                    })
                                    predict_golden['intent'].append({
                                        'predict': [x for x in predicts if not is_slot_da(x)],
                                        'golden': [x for x in labels if not is_slot_da(x)]
                                    })

                            # for j in range(10):
                            #     writer.add_text('val_sample_{}'.format(j),
                            #                     json.dumps(predict_golden['overall'][j], indent=2, ensure_ascii=False),
                            #                     global_step=step)

                            total = len(dataloader.data['val'])
                            val_slot_loss /= total
                            val_intent_loss /= total
                            val_loss = val_slot_loss + val_intent_loss

                            writer.add_scalar('slot_loss/val', val_slot_loss, global_step=step)
                            writer.add_scalar('intent_loss/val', val_intent_loss, global_step=step)

                            for x in ['intent', 'slot', 'overall']:
                                precision, recall, F1 = calculateF1(predict_golden[x])
                                print('-' * 20 + x + '-' * 20)
                                print('\t Precision: %.2f' % (100 * precision))
                                print('\t Recall: %.2f' % (100 * recall))
                                print('\t F1: %.2f' % (100 * F1))

                                writer.add_scalar('val_{}/precision'.format(x), precision, global_step=step)
                                writer.add_scalar('val_{}/recall'.format(x), recall, global_step=step)
                                writer.add_scalar('val_{}/F1'.format(x), F1, global_step=step)

                            if F1 > best_val_f1:
                                best_val_f1 = F1
                                torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                                writer.add_text('best val overall F1', '%.2f' % (100 * best_val_f1), global_step=step)
                                print('best val F1 %.4f' % best_val_f1)
                                print('save on', output_dir)

                        elif step == max_step:
                            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                            print('save max step model on', output_dir)

                        train_slot_loss, train_intent_loss = 0, 0
                        train_steps = 0

            if args.do_eval:
                model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
                model.eval()
                batch_size = args.eval_batch_size

                data_key = 'test'
                predict_golden = {'intent': [], 'slot': [], 'overall': [], 'intent_tag_acc': []}
                test_slot_loss, test_intent_loss = 0, 0
                for pad_batch, ori_batch, real_batch_size in tqdm(
                        dataloader.yield_batches(batch_size, data_key=data_key)):
                    pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
                    with torch.no_grad():
                        slot_logits, intent_logits, slot_loss, intent_loss = model.forward(*pad_batch)
                    slot_logits, intent_logits, slot_loss, intent_loss = slot_logits.cpu(), intent_logits.cpu(), slot_loss.cpu(), intent_loss.cpu()
                    test_slot_loss += slot_loss.item() * real_batch_size
                    test_intent_loss += intent_loss.item() * real_batch_size
                    for j in range(real_batch_size):
                        predicts = processor.recover_dialogue_act(dataloader,
                                                                  slot_logits[j], intent_logits[j],
                                                                  pad_batch[-2][j], ori_batch[j][6], ori_batch[j][7],
                                                                  ori_batch[j][8])
                        labels = ori_batch[j][4]

                        intents, gold_intents, tags, gold_tags = processor.recover_intents_tags(dataloader,
                                                                                                slot_logits[j],
                                                                                                intent_logits[j],
                                                                                                pad_batch[-2][j],
                                                                                                pad_batch[-1][j],
                                                                                                pad_batch[-3][j])

                        predict_golden['overall'].append({
                            'predict': predicts,
                            'golden': labels
                        })
                        predict_golden['slot'].append({
                            'predict': [x for x in predicts if is_slot_da(x)],
                            'golden': [x for x in labels if is_slot_da(x)]
                        })
                        predict_golden['intent'].append({
                            'predict': [x for x in predicts if not is_slot_da(x)],
                            'golden': [x for x in labels if not is_slot_da(x)]
                        })
                        predict_golden['intent_tag_acc'].append({
                            'predict_intents': intents,
                            'gold_intents': gold_intents,
                            'predict_tags': tags,
                            'gold_tags': gold_tags
                        })

                log_file = open(os.path.join(output_dir, 'log.txt'), 'w')
                total = len(dataloader.data[data_key])
                test_slot_loss /= total
                test_intent_loss /= total
                print('%d samples %s' % (total, data_key), file=log_file)
                print('\t slot loss:', test_slot_loss, file=log_file)
                print('\t intent loss:', test_intent_loss, file=log_file)
                print('%d samples %s' % (total, data_key))
                print('\t slot loss:', test_slot_loss)
                print('\t intent loss:', test_intent_loss)

                overall_f1 = 0.
                for x in ['intent', 'slot', 'overall']:
                    precision, recall, F1 = calculateF1(predict_golden[x])
                    print('-' * 20 + x + '-' * 20, file=log_file)
                    print('\t Precision: %.2f' % (100 * precision), file=log_file)
                    print('\t Recall: %.2f' % (100 * recall), file=log_file)
                    print('\t F1: %.2f' % (100 * F1), file=log_file)
                    print('-' * 20 + x + '-' * 20)
                    print('\t Precision: %.2f' % (100 * precision))
                    print('\t Recall: %.2f' % (100 * recall))
                    print('\t F1: %.2f' % (100 * F1))
                    if x == 'overall':
                        writer.add_text('test overall F1', '%.2f' % (100 * F1))
                        overall_f1 = F1
                intent_acc, slot_acc, overall_acc = calculateAcc(predict_golden['intent_tag_acc'])
                print('intent acc %.2f, slot acc %.2f, overall acc %.2f' % (
                    100 * intent_acc, 100 * slot_acc, 100 * overall_acc), file=log_file)
                print('intent acc %.2f, slot acc %.2f, overall acc %.2f' % (
                    100 * intent_acc, 100 * slot_acc, 100 * overall_acc))

                output_file = os.path.join(output_dir, 'output.json')
                json.dump(predict_golden['overall'], open(output_file, 'w', encoding='utf-8'), indent=2,
                          ensure_ascii=False)
                output_file = os.path.join(output_dir, 'intent_tag.json')
                json.dump(predict_golden['intent_tag_acc'], open(output_file, 'w', encoding='utf-8'), indent=2,
                          ensure_ascii=False)

                writer.add_text('test overall acc', '%.2f' % (100 * overall_acc))
                eval_res.setdefault(train_ratio, {"f1": [], "acc": []})
                eval_res[train_ratio]["f1"].append(overall_f1 * 100)
                eval_res[train_ratio]["acc"].append(overall_acc * 100)

            writer.close()

    if args.do_eval:
        writer = SummaryWriter(os.path.join(args.output_dir, 'log'))
        writer.add_text('eval_res', json.dumps(eval_res))
        for train_ratio, res in eval_res.items():
            mean_f1, std_f1 = np.mean(res["f1"]), np.std(res["f1"])
            mean_acc, std_acc = np.mean(res["acc"]), np.std(res["acc"])
            writer.add_text('test_result',
                            'train ratio: %f avg. f1/acc: %.2f(%.2f)/%.2f(%.2f)' % (
                            train_ratio, mean_f1, std_f1, mean_acc, std_acc))
        writer.close()
