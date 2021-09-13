import json
import os
import random
import numpy as np
from pprint import pprint
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from convlab2.ptm import DATA_PTM_PATH
from convlab2.ptm.eval_tasks.dst import parser, get_transformer_settings
from convlab2.ptm.eval_tasks.dst.model import DstModel
from convlab2.ptm.eval_tasks.dst.model_split import DstModelSplit
from convlab2.ptm.eval_tasks.dst.data_utils.MultiwozDataset import MultiwozDataset
from convlab2.ptm.eval_tasks.dst.data_utils.MultiwozDataset_split import MultiwozDatasetSplit
from convlab2.ptm.pretraining.model.optimization import AdamW, get_linear_schedule_with_warmup


def do_evaluation(model_path, eval_config_path, output_error=False):
    with open(eval_config_path, "r") as f:
        eval_config = json.load(f)
    model_dir, eval_model = os.path.split(model_path)
    eval_name = eval_model + "_eval_results.txt"
    print("test results saved to {}".format(os.path.join(model_dir, eval_name)))
    f = open(os.path.join(model_dir, eval_name), "w")
    print("test:")
    f.write("test:\n")
    # test_loader = DataLoader(test_set, batch_size=eval_batch_size, collate_fn=test_set.collate_eval)
    error_dir = model_dir if output_error else None
    dst_res, val_res = test_func(test_loader, eval_config, error_dir=error_dir)
    for k, v in dst_res.items():
        print(k)
        print(v)
        f.write(str(k) + "\n")
        f.write(str(v) + "\n")
    for k, v in val_res.items():
        print(k)
        print(v)
        f.write(str(k) + "\n")
        f.write(str(v) + "\n")

    f.close()


if __name__ == '__main__':
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=8)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=16)
    parser.add_argument('--w_update', type=float, default=1.)
    parser.add_argument('--w_value', type=float, default=1.)
    parser.add_argument('--w_span', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dataset', default='multiwoz21', help='dataset training on.')
    parser.add_argument('--model_name', default='model', help='model name to be saved.')
    parser.add_argument('--output_dir', default='model', help='model output directory name.')
    parser.add_argument('--transformer', choices=['bert', 'tod-bert', 'dialog-bert'], default='bert', help='transformer model to be used in DST model.')
    parser.add_argument('--transformer_path', default='bert-base-uncased', help='pre-trained transformer model name or path.')
    parser.add_argument('--value-cat-with', choices=['cls', 'slot'], default='cls')
    parser.add_argument('--span-cat-with', choices=['cls', 'slot'], default='cls')
    parser.add_argument('--logging_steps', type=int, default=100, help="steps between logging.")
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--eval_config', type=str, default="", help="indicate the evaluation metrics.")
    parser.add_argument('--ckpt', default="", help="the path for a checkpoint. can be used in evaluation.")
    parser.add_argument('--ratio', type=float, default=1, help="the ratio we split the training se.t")
    parser.add_argument('--update_ratio', type=float, help='sample ratio of train data for update id=1.', default=1)
    parser.add_argument('--save_all', action="store_true", help="save all checkpoints in the training process.")
    parser.add_argument('--grad_acc', type=int, default=1, help="gradient accumulation steps.")
    parser.add_argument('--output_error', action="store_true", help="output error analysis file.")
    parser.add_argument('--eval_ratio', type=float, default=1, help="the ratio we split the test/valid set.")
    parser.add_argument('--split', action="store_true", help="use the model that seperately encodes descriptions and dialog.")
    parser.add_argument("--block_size", type=int, default=512, help="the max input sentence length.")
    parser.add_argument("--balance", action="store_true", help="use different weights of update. the weights are hard-coded.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warm_up steps used in scheduler.")
    parser.add_argument("--sched", action="store_true", help="use learning rate scheduler. if not set, the learning rate is always constant.")
    parser.add_argument('--seed', type=int, default="23333")


    args = parser.parse_args()
    pprint(args)

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random seed: {}'.format(SEED))

    n_gpu = 0 if args.device == "cpu" else torch.cuda.device_count()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir_name = '{}_wu({})_wv({})_ws({})_bs{}_lr({})_upd{}'.format(
        args.model_name,
        args.w_update,
        args.w_value,
        args.w_span,
        max(1, n_gpu) * args.per_gpu_train_batch_size * args.grad_acc,
        args.lr,
        args.update_ratio,
    )
    model_dir_path = os.path.join(args.output_dir, model_dir_name, str(args.ratio), str(args.seed))
    log_dir_path = os.path.join(model_dir_path, "logs")
    best_model_name = '{}_best.pt'.format(args.model_name)
    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(log_dir_path, exist_ok=True)

    with open(os.path.join(model_dir_path, "args.json"), "w") as f:
        json.dump(vars(args), f)
    
    if args.dataset not in ['multiwoz21', 'multiwoz25']:
        print('train on MultiWOZ dataset')
        exit(1)
    if args.do_eval and len(args.eval_config) == 0:
        print('must specify an eval config')
        exit(1)

    data_dir = os.path.join(DATA_PTM_PATH, args.dataset)
    print(data_dir)
    ontology = json.load(open(os.path.join(data_dir, 'ontology.json')))
    tokenizer_cls, model_cls, config_cls = get_transformer_settings(args.transformer)
    tokenizer = tokenizer_cls.from_pretrained(args.transformer_path)
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[USR]', '[SYS]', '[DOMAIN]', '[SLOT]', '[VALUE]']
    })
    print(tokenizer.convert_tokens_to_ids(['[USR]', '[SYS]', '[DOMAIN]', '[SLOT]', '[VALUE]']))

    dataset_class = MultiwozDatasetSplit if args.split else MultiwozDataset

    if args.do_train:
        train_set = dataset_class(args, tokenizer, data_dir, 'train', ratio=args.ratio, update_ratio=args.update_ratio)
        val_set = dataset_class(args, tokenizer, data_dir, 'val', ratio=args.eval_ratio)
    if args.do_eval:
        test_set = dataset_class(args, tokenizer, data_dir, 'test', ratio=args.eval_ratio)

    train_batch_size = max(1, n_gpu) * args.per_gpu_train_batch_size
    eval_batch_size = max(1, n_gpu) * args.per_gpu_eval_batch_size
    if args.do_train:
        train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, collate_fn=train_set.collate)
        val_loader = DataLoader(val_set, batch_size=eval_batch_size, collate_fn=val_set.collate_eval)
    if args.do_eval:
        test_loader = DataLoader(test_set, batch_size=eval_batch_size, collate_fn=test_set.collate_eval)

    model_class = DstModelSplit if args.split else DstModel

    dst_model = model_class(args, ontology, tokenizer).to(args.device)
    dst_model.resize_token_embeddings(len(tokenizer))
    if args.ckpt:
        dst_model.load_state_dict(torch.load(args.ckpt))
    test_func = dst_model.test

    if args.do_train:
        t_total = len(train_loader) // args.grad_acc * args.epochs
    else:
        t_total = 10
    param_optimizer = [(n, p) for n, p in dst_model.named_parameters() if p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': args.lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    if args.sched:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    if n_gpu > 1:
        dst_model = torch.nn.DataParallel(dst_model)

    best_loss = float("inf")
    best_joint_acc = 0
    best_epoch = -1

    global_steps = 0

    tot_loss = {
        k: 0 for k in ['update', 'value', 'start', 'end']
    }

    trn_loss = {
        k: 0.0 for k in ['tot', 'update', 'value', 'start', 'end']
    }
    logging_loss = {
        k: 0.0 for k in ['tot', 'update', 'value', 'start', 'end']
    }

    if args.do_train:
        eval_config = {
            "all_fixed": {
                "prev_truth": False,
                "upd_truth": False,
                "value_truth": False,
                "label_type": "fixed"
            },
        }
        patience = 0
        for epoch in range(args.epochs):
            dst_model.train()
            # implement for split temporarily
            for step, data in enumerate(tqdm(train_loader, desc=f'training epoch {epoch}', ncols=80)):
                if args.split:
                    ids, inputs, span_mask, labels = data
                    pred, loss = dst_model.forward(inputs, span_mask, labels)
                else:
                    ids, inputs, domain_pos, slot_pos, value_mask, span_mask, labels = data
                    pred, loss = dst_model.forward(inputs, domain_pos, slot_pos, value_mask, span_mask, labels)

                if n_gpu > 1:
                    for k, v in loss.items():
                        loss[k] = v.mean(dim=0)
                if args.grad_acc > 1:
                    for k in loss:
                        loss[k] = loss[k] / args.grad_acc

                if args.split:
                    # sum up loss of all slots
                    for k, v in loss.items():
                        loss[k] = v.sum()

                loss['tot'].backward()

                for k in trn_loss:
                    trn_loss[k] += loss[k].item()

                if (step + 1) % args.grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(dst_model.parameters(), 1.0)
                    optimizer.step()
                    if args.sched:
                        scheduler.step()
                    dst_model.zero_grad()
                    global_steps += 1
                    if global_steps % args.logging_steps == 0:
                        print('\ntrain loss:')
                        for k in trn_loss:
                            print('{}: {}'.format(k, (trn_loss[k] - logging_loss[k]) / args.logging_steps))
                        if args.sched:
                            print("lr: {}".format(scheduler.get_lr()))
                        logging_loss = deepcopy(trn_loss)

            dst_res, val_res = test_func(val_loader, eval_config)
            joint_acc = dst_res["all_fixed"]["joint_acc"]
            with open(os.path.join(log_dir_path, "val_log_e{}.txt".format(epoch)), "w") as f:
                f.write("joint_acc: {}\n".format(joint_acc))
                f.write("val results: \n")
                for k, v in val_res.items():
                    f.write("{}\n".format(k))
                    f.write("{}\n".format(v))
                f.write("\n")
                f.write("dst results: \n")
                for k, v in dst_res.items():
                    f.write("{}\n".format(k))
                    f.write("{}\n".format(v))
                f.write("\n")
            
            print('best_joint_acc: {}, val results: {}, dst results: {}'.format(
                best_joint_acc, val_res, dst_res))

            if joint_acc > best_joint_acc:
                patience = 0
                best_joint_acc = dst_res["all_fixed"]["joint_acc"]
                best_epoch = epoch
                model_to_save = (dst_model.module if hasattr(dst_model, "module") else dst_model)
                torch.save(model_to_save.state_dict(), os.path.join(model_dir_path, best_model_name))
                print('best model updated at epoch {}'.format(dst_res))
            else:
                patience += 1
                if patience == args.patience:
                    print('run out of patience. best model at {}. best joint acc: {}'.format(
                        best_epoch, best_joint_acc))
                    with open(os.path.join(log_dir_path, "val_end.txt"), "w") as f:
                        f.write("best model saved at {}\n".format(best_epoch))
                        f.write("best joint acc: {}\n".format(best_joint_acc))
                    break

            if args.save_all:
                model_to_save = (dst_model.module if hasattr(dst_model, "module") else dst_model)
                model_full_name = '{}_e{}.pt'.format(args.model_name, epoch)
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, model_dir_name, model_full_name))

        if args.do_eval:
            do_evaluation(os.path.join(model_dir_path, best_model_name), args.eval_config, output_error=args.output_error)

    elif args.do_eval:
        do_evaluation(args.ckpt, args.eval_config, output_error=args.output_error)
    
    else:
        print("nothing to do...")
