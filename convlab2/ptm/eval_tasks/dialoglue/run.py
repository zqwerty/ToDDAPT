import argparse
import json
import logging
import numpy as np
import os
import random
import torch

from typing import Any
from typing import Dict
from typing import TextIO
from typing import Tuple

from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange
from transformers import AdamW, BertTokenizer

from constants import SPECIAL_TOKENS
from data_readers import IntentDataset, SlotDataset, TOPDataset, DialogBertIntentDataset, DialogBertSlotDataset, DialogBertTOPDataset
from data_readers import TodBertIntentDataset, TodBertSlotDataset, TodBertTOPDataset
from bert_models import BertPretrain, IntentBertModel, JointSlotIntentBertModel, SlotBertModel
from dialogbert_models import DialogBertPretrain, IntentDialogBertModel, JointSlotIntentDialogBertModel, \
    SlotDialogBertModel
from convlab2.ptm.pretraining.model import DialogBertTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--val_data_path", type=str, default='')
    parser.add_argument("--mlm_data_path", type=str, default='')
    parser.add_argument("--token_vocab_path", type=str)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--output_dir_prefix", type=str, default='')
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--task", type=str, choices=["intent", "slot", "top"])
    parser.add_argument("--dump_outputs", action="store_true")
    parser.add_argument("--mlm_pre", action="store_true")
    parser.add_argument("--mlm_during", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--do_lowercase", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--device", default=0, type=int, help="GPU device #")
    parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--eval_file', help='file to write all results', default='')
    parser.add_argument('--run_probing', help='probing mode', action='store_true')
    parser.add_argument('--ignore_pooler', help='use hidden_state[0] instead of pooler(hidden_state[0]) as representation \
                                                  of [CLS]', action='store_true')
    return parser.parse_args()


def to_cuda(batch: dict):
    return {k: v.cuda() for k, v in batch.items()}


def evaluate(model: torch.nn.Module,
             eval_dataloader: DataLoader,
             tokenizer: Any,
             task: str = "intent",
             example: bool = False,
             device: int = 0,
             args: Any = None) -> Tuple[float, float, float]:
    model.eval()
    pred = []
    true = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        assert batch['attention_mask'].shape == batch['input_ids'].shape
        # assert batch['input_ids'].shape[1] == 50
        with torch.no_grad():
            # Move to GPU
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue

                    batch[key] = batch[key].to(device)

            if task == "intent":
                intent_logits, intent_loss = model(**batch)

                # Argmax to get predictions
                intent_preds = torch.argmax(intent_logits, dim=1).cpu().tolist()

                pred += intent_preds
                true += batch["intent_label"].cpu().tolist()

            elif task == "slot":
                slot_logits, slot_loss = model(**batch)

                # Argmax to get predictions
                slot_preds = torch.argmax(slot_logits, dim=2).detach().cpu().numpy()

                # Generate words, true slots and pred slots
                words = [tokenizer.decode([e]) for e in batch["input_ids"][0].tolist()]
                actual_gold_slots = batch["slot_labels"].cpu().numpy().squeeze().tolist()
                true_slots = [eval_dataloader.dataset.slot_idx_to_label[s] for s in actual_gold_slots]
                actual_predicted_slots = slot_preds.squeeze().tolist()
                pred_slots = [eval_dataloader.dataset.slot_idx_to_label[s] for s in actual_predicted_slots]

                # Find the last turn and only include that. Irrelevant for restaurant8k/dstc8-sgd.
                if '>' in words:
                    ind = words[::-1].index('>')
                    words = words[-ind:]
                    true_slots = true_slots[-ind:]
                    pred_slots = pred_slots[-ind:]

                # Filter out words that are padding

                # in tokenizers package, special tokens are replaced by ''
                # like ['', 'i', 'would', 'like', '7', ':', '45', 'please', '.', '', '']
                # while in transformers, special tokens are kept
                # like ['[CLS]', '[USR]', 'i', 'would', 'like', '7', ':', '45', 'please', '.', '[SEP]', '[PAD]']
                filt_words = [w for w in words if w not in ['', 'user', '[PAD]', '[CLS]', '[SEP]', '[USR]']]
                true_slots = [s for w, s in zip(words, true_slots) if w not in ['', 'user', '[PAD]', '[CLS]', '[SEP]', '[USR]']]
                pred_slots = [s for w, s in zip(words, pred_slots) if w not in ['', 'user', '[PAD]', '[CLS]', '[SEP]', '[USR]']]

                # print(words)
                # print(filt_words)
                # print(true_slots)
                # print(pred_slots)
                # print('=======================')
                # Convert to slot labels
                pred.append(pred_slots)
                true.append(true_slots)

                assert len(pred_slots) == len(true_slots)
                assert len(pred_slots) == len(filt_words)
            elif task == "top":
                intent_logits, slot_logits, _ = model(**batch)

                # Argmax to get intent predictions
                intent_preds = torch.argmax(intent_logits, dim=1).cpu().tolist()

                # Argmax to get slot predictions
                slot_preds = torch.argmax(slot_logits, dim=2).detach().cpu().numpy()
                intent_true = batch["intent_label"].cpu().tolist()

                # Only unmasked
                # pad_ind = batch["attention_mask"].tolist()[0].index(0)
                # actual_gold_slots = actual_gold_slots[1:pad_ind - 1]
                # actual_predicted_slots = actual_predicted_slots[1:pad_ind - 1]

                words = [tokenizer.decode([e]) for e in batch["input_ids"][0].tolist()]
                actual_gold_slots = batch["slot_labels"].cpu().numpy().squeeze().tolist()
                true_slots = [eval_dataloader.dataset.slot_idx_to_label[s] for s in actual_gold_slots]
                actual_predicted_slots = slot_preds.squeeze().tolist()
                pred_slots = [eval_dataloader.dataset.slot_idx_to_label[s] for s in actual_predicted_slots]
                true_slots = [s for w, s in zip(words, true_slots) if
                              w not in ['', 'user', '[PAD]', '[CLS]', '[SEP]', '[USR]']]
                pred_slots = [s for w, s in zip(words, pred_slots) if
                              w not in ['', 'user', '[PAD]', '[CLS]', '[SEP]', '[USR]']]

                pred.append((intent_preds if type(intent_preds) is int else intent_preds[0], pred_slots))
                true.append((intent_true[0], true_slots))


    def _extract(slot_labels):
        """
        Convert from IBO slot labels to spans.
        """
        slots = []
        cur_key = None
        start_ind = -1
        for i, s in enumerate(slot_labels):
            if s == "O" or s == "[PAD]":
                # Add on-going slot if there is one
                if cur_key is not None:
                    slots.append("{}:{}-{}".format(cur_key, start_ind, i))

                cur_key = None
                continue

            token_type, slot_key = s.split("-", 1)
            if token_type == "B":
                # If there is an on-going slot right now, add it
                if cur_key is not None:
                    slots.append("{}:{}-{}".format(cur_key, start_ind, i))

                cur_key = slot_key
                start_ind = i
            elif token_type == "I":
                # If the slot key doesn't match the currently active, this is invalid. 
                # Treat this as an O.
                if slot_key != cur_key:
                    if cur_key is not None:
                        slots.append("{}:{}-{}".format(cur_key, start_ind, i))

                    cur_key = None
                    continue

        # After the loop, add any oongoing slots
        if cur_key is not None:
            slots.append("{}:{}-{}".format(cur_key, start_ind, len(slot_labels)))

        return slots

    # Perform evaluation
    if task == "intent":
        if args.dump_outputs:
            pred_labels = [eval_dataloader.dataset.intent_idx_to_label.get(p) for p in pred]
            json.dump(pred_labels, open(args.output_dir + "outputs.json", "w+"))

        return sum(p == t for p, t in zip(pred, true)) / len(pred)
    elif task == "slot":
        pred_slots = [_extract(e) for e in pred]
        true_slots = [_extract(e) for e in true]

        if args.dump_outputs:
            json.dump(pred_slots, open(args.output_dir + "outputs.json", "w+"))

        slot_types = set([slot.split(":")[0] for row in true_slots for slot in row])
        slot_type_f1_scores = []

        for slot_type in slot_types:
            predictions_for_slot = [
                [p for p in prediction if slot_type in p] for prediction in pred_slots
            ]
            labels_for_slot = [
                [l for l in label if slot_type in l] for label in true_slots
            ]

            proposal_made = [len(p) > 0 for p in predictions_for_slot]
            has_label = [len(l) > 0 for l in labels_for_slot]
            prediction_correct = [
                prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)
            ]
            true_positives = sum([
                int(proposed and correct)
                for proposed, correct in zip(proposal_made, prediction_correct)
            ])
            num_predicted = sum([int(proposed) for proposed in proposal_made])
            num_to_recall = sum([int(hl) for hl in has_label])

            precision = true_positives / (1e-5 + num_predicted)
            recall = true_positives / (1e-5 + num_to_recall)

            f1_score = 2 * precision * recall / (1e-5 + precision + recall)
            slot_type_f1_scores.append(f1_score)

        return np.mean(slot_type_f1_scores)
    elif task == "top":
        # print(pred)

        if args.dump_outputs:
            pred_labels = [(intent, slots) for intent, slots in pred]
            json.dump(pred_labels, open(args.output_dir + "outputs.json", "w+"))

        return sum(p == t for p, t in zip(pred, true)) / len(pred)


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    ignore_idx = -100
    if isinstance(tokenizer, BertWordPieceTokenizer):
        ignore_idx = -1
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(labels == 0, dtype=torch.bool), value=0.0)
    # probability_matrix.masked_fill_(torch.tensor(labels >= 30522, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = ignore_idx  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    try:
        inputs[indices_replaced] = tokenizer.token_to_id("[MASK]")
    except:
        inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    try:
        random_words = torch.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
    except:
        random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)

    inputs[indices_random] = random_words[indices_random].cuda()

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def printBatch(batch, tokenizer, dataset):
    # intent
    # for seq, pos_id, turn_id, role_id, attn_mask, label in zip(batch['input_ids'], batch['position_ids'], batch['turn_ids'], batch['role_ids'], batch['attention_mask'], batch['intent_label']):
    #     print('=========================')
    #     print(tokenizer.convert_ids_to_tokens(seq))
    #     print(attn_mask)
    #     print(pos_id)
    #     print(turn_id)
    #     print(role_id)
    #     print(dataset.intent_idx_to_label[label.item()])
    # raise

    # mlm dialogbert
    # for seq, pos_id, turn_id, role_id, attn_mask, label in zip(batch['input_ids'], batch['position_ids'], batch['turn_ids'], batch['role_ids'], batch['attention_mask'], batch['mlm_labels']):
    #     print('=========================')
    #     print(tokenizer.convert_ids_to_tokens(seq))
    #     print(attn_mask)
    #     print(pos_id)
    #     print(turn_id)
    #     print(role_id)
    #     print(tokenizer.convert_ids_to_tokens(label))
    # raise

    # mlm bert
    # for seq, label in zip(batch['input_ids'], batch['mlm_labels']):
    #     print('=========================')
    #     print([tokenizer.id_to_token(_id) for _id in seq])
    #     # print(label)
    #     print([tokenizer.id_to_token(_id) if _id > -1 else '[UNK]' for _id in label])
    # raise

    # slot
    # for seq, pos_id, turn_id, role_id, attn_mask, label in zip(batch['input_ids'], batch['position_ids'], batch['turn_ids'], batch['role_ids'], batch['attention_mask'], batch['slot_labels']):
    #     print('=========================')
    #     print(tokenizer.convert_ids_to_tokens(seq))
    #     print([dataset.slot_idx_to_label[l.item()] for l in label])
    #     print(attn_mask)
    #     print(pos_id)
    #     print(turn_id)
    #     print(role_id)

    # bert slot
    # for seq, attn_mask, label in zip(batch['input_ids'], batch['attention_mask'], batch['slot_labels']):
    #     print('=========================')
    #     print([tokenizer.id_to_token(s) for s in seq])
    #     print([dataset.slot_idx_to_label[l.item()] for l in label])
    #     print(attn_mask)

    # joint
    for seq, label in zip(batch['input_ids'], batch['slot_labels']):
        print('=========================')
        # print(seq.tolist())
        # list_seq = seq.tolist()
        print(seq)
        print([tokenizer.id_to_token(_id) for _id in seq])
        print([dataset.slot_idx_to_label[l.item()] for l in label])




def train(args, rep):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Rename output dir based on arguments
    if args.output_dir == "":
        cwd = os.getcwd() if args.output_dir_prefix == "" else args.output_dir_prefix
        base = args.model_name_or_path.split("/")[-2]
        data_path = '_' + '_'.join(args.train_data_path.split("/")[-2:]).replace(".csv", "")
        mlm_pre = "_mlmpre" if args.mlm_pre else ""
        mlm_dur = "_mlmdur" if args.mlm_during else ""
        args_add_str = ""
        if args.ignore_pooler:
            args_add_str = "_ignorepooler"
        if args.run_probing:
            args_add_str += "_probing"
        fewshot_str = "_fewshot" if args.repeat > 1 else ""
        name = base + data_path + args_add_str + mlm_pre + mlm_dur + '_' + args.model_type + fewshot_str + "_v{}".format(rep)
        args.output_dir = os.path.join(cwd, "checkpoints", name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.num_epochs == 0:
        # This means we're evaluating. Don't create the directory.
        pass
    else:
        raise Exception("Directory {} already exists".format(args.output_dir))

    # Dump arguments to the checkpoint directory, to ensure reproducability.
    if args.num_epochs > 0:
        json.dump(args.__dict__, open(os.path.join(args.output_dir, 'args.json'), "w+"))
        torch.save(args, os.path.join(args.output_dir, "run_args"))

    # Configure tensorboard writer
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    task_specific_args = {}
    # Configure tokenizer
    if args.model_type == 'bert':
        token_vocab_name = os.path.basename(args.token_vocab_path).replace(".txt", "")
        tokenizer = BertWordPieceTokenizer(args.token_vocab_path,
                                           lowercase=args.do_lowercase)
        tokenizer.enable_padding(max_length=args.max_seq_length)

        # if args.num_epochs > 0:
        #     tokenizer.save_model(args.output_dir)

        # Data readers
        if args.task == "intent":
            dataset_initializer = IntentDataset
        elif args.task == "slot":
            dataset_initializer = SlotDataset
        elif args.task == "top":
            dataset_initializer = TOPDataset
        else:
            raise ValueError("Not a valid task type: {}".format(args.task))

    elif args.model_type == 'dialogbert':
        tokenizer = DialogBertTokenizer.from_pretrained(args.model_name_or_path)

        token_vocab_name = 'dialogbert'
        if args.task == "intent":
            dataset_initializer = DialogBertIntentDataset
        elif args.task == "slot":
            dataset_initializer = DialogBertSlotDataset
            task_specific_args.update({'help_tokenizer': BertWordPieceTokenizer(args.token_vocab_path,
                                                                                lowercase=args.do_lowercase)})
        elif args.task == "top":
            dataset_initializer = DialogBertTOPDataset
            task_specific_args.update({'help_tokenizer': BertWordPieceTokenizer(args.token_vocab_path,
                                                                                lowercase=args.do_lowercase)})
        else:
            raise ValueError("Not a valid task type: {}".format(args.task))
    elif args.model_type == 'todbert':
        token_vocab_name = os.path.basename(args.token_vocab_path).replace(".txt", "")
        print(args.token_vocab_path)
        tokenizer = BertWordPieceTokenizer(args.token_vocab_path,
                                           lowercase=args.do_lowercase)
        tokenizer.enable_padding(max_length=args.max_seq_length)
        tokenizer.add_tokens(['[sys]', '[usr]'])

        if args.task == "intent":
            dataset_initializer = TodBertIntentDataset
        elif args.task == "slot":
            dataset_initializer = TodBertSlotDataset
        elif args.task == "top":
            dataset_initializer = TodBertTOPDataset
        else:
            raise ValueError("Not a valid task type: {}".format(args.task))
    else:
        raise ValueError()

    train_dataset = dataset_initializer(args.train_data_path,
                                        tokenizer,
                                        args.max_seq_length,
                                        token_vocab_name,
                                        **task_specific_args)

    if args.mlm_data_path != '':
        mlm_dataset = dataset_initializer(args.mlm_data_path,
                                          tokenizer,
                                          args.max_seq_length,
                                          token_vocab_name,
                                          **task_specific_args)
    else:
        mlm_dataset = train_dataset

    val_dataset = dataset_initializer(args.val_data_path,
                                      tokenizer,
                                      args.max_seq_length,
                                      token_vocab_name,
                                      **task_specific_args) if args.val_data_path else None


    test_dataset = dataset_initializer(args.test_data_path,
                                       tokenizer,
                                       args.max_seq_length,
                                       token_vocab_name,
                                       **task_specific_args)

    # Data loaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  pin_memory=True)

    mlm_dataloader = DataLoader(dataset=mlm_dataset,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                pin_memory=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                pin_memory=True) if val_dataset else None

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 pin_memory=True)

    # Load model
    if args.model_type == 'bert' or args.model_type == 'todbert':
        if args.task == "intent":
            model = IntentBertModel(args.model_name_or_path,
                                    dropout=args.dropout,
                                    num_intent_labels=len(train_dataset.intent_label_to_idx),
                                    probing=args.run_probing,
                                    ignore_pooler=args.ignore_pooler,
                                    )
        elif args.task == "slot":
            model = SlotBertModel(args.model_name_or_path,
                                  dropout=args.dropout,
                                  num_slot_labels=len(train_dataset.slot_label_to_idx),
                                  probing=args.run_probing,
                                  ignore_pooler=args.ignore_pooler,
                                  )
        elif args.task == "top":
            model = JointSlotIntentBertModel(args.model_name_or_path,
                                             dropout=args.dropout,
                                             num_intent_labels=len(train_dataset.intent_label_to_idx),
                                             num_slot_labels=len(train_dataset.slot_label_to_idx),
                                             probing=args.run_probing,
                                             ignore_pooler=args.ignore_pooler,
                                             )
        else:
            raise ValueError("Cannot instantiate model for task: {}".format(args.task))

    elif args.model_type == 'dialogbert':
        if args.task == "intent":
            model = IntentDialogBertModel(args.model_name_or_path,
                                          dropout=args.dropout,
                                          num_intent_labels=len(train_dataset.intent_label_to_idx),
                                          probing=args.run_probing,
                                          ignore_pooler=args.ignore_pooler,
                                          )
        elif args.task == "slot":
            model = SlotDialogBertModel(args.model_name_or_path,
                                        dropout=args.dropout,
                                        num_slot_labels=len(train_dataset.slot_label_to_idx),
                                        probing=args.run_probing,
                                        ignore_pooler=args.ignore_pooler,
                                        )
        elif args.task == 'top':
            model = JointSlotIntentDialogBertModel(args.model_name_or_path,
                                                     dropout=args.dropout,
                                                     num_intent_labels=len(train_dataset.intent_label_to_idx),
                                                     num_slot_labels=len(train_dataset.slot_label_to_idx),
                                                   probing=args.run_probing,
                                                   ignore_pooler=args.ignore_pooler,
                                                   )
        else:
            raise

    if torch.cuda.is_available():
        model.to(args.device)

    # Initialize MLM model
    if args.model_type == 'bert':
        if args.mlm_pre or args.mlm_during:
            pre_model = BertPretrain(args.model_name_or_path)
            mlm_optimizer = AdamW(pre_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
            if torch.cuda.is_available():
                pre_model.to(args.device)

    elif args.model_type == 'dialogbert':
        if args.mlm_pre or args.mlm_during:
            pre_model = DialogBertPretrain(args.model_name_or_path)
            mlm_optimizer = AdamW(pre_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
            if torch.cuda.is_available():
                pre_model.to(args.device)

    # MLM Pre-train
    if args.mlm_pre and args.num_epochs > 0:
        # Maintain most recent score per label. 
        for epoch in trange(3, desc="Pre-train Epochs"):

            pre_model.train()
            epoch_loss = 0
            num_batches = 0
            for batch in tqdm(mlm_dataloader):
                num_batches += 1

                # Train model
                if "input_ids" in batch:
                    inputs, labels = mask_tokens(batch["input_ids"].cuda(), tokenizer)
                else:
                    inputs, labels = mask_tokens(batch["ctx_input_ids"].cuda(), tokenizer)

                mlm_dict = {'input_ids': inputs, 'mlm_labels': labels}
                if args.model_type == 'dialogbert':
                    mlm_dict.update({'attention_mask': batch['attention_mask'],
                                     'turn_ids': batch['turn_ids'],
                                     'position_ids': batch['position_ids'],
                                     'role_ids': batch['role_ids']})

                mlm_dict = to_cuda(mlm_dict)
                loss = pre_model(**mlm_dict)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item()

                if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(pre_model.parameters(), args.max_grad_norm)

                    mlm_optimizer.step()
                    pre_model.zero_grad()

            LOGGER.info("Epoch loss: {}".format(epoch_loss / num_batches))

        # Transfer BERT weights
        model.bert_model = pre_model.bert_model.bert

    # Train
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    global_step = 0
    metrics_to_log = {}
    best_score = -1
    patience = 0
    for epoch in trange(args.num_epochs, desc="Epoch"):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(train_dataloader):

            num_batches += 1
            global_step += 1
            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue

                    batch[key] = batch[key].to(args.device)

            # Train model
            if args.task == "intent":

                _, intent_loss = model(**to_cuda(batch))

                if args.grad_accum > 1:
                    intent_loss = intent_loss / args.grad_accum

                intent_loss.backward()
                epoch_loss += intent_loss.item()

            elif args.task == "slot":
                _, slot_loss = model(**to_cuda(batch))

                if args.grad_accum > 1:
                    slot_loss = slot_loss / args.grad_accum
                slot_loss.backward()
                epoch_loss += slot_loss.item()

            elif args.task == "top":

                _, _, loss = model(**to_cuda(batch))

                if args.grad_accum > 1:
                    loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item()

            if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()

        LOGGER.info("Epoch loss: {}".format(epoch_loss / num_batches))

        # Evaluate and save checkpoint
        score = evaluate(model, val_dataloader, tokenizer, task=args.task, device=args.device, args=args)
        metrics_to_log["eval_score"] = score
        LOGGER.info("Task: {}, score: {}---".format(args.task,
                                                    score))

        if score < best_score:
            patience += 1
        else:
            patience = 0

        if score > best_score:
            LOGGER.info("New best results found for {}! Score: {}".format(args.task,
                                                                          score))
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
            best_score = score

        for name, val in metrics_to_log.items():
            tb_writer.add_scalar(name, val, global_step)

        if patience >= args.patience:
            LOGGER.info("Stopping early due to patience")
            break

        # Run MLM during training
        if args.mlm_during:
            pre_model.train()
            epoch_loss = 0
            num_batches = 0
            for batch in tqdm(mlm_dataloader):
                num_batches += 1

                # Train model
                if "input_ids" in batch:
                    inputs, labels = mask_tokens(batch["input_ids"].cuda(), tokenizer)
                else:
                    inputs, labels = mask_tokens(batch["ctx_input_ids"].cuda(), tokenizer)

                mlm_dict = {'input_ids': inputs, 'mlm_labels': labels}
                if args.model_type == 'dialogbert':
                    mlm_dict.update({'turn_ids': batch['turn_ids'], 'position_ids': batch['position_ids'],
                                     'role_ids': batch['role_ids'], 'attention_mask': batch['attention_mask']})

                mlm_dict = to_cuda(mlm_dict)
                loss = pre_model(**mlm_dict)

                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

                loss.backward()
                epoch_loss += loss.item()

                if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(pre_model.parameters(), args.max_grad_norm)

                    mlm_optimizer.step()
                    pre_model.zero_grad()

            LOGGER.info("MLMloss: {}".format(epoch_loss / num_batches))

    # Evaluate on test set
    LOGGER.info("Loading up best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    score = evaluate(model, test_dataloader, tokenizer, task=args.task, device=args.device, args=args)
    print("Best result for {}: Score: {}".format(args.task, score))
    tb_writer.add_scalar("final_test_score", score, global_step)
    tb_writer.close()
    return score


if __name__ == "__main__":
    args = read_args()
    print(args)
    if args.eval_file == '':
        args_add_str = ""
        if args.ignore_pooler:
            args_add_str = "_ignorepooler"
        if args.run_probing:
            args_add_str += "_probing"
        ckpt_name = '_'.join(args.model_name_or_path.split('/')[-2:]) + args_add_str
        if args.repeat > 1:
            ckpt_name += '_fewshot'
        save_log_dir = os.getcwd() if args.output_dir_prefix == "" else args.output_dir_prefix
        args.eval_file = os.path.join(save_log_dir, 'eval_scores_{}'.format(ckpt_name))


    scores = []
    seeds = [33, 42, 19, 55, 34, 63]
    for i in range(args.repeat):
        if args.num_epochs > 0:
            args.output_dir = ""

        args.seed = seeds[i] if i < len(seeds) else random.randint(1, 999)
        scores.append(train(args, i))

        print("Average score so far:", np.mean(scores))

    print(scores)
    print(np.mean(scores), max(scores), min(scores))
    output_txt = open(args.eval_file, 'a')
    output_txt.write('============================='+'\n')
    output_txt.write(args.model_type+'\n')
    output_txt.write(args.model_name_or_path+'\n')
    output_txt.write(json.dumps(args.test_data_path.split('/')[-2])+'\n')
    output_txt.write('scores'+'\n')
    output_txt.write(json.dumps(scores)+'\n')
    output_txt.write('average scores'+'\n')
    output_txt.write(json.dumps(np.mean(scores))+'\n')
    output_txt.close()

