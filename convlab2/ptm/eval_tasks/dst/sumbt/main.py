import argparse
import collections
import csv
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from BeliefTrackerSlotQueryMultiSlot import BeliefTracker

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--use_dialog_bert', action='store_true')
parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help='bert model name or path')
parser.add_argument("--task_name",
                    default='bert-gru-sumbt',
                    type=str,
                    help="The name of the task to train: bert-gru-sumbt, bert-lstm-sumbt"
                         "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
parser.add_argument("--output_dir",
                    default='log',
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_dir",
                    default='log',
                    type=str,
                    help="The path to load trained models.")
parser.add_argument("--target_slot",
                    default='all',
                    type=str,
                    help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'")
parser.add_argument("--fix_utterance_encoder",
                    action='store_true',
                    help="Do not train BERT utterance encoder")

## Other parameters
parser.add_argument("--max_seq_length",
                    default=64,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--max_label_length",
                    default=32,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--max_turn_length",
                    default=22,
                    type=int,
                    help="The maximum total input turn length. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument('--hidden_dim',
                    type=int,
                    default=400,
                    help="hidden dimension used in belief tracker")
parser.add_argument('--num_rnn_layers',
                    type=int,
                    default=1,
                    help="number of RNN layers")
parser.add_argument('--zero_init_rnn',
                    action='store_true',
                    help="set initial hidden of rnns zero")
parser.add_argument('--attn_head',
                    type=int,
                    default=4,
                    help="the number of heads in multi-headed attention")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the test set.")
parser.add_argument("--distance_metric",
                    type=str,
                    default="euclidean",
                    help="The metric for distance between label embeddings: cosine, euclidean.")
parser.add_argument("--train_batch_size",
                    default=6,
                    type=int,
                    help="Total dialog batch size for training.")
parser.add_argument("--dev_batch_size",
                    default=1,
                    type=int,
                    help="Total dialog batch size for validation.")
parser.add_argument("--eval_batch_size",
                    default=8,
                    type=int,
                    help="Total dialog batch size for evaluation.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for BertAdam.")
parser.add_argument("--num_train_epochs",
                    default=300,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--patience",
                    default=10,
                    type=float,
                    help="The number of epochs to allow no further improvement.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether t"
                         "o use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")

args = parser.parse_args()
print(args)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':  # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()
        ontology = json.load(open(os.path.join(config.data_dir,  f"ontology.json")))
        for slot in ontology.keys():
            ontology[slot].append("none")

        if not config.target_slot == 'all':
            slot_idx = {'attraction': '0:1:2', 'bus': '3:4:5:6', 'hospital': '7',
                        'hotel': '8:9:10:11:12:13:14:15:16:17',
                        'restaurant': '18:19:20:21:22:23:24', 'taxi': '25:26:27:28', 'train': '29:30:31:32:33:34'}
            target_slot = []
            for key, value in slot_idx.items():
                if key != config.target_slot:
                    target_slot.append(value)
            config.target_slot = ':'.join(target_slot)


        # sorting the ontology according to the alphabetic order of the slots
        ontology = collections.OrderedDict(sorted(ontology.items()))

        # select slots to train
        nslots = len(ontology.keys())
        target_slot = list(ontology.keys())
        if config.target_slot == 'all':
            self.target_slot_idx = [*range(0, nslots)]
        else:
            self.target_slot_idx = sorted([int(x) for x in config.target_slot.split(':')])

        for idx in range(0, nslots):
            if not idx in self.target_slot_idx:
                del ontology[target_slot[idx]]

        self.ontology = ontology
        self.target_slot = list(self.ontology.keys())

        for i, slot in enumerate(self.target_slot):
            if slot == "pricerange":
                self.target_slot[i] = "pricerange"

        logger.info('Processor: target_slot')
        logger.info(self.target_slot)

    def get_train_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", accumulation)

    def get_dev_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", accumulation)

    def get_test_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", accumulation)

    def get_labels(self):
        """See base class."""
        return [self.ontology[slot] for slot in self.target_slot]

    def _create_examples(self, lines, set_type, accumulation=False):
        """Creates examples for the training and dev sets."""
        prev_dialogue_index = None
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, line[0], line[1])  # line[0]: dialogue index, line[1]: turn index
            if accumulation:
                if prev_dialogue_index is None or prev_dialogue_index != line[0]:
                    text_a = line[2]
                    text_b = line[3]
                    prev_dialogue_index = line[0]
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = line[2] + " # " + text_a
                    text_b = line[3] + " # " + text_b
            else:
                text_a = line[2]  # line[2]: user utterance
                text_b = line[3]  # line[3]: system response

            label = [line[4 + idx] for idx in self.target_slot_idx]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
    slot_dim = len(label_list)

    features = []
    prev_dialogue_idx = None
    all_padding = [0] * max_seq_length
    all_padding_len = [0, 0]

    max_turn = 0
    for (ex_index, example) in enumerate(examples):
        if max_turn < int(example.guid.split('-')[2]):
            max_turn = int(example.guid.split('-')[2])
    max_turn_length = min(max_turn + 1, max_turn_length)
    logger.info("max_turn_length = %d" % max_turn)

    for (ex_index, example) in enumerate(examples):
        tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]
        tokens_b = None
        if example.text_b:
            tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"; or [CLS] [USR] ... [SEP] [SYS] ... [SEP] for dialog-bert
            reserve = 5 if args.use_dialog_bert else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - reserve)
        else:
            reserve = 3 if args.use_dialog_bert else 2
            # Account for [CLS] and [SEP] with "- 2"; or [CLS] [USR] ... [SEP] for dialog-bert
            tokens_a = tokens_a[:max_seq_length - reserve]

        tokens_head = ['[CLS]']
        if args.use_dialog_bert:
            tokens_head.append('[USR]')
        tokens = tokens_head + tokens_a + ["[SEP]"]
        input_len = [len(tokens), 0]
        if tokens_b:
            tokens_head = []
            if args.use_dialog_bert:
                tokens_head.append('[SYS]')
            tokens += tokens_head + tokens_b + ["[SEP]"]
            input_len[1] = len(tokens_head) + len(tokens_b) + 1
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length


        label_id = []
        label_info = 'label: '
        for i, label in enumerate(example.label):
            if label == 'do not care':
                label = 'dontcare'
            # print(i)
            label_id.append(label_map[i][label])
            label_info += '%s (id = %d) ' % (label, label_map[i][label])

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_len: %s" % " ".join([str(x) for x in input_len]))
            logger.info("label: " + label_info)

        curr_dialogue_idx = example.guid.split('-')[1]
        # print(curr_dialogue_idx)
        curr_turn_idx = int(example.guid.split('-')[2])

        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           label_id=[-1] * slot_dim)] \
                            * (max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   label_id=[-1] * slot_dim)] \
                    * (max_turn_length - prev_turn_idx - 1)
    assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)

    return all_input_ids, all_input_len, all_label_ids


def get_label_embedding(labels, max_seq_length, tokenizer, device):
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def eval_all_accs(pred_slot, labels, accuracies):
    def _eval_acc(_pred_slot, _labels):
        slot_dim = _labels.size(-1)
        accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_labels[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_labels > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        return joint_acc, slot_acc, num_turn, num_data

    # 7 domains
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot, labels)
    accuracies['joint7'] += joint_acc
    accuracies['slot7'] += slot_acc
    accuracies['num_turn'] += num_turn
    accuracies['num_slot7'] += num_data

    # restaurant domain
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot[:, :, 18:25], labels[:, :, 18:25])
    accuracies['joint_rest'] += joint_acc
    accuracies['slot_rest'] += slot_acc
    accuracies['num_slot_rest'] += num_data

    pred_slot5 = torch.cat((pred_slot[:, :, 0:3], pred_slot[:, :, 8:]), 2)
    label_slot5 = torch.cat((labels[:, :, 0:3], labels[:, :, 8:]), 2)
    # print("pred_slot: ", pred_slot5)
    # print("label_slot5: ", label_slot5)
    # print("******************** A SEPERATE LINE *****************************")

    # 5 domains (excluding bus and hotel domain)
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot5, label_slot5)
    accuracies['joint5'] += joint_acc
    accuracies['slot5'] += slot_acc
    accuracies['num_slot5'] += num_data

    return accuracies


def save_configure(args, num_labels, ontology):
    with open(os.path.join(args.output_dir, "config.json"), 'w') as outfile:
        data = {"hidden_dim": args.hidden_dim,
                "num_rnn_layers": args.num_rnn_layers,
                "zero_init_rnn": args.zero_init_rnn,
                "max_seq_length": args.max_seq_length,
                "max_label_length": args.max_label_length,
                "num_labels": num_labels,
                "attn_head": args.attn_head,
                "distance_metric": args.distance_metric,
                "fix_utterance_encoder": args.fix_utterance_encoder,
                "task_name": args.task_name,
                "bert_model": args.bert_model,
                "ontology": ontology}
        json.dump(data, outfile, indent=4)


if __name__ == '__main__':
    # check output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    # CUDA setup
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ###############################################################################
    # Load data
    ###############################################################################

    # Get Processor
    processor = Processor(args)
    label_list = processor.get_labels()
    num_labels = [len(labels) for labels in label_list]  # number of slot-values in each slot-type

    if args.use_dialog_bert:
        from convlab2.ptm.model.transformers import DialogBertTokenizer, DialogBertModel
        tokenizer = DialogBertTokenizer.from_pretrained(args.bert_model)
    else:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    num_train_steps = None
    accumulation = False

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, accumulation=accumulation)
        dev_examples = processor.get_dev_examples(args.data_dir, accumulation=accumulation)

        ## Training utterances
        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)

        num_train_batches = all_input_ids.size(0)
        num_train_steps = int(
            num_train_batches / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(
            device), all_label_ids.to(device)

        train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        ## Dev utterances
        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)

        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)

        all_input_ids_dev, all_input_len_dev, all_label_ids_dev = \
            all_input_ids_dev.to(device), all_input_len_dev.to(device), all_label_ids_dev.to(device)

        dev_data = TensorDataset(all_input_ids_dev, all_input_len_dev, all_label_ids_dev)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    model = BeliefTracker(args, num_labels, device, tokenizer)
    if args.fp16:
        model.half()
    model.to(device)
    save_configure(args, num_labels, processor.ontology)

    ## Get slot-value embeddings
    label_token_ids, label_len = [], []
    for labels in label_list:
        token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    ## Get domain-slot-type embeddings
    slot_token_ids, slot_len = \
        get_label_embedding(processor.target_slot, args.max_label_length, tokenizer, device)

    ## Initialize slot and value embeddings
    model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)

    # Data parallelize when use multi-gpus
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.learning_rate},
            ]
            return optimizer_grouped_parameters

        if n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model.module)

        t_total = num_train_steps

        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            if args.use_dialog_bert:
                from convlab2.ptm.model.transformers import AdamW, get_linear_schedule_with_warmup
            else:
                from transformers import AdamW, get_linear_schedule_with_warmup
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                correct_bias=False,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_train_steps * args.warmup_proportion,
                num_training_steps=num_train_steps
            )  # PyTorch scheduler
        logger.info(optimizer)

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = None
        best_loss = None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch", ncols=80):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=80)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids = batch

                # Forward
                if n_gpu == 1:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)
                else:
                    loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if isinstance(model, torch.nn.DataParallel):
                        torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            # Perform evaluation on validation dataset
            model.eval()
            dev_loss = 0
            dev_acc = 0
            dev_loss_slot, dev_acc_slot = None, None
            nb_dev_examples, nb_dev_steps = 0, 0

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation", ncols=80)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len, label_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if n_gpu == 1:
                        loss, loss_slot, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)
                    else:
                        loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)

                        # average to multi-gpus
                        loss = loss.mean()
                        acc = acc.mean()
                        acc_slot = acc_slot.mean(0)

                num_valid_turn = torch.sum(label_ids[:, :, 0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn

                if n_gpu == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn

                nb_dev_examples += num_valid_turn

            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples

            if n_gpu == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples

            dev_loss = round(dev_loss, 6)
            if last_update is None or dev_loss < best_loss:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin".format(epoch))
                if args.do_train and dev_loss > 0.50:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_loss
                best_acc = dev_acc

                logger.info("*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" % (
                last_update, best_loss, best_acc))
            else:
                output_model_file = os.path.join(args.output_dir, "pytorch_model_notupdate_{}.bin".format(epoch))
                if args.do_train and dev_loss > 0.55:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)
                logger.info("*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (
                epoch, dev_loss, dev_acc))

            if last_update + args.patience <= epoch:
                break

    ###############################################################################
    # Evaluation
    ###############################################################################
    # Load a trained model that you have fine-tuned
    #     file_list = os.listdir('/mnt/nas6/users/hanting/SUMBT-modified/exp-multiwoz/exp/')
    file_list = os.listdir(args.model_dir)
    file_list = [x for x in file_list if x.startswith('pytorch_model')]

    for fileName in file_list:

        #         output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        #         output_model_file = '/mnt/nas6/users/hanting/SUMBT-modified/exp-multiwoz/exp/'+fileName
        output_model_file = os.path.join(args.model_dir, fileName)

        print('output_model_file:', output_model_file)

        model = BeliefTracker(args, num_labels, device, tokenizer)

        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # in the case that slot and values are different between the training and evaluation
        ptr_model = torch.load(output_model_file)
        if n_gpu == 1:
            state = model.state_dict()
            state.update(ptr_model)
            model.load_state_dict(state)
        else:
            print("Evaluate using only one device!")
            model.module.load_state_dict(ptr_model)

        model.to(device)

        # Evaluation
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

            eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
            all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, args.max_turn_length)
            all_input_ids, all_input_len, all_label_ids = all_input_ids.to(device), all_input_len.to(
                device), all_label_ids.to(device)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            eval_loss_slot, eval_acc_slot = None, None
            nb_eval_steps, nb_eval_examples = 0, 0

            accuracies = {'joint7': 0, 'slot7': 0, 'joint5': 0, 'slot5': 0, 'joint_rest': 0, 'slot_rest': 0,
                          'num_turn': 0, 'num_slot7': 0, 'num_slot5': 0, 'num_slot_rest': 0}

            for input_ids, input_len, label_ids in tqdm(eval_dataloader, desc="Evaluating", ncols=80):
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if n_gpu == 1:
                        loss, loss_slot, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, n_gpu)
                    else:
                        loss, _, acc, acc_slot, pred_slot = model(input_ids, input_len, label_ids, n_gpu)
                        nbatch = label_ids.size(0)
                        nslot = pred_slot.size(3)
                        pred_slot = pred_slot.view(nbatch, -1, nslot)

                accuracies = eval_all_accs(pred_slot, label_ids, accuracies)

                nb_eval_ex = (label_ids[:, :, 0].view(-1) != -1).sum().item()
                nb_eval_examples += nb_eval_ex
                nb_eval_steps += 1

                if n_gpu == 1:
                    eval_loss += loss.item() * nb_eval_ex
                    eval_accuracy += acc.item() * nb_eval_ex
                    if eval_loss_slot is None:
                        eval_loss_slot = [l * nb_eval_ex for l in loss_slot]
                        eval_acc_slot = acc_slot * nb_eval_ex
                    else:
                        for i, l in enumerate(loss_slot):
                            eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                        eval_acc_slot += acc_slot * nb_eval_ex
                else:
                    eval_loss += sum(loss) * nb_eval_ex
                    eval_accuracy += sum(acc) * nb_eval_ex

            eval_loss = eval_loss / nb_eval_examples
            eval_accuracy = eval_accuracy / nb_eval_examples
            if n_gpu == 1:
                eval_acc_slot = eval_acc_slot / nb_eval_examples

            loss = tr_loss / nb_tr_steps if args.do_train else None

            if n_gpu == 1:
                result = {'eval_loss': eval_loss,
                          'eval_accuracy': eval_accuracy,
                          'loss': loss,
                          'eval_loss_slot': '\t'.join([str(val / nb_eval_examples) for val in eval_loss_slot]),
                          'eval_acc_slot': '\t'.join([str((val).item()) for val in eval_acc_slot])
                          }
            else:
                result = {'eval_loss': eval_loss,
                          'eval_accuracy': eval_accuracy,
                          'loss': loss
                          }

            out_file_name = 'eval_results'
            if args.target_slot == 'all':
                out_file_name += '_all'
            output_eval_file = os.path.join(args.output_dir, "%s.txt" % out_file_name)
            if n_gpu == 1:
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

            out_file_name = 'eval_all_accuracies_' + fileName.split('.')[0]
            with open(os.path.join(args.output_dir, "%s.txt" % out_file_name), 'w') as f:
                f.write(
                    'joint acc (7 domain) : slot acc (7 domain) : joint acc (5 domain): slot acc (5 domain): joint restaurant : slot acc restaurant \n')
                f.write('%.5f : %.5f : %.5f : %.5f : %.5f : %.5f \n' % (
                    (accuracies['joint7'] / accuracies['num_turn']).item(),
                    (accuracies['slot7'] / accuracies['num_slot7']).item(),
                    (accuracies['joint5'] / accuracies['num_turn']).item(),
                    (accuracies['slot5'] / accuracies['num_slot5']).item(),
                    (accuracies['joint_rest'] / accuracies['num_turn']).item(),
                    (accuracies['slot_rest'] / accuracies['num_slot_rest']).item()
                ))
