import copy
import os
import logging
import sys
import random
from tqdm import tqdm
import re
import json
from transformers import BertTokenizer
import unicodedata
import string
import argparse
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)


def parent_dir(path, time=1):
    for _ in range(time):
        path = os.path.dirname(path)
    return path


self_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = parent_dir(self_dir, 4)

sys.path.append(root_dir)
from convlab2.util.file_util import read_zipped_json, write_zipped_json
from convlab2.ptm.pretraining.model.tokenization_dialog_bert import DialogBertTokenizer

random.seed(1234)
# slot_neg_ratio = 0.01  now sample two none slot for every domain in every turn

# datasets used in pretraining
dataset_names = [
    # 'schema',
    'multiwoz21',
    # 'woz',
    # 'm2m'
]

data_dir = os.path.join(root_dir, 'data_ptm')
processed_data_dir = os.path.join(self_dir, 'processed_data21')
SPECIAL_VALUE_LIST = ['none', 'dontcare']

parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_utts_to_keep',
    type=int,
    default=-1
)
# parser.add_argument(
#     '--max_utts_to_keep_cat',
#     type=int,
#     default=-1
# )
# parser.add_argument(
#     '--max_utts_to_keep_noncat',
#     type=int,
#     default=-1
# )
parser.add_argument(
    '--model',
    type=str,
    choices=['bert', 'dialogbert']
)
parser.add_argument(
    '--reverse',
    action='store_true',
)


def generate_span_data(utterance, span, tokenizer):
    """
    generate span data for a sentence with BERT tokenizer
    :param utterance: str,
    :param span: [start, end]
    :param tokenizer: a bert tokenizer by default
    :return: tokenized utterance and start/end of tokenized sequence
    """
    # char-span to word-span
    ori_tokens, bert_tokens, char2berttoken, bert2oritoken, continue_berttokenidx = _tokenize(utterance, tokenizer)
    start, end = span
    assert start in char2berttoken and end - 1 in char2berttoken, print(utterance, bert_tokens, char2berttoken)
    tok_start, tok_end = char2berttoken[start], char2berttoken[end - 1]
    Xmask = [1] * len(bert_tokens)
    for idx in continue_berttokenidx:
        Xmask[idx] = 0
    return bert_tokens, [tok_start, tok_end], Xmask, ori_tokens, bert2oritoken


def _tokenize(utterance, tokenizer):
    """
    Tokenize the utterance using word-piece tokenization used by BERT.
    :param utterance: A string containing the utterance to be tokenized.
    :param tokenizer: BERT tokenizer
    :return:
        ori_tokens: from _naive_tokenize
        bert_tokens: from BERT tokenizer
        char2bert_token: map char idx to bert_token idx
        bert_token2ori_token: map bert_token idx to ori_token idx
        continued_bert_token_idxs: list of idx that corresponding bert_token is continued subword startswith "##"
    """
    utterance = shave_marks_latin(utterance)
    utterance = convert_to_unicode(utterance)
    # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
    # direct concatenation of all the tokens in the sequence will be the
    # original string.
    ori_tokens = _naive_tokenize(utterance)
    char2bert_token = {}
    char_index = 0
    bert_tokens = []
    bert_token2ori_token = {}
    continued_bert_token_idxs = []  # for "##xxx" sub-words split by wordpiece tokenizer
    for j, token in enumerate(ori_tokens):
        if token.strip():
            subwords = tokenizer.tokenize(token)
            # Store the alignment for the index of character to corresponding token
            char_in_token_idx = 0
            # assert subwords != ['[UNK]']
            if subwords != ['[UNK]']:
                for i, sw in enumerate(subwords):
                    if sw.startswith('##'):
                        sw = sw[2:]
                        continued_bert_token_idxs.append(len(bert_tokens) + i)
                    bert_token2ori_token[len(bert_tokens) + i] = j
                    for c in sw:
                        assert char_in_token_idx < len(token), print(len(token), char_in_token_idx, utterance, [token],
                                                                     subwords, [c], [sw], i)
                        # assert (c == token[char_in_token_idx].lower() or subwords == ['[UNK]']), print(utterance, [token], subwords, c, token[char_in_token_idx], sw)
                        char2bert_token[char_index + char_in_token_idx] = len(bert_tokens) + i
                        char_in_token_idx += 1
            else:
                bert_token2ori_token[len(bert_tokens)] = j
                for char_in_token_idx in range(len(token)):
                    char2bert_token[char_index + char_in_token_idx] = len(bert_tokens)
                print('[UNK]:', [token], utterance)

            bert_tokens.extend(subwords)
        char_index += len(token)
    # print(ori_tokens, bert_tokens, char2bert_token, bert_token2ori_token, continued_bert_token_idxs)
    return ori_tokens, bert_tokens, char2bert_token, bert_token2ori_token, continued_bert_token_idxs


def _naive_tokenize(s):
    """Tokenize a string, separating words, spaces and punctuations."""
    # Spaces and punctuation marks are all retained, i.e. direct concatenation
    # of all the tokens in the sequence will be the original string.
    seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
    return seq_tok


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def shave_marks_latin(txt):
    """把拉丁基字符中所有的变音符号删除 https://hellowac.github.io/programing%20teach/2017/05/10/fluentpython04.html"""
    norm_txt = unicodedata.normalize('NFD', txt)  # 分解成基字符和组合字符。
    latin_base = True
    keepers = []
    for c in norm_txt:
        if unicodedata.combining(c) and latin_base:  # 基字符为拉丁字母时，跳过组合字符。
            continue  # 忽略拉丁基字符上的变音符号
        keepers.append(c)  # 否则，保存当前字符。
        # 如果不是组合字符，那就是新的基字符
        if not unicodedata.combining(c):  # 检测新的基字符，判断是不是拉丁字母。
            latin_base = c in string.ascii_letters  # ascii 是 拉丁字符罗？ 为false时，即为标记.
    shaved = ''.join(keepers)
    return unicodedata.normalize('NFC', shaved)  # 重组所有字符


def preprocess(
        dataset_names,
        args,
        out_train_filename='_bio_train.json',
        out_val_filename='_bio_val.json',
        out_test_filename='_bio_test.json',
):
    """
    preprocess datasets for span prediction, containing
    :param dataset_names: List, elements of dataset_names should match names of dataset directories in data/
    :return:
    todo: returns could start with sys
    output form: [usr_t  sys_t  usr_t-1  sys_t-1 ... usr_t-history_length, sys_t-history_length], padded with ' '
    """
    out_train_filename = '_{}'.format(args.model) + out_train_filename
    out_val_filename = '_{}'.format(args.model) + out_val_filename
    out_test_filename = '_{}'.format(args.model) + out_test_filename
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/data/zhuqi/pre-trained-models/dialogbert/mlm/mlm_wwm_120k_0831_bert')

    special_token_dict = {
        'additional_special_tokens': ['[USR]', '[SYS]', '[DOMAIN]', '[SLOT]', '[VALUE]',
                                      '[STATE]', '[DIALOG_ACT]',
                                      '[NEXT_SENTENCE]']}
    tokenizer.add_special_tokens(special_token_dict)

    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)

    for _ds in dataset_names:
        # if os.path.exists(os.path.join(processed_data_dir, _ds + 'cat' + out_train_filename)):
        #     continue
        logging.info('preprocessing {} dataset'.format(_ds))
        _dataset_dir = os.path.join(data_dir, _ds)
        _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
        if not os.path.exists(_data_zipfile):
            raise FileNotFoundError('Data Not Found!', _data_zipfile)
        else:
            _dialogs = read_zipped_json(_data_zipfile, 'data.json')
            _ontology = json.load(open(os.path.join(_dataset_dir, 'ontology.json')))

        # if _ds in ['multiwoz25', 'schema']:
        #     # only training set
        #     _dialogs = [d for d in _dialogs if d['data_split'] == 'train']

        all_labels = []
        for _domain in _ontology['state']:
            for _slot in _ontology['state'][_domain]:
                all_labels.append('B-' + _domain + '-' + _slot)
                all_labels.append('I-' + _domain + '-' + _slot)
        all_labels.append('O')
        label_to_id = {l: ind for ind, l in enumerate(all_labels)}
        label_to_save = {'id_to_label': all_labels, 'label_to_id': label_to_id}

        categorical_ontology = {}
        categorical_slot_list = []
        noncategorical_ontology = {}
        noncategorical_slot_list = []
        for _domain in _ontology['state']:
            for _slot in _ontology['state'][_domain]:
                if _ontology['state'][_domain][_slot]['is_categorical']:
                    if not _domain in categorical_ontology:
                        categorical_ontology[_domain] = {}
                    categorical_ontology[_domain][_slot] = SPECIAL_VALUE_LIST + _ontology['state'][_domain][_slot][
                        'possible_values']

                    categorical_slot_list.append((_domain, _slot))
                else:
                    noncategorical_slot_list.append((_domain, _slot))

        bio_train_data = []
        bio_val_data = []
        bio_test_data = []
        random.seed(42)
        num_train_samples, num_val_samples = 0, 0

        slot_delete_cnt = 0
        for dial_id, _d in enumerate(tqdm(_dialogs, desc='processing {}'.format(_ds))):
            history = []  # token_ids List[List[int]]
            char2tokens = []

            _turns = _d['turns']

            if len(_turns) > 60:
                # exceed max turn_id limit = (30)
                print('cut turn', len(_turns))
                _turns = _turns[:59]

            for turn_id, _turn in enumerate(_turns):
                _utterance = _turn['utterance']

                ori_tokens, bert_tokens, char2bert_token, bert_token2ori_token, continued_bert_token_idxs = _tokenize(
                    _utterance, tokenizer)

                history.append(bert_tokens)
                char2tokens.append(char2bert_token)

                if _turn['speaker'] == 'user':
                    _state_update = _turn['state_update']

                    num_utts_to_keep = len(history)
                    if args.max_utts_to_keep > 0:
                        num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep)

                    if args.model == 'dialogbert':
                        encoded_dict = tokenizer.prepare_input_seq(history[-num_utts_to_keep:], return_lengths=True)
                        encoded_dict['bio_tags'] = ['O' for _ in encoded_dict['input_ids']]

                        if num_utts_to_keep > 0 and encoded_dict['length'] > 512:
                            print('utterances too long, can\'t keep {} utterances as context'.format(num_utts_to_keep))
                            continue

                        encoded_dict['categorical'] = [0 for _ in categorical_slot_list]

                        for sv in _state_update['categorical']:
                            domain, slot, value = sv['domain'], sv['slot'], sv['value']
                            if value == '':
                                value = 'none'
                            if value not in categorical_ontology[domain][slot]:
                                print('categorical value not in ontology')
                                print(value, categorical_ontology[domain][slot])
                            try:
                                value_idx = categorical_ontology[domain][slot].index(value)
                            except:
                                print('{} not in categorical ontology'.format(value))
                                continue
                            slot_idx = categorical_slot_list.index((domain, slot))
                            encoded_dict['categorical'][slot_idx] = value_idx

                        encoded_dict['non-categorical'] = [0 for _ in noncategorical_slot_list]
                        for sv in _state_update['non-categorical']:
                            domain, slot, value = sv['domain'], sv['slot'], sv['value']
                            value = sv['value'] if 'fixed_value' not in sv else sv['fixed_value']
                            annotated = 'start' in sv

                            if annotated:
                                start, end, utt_idx = sv['start'], sv['end'], sv['utt_idx']
                                _utt = history[utt_idx]
                                _char2bert_token = char2tokens[utt_idx]
                                token_start_idx = _char2bert_token[start]
                                token_end_idx = _char2bert_token[end - 1]

                                if value.replace(' ', '') != tokenizer.convert_tokens_to_string(
                                        _utt[token_start_idx: token_end_idx + 1]).replace(' ', ''):
                                    if value.replace(' ', '') == 'centre' and tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]).replace(' ', '') == 'north':
                                        # multiwoz25 annotation error
                                        continue
                                    elif value.replace(' ', '') == 'gastropub' and tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]).replace(' ', '') == 'gastropubs':
                                        # woz annotation error
                                        pass
                                    elif value.replace(' ', '') == 'afghan' and tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]).replace(' ', '') == 'afghanistan':
                                        # woz annotation error
                                        pass
                                    else:
                                        print(value, tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]))

                                if len(history[utt_idx:]) > num_utts_to_keep:
                                    print('non categorical value requires keeping more utterances!')
                                    continue
                                _try_encoded_dict = tokenizer.prepare_input_seq(history[utt_idx + 1:],
                                                                                return_lengths=True)

                                if num_utts_to_keep > 0 and encoded_dict['length'] > 512:
                                    print('utterances too long, can\'t keep {} utterances to ensure span exists'.format(
                                        num_utts_to_keep))
                                    continue
                                len_prefix = len(_try_encoded_dict['input_ids'])
                                span_start, span_end = token_start_idx + len_prefix + 1, token_end_idx + len_prefix + 1
                                encoded_dict['bio_tags'][span_start] = 'B-' + domain + '-' + slot
                                for ind in range(span_start + 1, span_end + 1):
                                    encoded_dict['bio_tags'][ind] = 'I-' + domain + '-' + slot

                                if not value.replace(' ', '') == tokenizer.decode(
                                        encoded_dict['input_ids'][span_start: span_end + 1]).replace(' ', ''):
                                    print(value, tokenizer.decode(encoded_dict['input_ids'][span_start: span_end + 1]))

                            else:
                                if value in SPECIAL_VALUE_LIST:
                                    value_idx = SPECIAL_VALUE_LIST.index(value)
                                    slot_idx = noncategorical_slot_list.index((domain, slot))
                                    encoded_dict['non-categorical'][slot_idx] = value_idx

                        encoded_dict['bio_labels'] = [label_to_id[l] for l in encoded_dict['bio_tags']]
                        encoded_dict['turn_id'] = turn_id
                        encoded_dict['dial_id'] = dial_id
                        encoded_dict['golden_state_update'] = _state_update

                        assert len(encoded_dict['input_ids']) == len(encoded_dict['bio_labels'])
                        if _d['data_split'] == 'train':
                            bio_train_data.append(copy.deepcopy(encoded_dict))
                        elif _d['data_split'] == 'val':
                            bio_val_data.append(copy.deepcopy(encoded_dict))
                        elif _d['data_split'] == 'test':
                            bio_test_data.append(copy.deepcopy(encoded_dict))

                    elif args.model == 'bert':
                        if args.reverse:
                            _start_idx = len(history) - 1
                            _end_idx = len(history) - num_utts_to_keep - 1
                            _step = -1
                        else:
                            _start_idx = len(history) - num_utts_to_keep
                            _end_idx = len(history)
                            _step = 1
                        encoded_dict = {'input_ids': [], 'bio_tags': [], 'bio_labels': [], 'input_seq': []}
                        utt_idx_to_seq_position = {}
                        encoded_dict['input_seq'].append('[CLS]')
                        for _uidx in range(_start_idx, _end_idx, _step):
                            utt_idx_to_seq_position[_uidx] = len(encoded_dict['input_seq'])
                            encoded_dict['input_seq'].extend(history[_uidx])
                            encoded_dict['input_seq'].append('[SEP]')

                        encoded_dict['input_ids'] = tokenizer.convert_tokens_to_ids(encoded_dict['input_seq'])
                        if len(encoded_dict['input_ids']) > 512:
                            print('utterances too long, can\'t keep {} utterances as context'.format(num_utts_to_keep))
                            continue

                        encoded_dict['bio_tags'] = ['O' for _ in encoded_dict['input_ids']]

                        encoded_dict['categorical'] = [0 for _ in categorical_slot_list]
                        for sv in _state_update['categorical']:
                            domain, slot, value = sv['domain'], sv['slot'], sv['value']
                            if value == '':
                                value = 'none'
                            if value not in categorical_ontology[domain][slot]:
                                print('categorical value not in ontology')
                                print(value, categorical_ontology[domain][slot])
                            try:
                                value_idx = categorical_ontology[domain][slot].index(value)
                            except:
                                print('{} not in categorical ontology'.format(value))
                                continue
                            slot_idx = categorical_slot_list.index((domain, slot))
                            encoded_dict['categorical'][slot_idx] = value_idx

                        encoded_dict['non-categorical'] = [0 for _ in noncategorical_slot_list]
                        for sv in _state_update['non-categorical']:
                            domain, slot, value = sv['domain'], sv['slot'], sv['value']
                            value = sv['value'] if 'fixed_value' not in sv else sv['fixed_value']
                            annotated = 'start' in sv

                            if annotated:
                                start, end, utt_idx = sv['start'], sv['end'], sv['utt_idx']
                                _utt = history[utt_idx]
                                _char2bert_token = char2tokens[utt_idx]
                                token_start_idx = _char2bert_token[start]
                                token_end_idx = _char2bert_token[end - 1]

                                if value.replace(' ', '') != tokenizer.convert_tokens_to_string(
                                        _utt[token_start_idx: token_end_idx + 1]).replace(' ', ''):
                                    if value.replace(' ', '') == 'centre' and tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]).replace(' ', '') == 'north':
                                        # multiwoz25 annotation error
                                        continue
                                    elif value.replace(' ', '') == 'gastropub' and tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]).replace(' ', '') == 'gastropubs':
                                        # woz annotation error
                                        pass
                                    elif value.replace(' ', '') == 'afghan' and tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]).replace(' ', '') == 'afghanistan':
                                        # woz annotation error
                                        pass
                                    else:
                                        print(value, tokenizer.convert_tokens_to_string(
                                            _utt[token_start_idx: token_end_idx + 1]))

                                if len(history[utt_idx:]) > num_utts_to_keep or utt_idx not in utt_idx_to_seq_position:
                                    print('non categorical value requires keeping more utterances!')
                                    continue

                                len_prefix = utt_idx_to_seq_position[utt_idx]
                                span_start, span_end = token_start_idx + len_prefix, token_end_idx + len_prefix
                                encoded_dict['bio_tags'][span_start] = 'B-' + domain + '-' + slot
                                for ind in range(span_start + 1, span_end + 1):
                                    encoded_dict['bio_tags'][ind] = 'I-' + domain + '-' + slot

                                if value.replace(' ', '') != tokenizer.decode(
                                        encoded_dict['input_ids'][span_start: span_end + 1]).replace(' ', ''):
                                    print('found wrong annotation')
                                    print(value, tokenizer.decode(encoded_dict['input_ids'][span_start: span_end + 1]))
                                    continue

                            else:
                                if value in SPECIAL_VALUE_LIST:
                                    value_idx = SPECIAL_VALUE_LIST.index(value)
                                    slot_idx = noncategorical_slot_list.index((domain, slot))
                                    encoded_dict['non-categorical'][slot_idx] = value_idx

                        encoded_dict['turn_id'] = turn_id
                        encoded_dict['dial_id'] = dial_id
                        encoded_dict['bio_labels'] = [label_to_id[l] for l in encoded_dict['bio_tags']]
                        encoded_dict['golden_state_update'] = _state_update

                        if _d['data_split'] == 'train':
                            bio_train_data.append(copy.deepcopy(encoded_dict))
                        elif _d['data_split'] == 'val':
                            bio_val_data.append(copy.deepcopy(encoded_dict))
                        elif _d['data_split'] == 'test':
                            bio_test_data.append(copy.deepcopy(encoded_dict))
        print('slot delete count: {}'.format(slot_delete_cnt))
        json.dump(bio_train_data,
                  open(os.path.join(processed_data_dir, _ds + out_train_filename), 'w'), indent=2)
        json.dump(bio_val_data, open(os.path.join(processed_data_dir, _ds + out_val_filename), 'w'),
                  indent=2)
        json.dump(bio_test_data, open(os.path.join(processed_data_dir, _ds + out_test_filename), 'w'),
                  indent=2)
        json.dump(label_to_save,
                  open(os.path.join(processed_data_dir, _ds + '_' + args.model + '_' + 'labels_map.json'), 'w'),
                  indent=2)
        json.dump({
            'categorical_slot_list': categorical_slot_list,
            'categorical_slot_ontology': categorical_ontology,
            'non-categorical_slot_list': noncategorical_slot_list
        },
            open(os.path.join(processed_data_dir, _ds + '_' + args.model + '_' + 'slot_ontology.json'), 'w'),
            indent=4
        )


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess(dataset_names, args=args)
