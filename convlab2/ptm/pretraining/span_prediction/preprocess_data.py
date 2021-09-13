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
dev_ratio = 0.01  # ratio of data used to evaluate
# slot_neg_ratio = 0.01  now sample two none slot for every domain in every turn

# datasets used in pretraining
# dev/test set of multiwoz and schema are excluded automatically
dataset_names = [
    # 'schema',
    'multiwoz25',
    # 'woz',
    # 'm2m'
]
dataset_only_span = [
    'taskmaster',
]

data_dir = os.path.join(root_dir, 'data_ptm')
processed_data_dir = os.path.join(self_dir, 'processed_data')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_utts_to_keep',
    type=int,
    default=-1
)
parser.add_argument(
    '--max_utts_to_keep_cat',
    type=int,
    default=-1
)
parser.add_argument(
    '--max_utts_to_keep_noncat',
    type=int,
    default=-1
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


def value_span_strip(span):
    """strip value for empty char"""
    value = span['value']
    start = span['start']
    end = span['end']
    start_shift, end_shift = 0, 0
    # print(span)
    for char in value:
        if char in [' ', u'\u200b']:
            start_shift += 1
        else:
            value = value[start_shift:]
            start += start_shift
            break
    for char in value[::-1]:
        if char in [' ', u'\u200b']:
            end_shift += 1
        else:
            if end_shift:
                value = value[:-end_shift]
                end -= end_shift
            break
    span['value'] = value
    span['start'] = start
    span['end'] = end


def preprocess(
        dataset_names,
        args,
        out_train_filename='_stateupdate_data_train.json',
        out_dev_filename='_stateupdate_data_dev.json'):
    """
    preprocess datasets for span prediction, containing
    :param dataset_names: List, elements of dataset_names should match names of dataset directories in data/
    :return:
    todo: returns could start with sys
    output form: [usr_t  sys_t  usr_t-1  sys_t-1 ... usr_t-history_length, sys_t-history_length], padded with ' '
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = DialogBertTokenizer.from_pretrained(
        '/home/libing/Convlab2-Pretraining/convlab2/ptm/model/dialog_mlm/output_all_fulldialog_7.29')

    special_token_dict = {
        'additional_special_tokens': ['[USR]', '[SYS]', '[DOMAIN]', '[SLOT]', '[VALUE]',
                                      '[STATE]', '[DIALOG_ACT]',
                                      '[NEXT_SENTENCE]']}
    tokenizer.add_special_tokens(special_token_dict)

    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)

    for _ds in dataset_names:
        if os.path.exists(os.path.join(processed_data_dir, _ds + 'cat' + out_train_filename)):
            continue
        logging.info('preprocessing {} dataset'.format(_ds))
        _dataset_dir = os.path.join(data_dir, _ds)
        _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
        if not os.path.exists(_data_zipfile):
            raise FileNotFoundError('Data Not Found!', _data_zipfile)
        else:
            _dialogs = read_zipped_json(_data_zipfile, 'data.json')
            _ontology = json.load(open(os.path.join(_dataset_dir, 'ontology.json')))

        if _ds in ['multiwoz25', 'schema']:
            # only training set
            _dialogs = [d for d in _dialogs if d['data_split'] == 'train']

        cat_train_data_json = []
        cat_dev_data_json = []
        noncat_train_data_json = []
        noncat_dev_data_json = []
        random.seed(42)
        num_train_samples, num_dev_samples = 0, 0

        slot_delete_cnt = 0
        for dial_id, _d in enumerate(tqdm(_dialogs[:100], desc='processing {}'.format(_ds))):
            history = []  # token_ids List[List[int]]
            char2tokens = []

            _turns = _d['turns']
            split = 'train'
            if random.random() < dev_ratio:
                split = 'dev'

            if len(_turns) > 60:
                # exceed max turn_id limit = (30)
                print('cut turn', len(_turns))
                _turns = _turns[:59]

            for _turn in _turns:
                _utterance = _turn['utterance']

                ori_tokens, bert_tokens, char2bert_token, bert_token2ori_token, continued_bert_token_idxs = _tokenize(
                    _utterance, tokenizer)

                history.append(bert_tokens)
                char2tokens.append(char2bert_token)

                # for negtive sampling
                active_domains = []
                active_slots = []

                if _turn['speaker'] == 'user':
                    _state_update = _turn['state_update']

                    for sv in _state_update['categorical']:
                        domain, slot, value = sv['domain'], sv['slot'], sv['value']
                        if domain not in active_domains:
                            active_domains.append(domain)
                        if slot not in active_slots:
                            active_slots.append(slot)
                        assert value in _ontology['domains'][domain]['slots'][slot][
                            'possible_values'] or value == 'dontcare' or value == ''
                        domain_desc = _ontology['domains'][domain]['description']
                        slot_desc = _ontology['domains'][domain]['slots'][slot]['description']
                        all_values = _ontology['domains'][domain]['slots'][slot]['possible_values']

                        num_utts_to_keep = len(history)
                        if args.max_utts_to_keep > 0:
                            num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep)
                        if args.max_utts_to_keep_cat > 0:
                            num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep_cat)

                        encoded_dict = tokenizer.prepare_input_seq(history[-num_utts_to_keep:],
                                                                   prefix_dict={'[DOMAIN]': domain_desc,
                                                                                '[SLOT]': slot_desc,
                                                                                '[VALUE]': all_values},
                                                                   return_lengths=True)
                        if num_utts_to_keep > 0 and encoded_dict['length'] > 512:
                            print('utterances too long, can\'t keep {} utterances as context'.format(num_utts_to_keep))
                            continue
                        encoded_dict['slot_token_idx'] = encoded_dict['input_ids'].index(tokenizer.convert_tokens_to_ids(['[SLOT]'])[0])
                        if value == 'dontcare':
                            encoded_dict['cls_label'] = value
                            encoded_dict['value_label'] = -100
                        elif value == '':
                            # process slot delete
                            encoded_dict['cls_label'] = 'delete'
                            slot_delete_cnt += 1
                            encoded_dict['value_label'] = -100
                        else:
                            encoded_dict['cls_label'] = 'has_value'
                            value_label = all_values.index(value)
                            encoded_dict['value_label'] = value_label

                        value_token_mask = tokenizer.get_tokens_mask(encoded_dict['input_ids'],
                                                                         ['[VALUE]'])  # 0 for all '[VALUE]' tokens
                        encoded_dict['value_token_mask'] = [1 - m for m in value_token_mask]
                        assert value_label < sum(encoded_dict['value_token_mask']), print(encoded_dict)

                        if split == 'train':
                            cat_train_data_json.append(copy.deepcopy(encoded_dict))
                        else:
                            cat_dev_data_json.append(copy.deepcopy(encoded_dict))

                    for sv in _state_update['non-categorical']:
                        domain, slot, value = sv['domain'], sv['slot'], sv['value']
                        annotated = 'start' in sv

                        if domain not in active_domains:
                            active_domains.append(domain)
                        domain_desc = _ontology['domains'][domain]['description']
                        slot_desc = _ontology['domains'][domain]['slots'][slot]['description']

                        num_utts_to_keep = len(history)
                        if args.max_utts_to_keep > 0:
                            num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep)
                        if args.max_utts_to_keep_noncat > 0:
                            num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep_noncat)

                        encoded_dict = tokenizer.prepare_input_seq(history[-num_utts_to_keep:],
                                                                   prefix_dict={'[DOMAIN]': domain_desc,
                                                                                '[SLOT]': slot_desc,
                                                                                },
                                                                   return_lengths=True)
                        if num_utts_to_keep > 0 and encoded_dict['length'] > 512:
                            print('utterances too long, can\'t keep {} utterances as context'.format(num_utts_to_keep))
                            continue

                        encoded_dict['slot_token_idx'] = encoded_dict['input_ids'].index(tokenizer.convert_tokens_to_ids(['[SLOT]'])[0])

                        if value == 'dontcare':
                            encoded_dict['cls_label'] = value
                            encoded_dict['start'] = -100
                            encoded_dict['end'] = -100
                        elif value == '':
                            # process slot delete
                            encoded_dict['cls_label'] = 'delete'
                            slot_delete_cnt += 1
                            encoded_dict['start'] = -100
                            encoded_dict['end'] = -100
                        else:
                            encoded_dict['cls_label'] = 'has_value'
                            encoded_dict['start'] = -100
                            encoded_dict['end'] = -100

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
                                                                                prefix_dict={'[DOMAIN]': domain_desc,
                                                                                             '[SLOT]': slot_desc, },
                                                                                return_lengths=True
                                                                                )

                                if num_utts_to_keep > 0 and encoded_dict['length'] > 512:
                                    print('utterances too long, can\'t keep {} utterances to ensure span exists'.format(
                                        num_utts_to_keep))
                                    continue
                                len_prefix = len(_try_encoded_dict['input_ids'])
                                span_start, span_end = token_start_idx + len_prefix + 1, token_end_idx + len_prefix + 1
                                encoded_dict['start'] = span_start
                                encoded_dict['end'] = span_end
                                assert value.replace(' ', '') == tokenizer.decode(
                                    encoded_dict['input_ids'][span_start: span_end + 1]).replace(' ', ''), print(value,
                                                                                                                 tokenizer.decode(
                                                                                                                     encoded_dict[
                                                                                                                         'input_ids'][
                                                                                                                     span_start: span_end + 1]))

                        if split == 'train':
                            noncat_train_data_json.append(copy.deepcopy(encoded_dict))
                        else:
                            noncat_dev_data_json.append(copy.deepcopy(encoded_dict))

                    # do negtive sample
                    for domain in active_domains:
                        all_slots = _ontology['domains'][domain]['slots'].keys()
                        all_slots = [s for s in all_slots if s not in active_slots]
                        neg_slot = random.choice(all_slots)
                        domain_desc = _ontology['domains'][domain]['description']
                        slot_desc = _ontology['domains'][domain]['slots'][neg_slot]['description']
                        all_values = _ontology['domains'][domain]['slots'][neg_slot]['possible_values']
                        if _ontology['domains'][domain]['slots'][neg_slot]['is_categorical']:

                            num_utts_to_keep = len(history)
                            if args.max_utts_to_keep > 0:
                                num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep)
                            if args.max_utts_to_keep_cat > 0:
                                num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep_cat)

                            encoded_dict = tokenizer.prepare_input_seq(history[-num_utts_to_keep:],
                                                                       prefix_dict={'[DOMAIN]': domain_desc,
                                                                                    '[SLOT]': slot_desc,
                                                                                    '[VALUE]': all_values},
                                                                       return_lengths=True)
                            encoded_dict['slot_token_idx'] = encoded_dict['input_ids'].index(
                                tokenizer.convert_tokens_to_ids(['[SLOT]'])[0])

                            if num_utts_to_keep > 0 and encoded_dict['length'] > 512:
                                print('utterances too long, can\'t keep {} utterances as context in negative sampling'.format(
                                    num_utts_to_keep))
                                continue
                            encoded_dict['cls_label'] = 'none'
                            value_token_mask = tokenizer.get_tokens_mask(encoded_dict['input_ids'],
                                                                         ['[VALUE]'])  # 0 for all '[VALUE]' tokens
                            encoded_dict['value_token_mask'] = [1 - m for m in value_token_mask]
                            encoded_dict['value_label'] = -100
                            if split == 'train':
                                cat_train_data_json.append(copy.deepcopy(encoded_dict))
                            else:
                                cat_dev_data_json.append(copy.deepcopy(encoded_dict))

                        else:
                            num_utts_to_keep = len(history)
                            if args.max_utts_to_keep > 0:
                                num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep)
                            if args.max_utts_to_keep_noncat > 0:
                                num_utts_to_keep = min(num_utts_to_keep, args.max_utts_to_keep_noncat)

                            encoded_dict = tokenizer.prepare_input_seq(history[-num_utts_to_keep:],
                                                                       prefix_dict={'[DOMAIN]': domain_desc,
                                                                                    '[SLOT]': slot_desc,
                                                                                    },
                                                                       return_lengths=True)
                            encoded_dict['slot_token_idx'] = encoded_dict['input_ids'].index(
                                tokenizer.convert_tokens_to_ids(['[SLOT]'])[0])

                            if num_utts_to_keep > 0 and encoded_dict['length'] > 512:
                                print('utterances too long, can\'t keep {} utterances as context in negtive sampling'.format(
                                    num_utts_to_keep))
                                continue
                            encoded_dict['cls_label'] = 'none'
                            encoded_dict['start'] = -100
                            encoded_dict['end'] = -100
                            if split == 'train':
                                noncat_train_data_json.append(copy.deepcopy(encoded_dict))
                            else:
                                noncat_dev_data_json.append(copy.deepcopy(encoded_dict))

        print('slot delete count: {}'.format(slot_delete_cnt))
        json.dump(cat_train_data_json, open(os.path.join(processed_data_dir, _ds + 'cat' + out_train_filename), 'w'),
                  indent=2)
        json.dump(cat_dev_data_json, open(os.path.join(processed_data_dir, _ds + 'cat' + out_dev_filename), 'w'),
                  indent=2)
        json.dump(noncat_train_data_json,
                  open(os.path.join(processed_data_dir, _ds + 'noncat' + out_train_filename), 'w'), indent=2)
        json.dump(noncat_dev_data_json, open(os.path.join(processed_data_dir, _ds + 'noncat' + out_dev_filename), 'w'),
                  indent=2)

        logging.info(
            '{} dataset preprocessed, written {} cat train examples, {} cat dev examples, {} noncat train examples, {} noncat dev examples'.format(
                _ds, len(cat_train_data_json), len(cat_dev_data_json), len(noncat_train_data_json),
                len(noncat_dev_data_json)))


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess(dataset_names, args=args)
