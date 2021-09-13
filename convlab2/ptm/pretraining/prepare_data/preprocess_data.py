import os
import logging
import sys
import random
from tqdm import tqdm
import re
import json
from transformers import BertTokenizer
from convlab2.util.file_util import read_zipped_json, write_zipped_json
import unicodedata
import string
from pprint import pprint
from collections import defaultdict
import numpy as np
logging.basicConfig(level=logging.INFO)


def parent_dir(path, time=1):
    for _ in range(time):
        path = os.path.dirname(path)
    return path

self_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = parent_dir(self_dir, 4)

sys.path.append(root_dir)
print(sys.path[-1])


random.seed(1234)
dev_ratio = 0.05  # ratio of data used to evaluate

# datasets used in pretraining
# dev/test set of multiwoz and schema are excluded automatically
dataset_names = [
    # 'mitmovie',
    # 'mitrestaurant',
    # 'facebook',
    'camrest',
    'woz',
    'kvret',
    'dstc2',
    'frames',
    'm2m',
    'mdc',
    'multiwoz21',
    # 'multiwoz25',
    'schema',
    'metalwoz',
    'taskmaster',
    'oos',
    'hwu',
    'clinc',
    'banking',
    'restaurant8k',
    'top'
]

data_dir = os.path.join(root_dir, 'data_ptm')
processed_data_dir = os.path.join(self_dir, 'full_dialog_mlm')


def example(utterance, spans):
    """example for tagging and recovering values from BIO tagging"""
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('/home/guyuxian/pretrain-models/bert-base-uncased/')
    ori_tokens, bert_tokens, char2bert_token, bert_token2ori_token, continued_bert_token_idxs = _tokenize(utterance,
                                                                                                          tokenizer)
    print('ori_tokens', ori_tokens)
    print('bert_tokens', bert_tokens)
    print('continued_bert_token_idxs', continued_bert_token_idxs)
    print()
    bert_tokens, tags, Xmasked, ori_tokens, bert_token2ori_token, char2bert_token = generate_bio_tag(utterance,
                                                                                                     spans,
                                                                                                     tokenizer)
    print('tags', tags)
    print('Xmasked', Xmasked)
    print()
    values = extract_value_from_bio(ori_tokens, tags, bert_token2ori_token)
    print('values', values)


def extract_value_from_bio(ori_tokens, tags, bert_token2ori_token):
    """
    Extract value indicated by BIO tags
    :param ori_tokens: ori_token from generate_bio_tag function
    :param tags: tags for bert_tokens from generate_bio_tag function
    :param bert_token2ori_token: index mapping from bert_tokens to ori_token. from generate_bio_tag function
    :return: list of extracted values
    """
    ori_tags = ['O'] * len(ori_tokens)
    for i, tag in enumerate(tags):
        if tag[0] in ['B', 'I']:
            ori_tags[bert_token2ori_token[i]] = tag
    print(ori_tags)
    values = []
    i = 0
    while i < len(ori_tags):
        if not ori_tokens[i].strip():
            i += 1
            continue
        tag = ori_tags[i]
        if tag.startswith('B'):
            value = ori_tokens[i]
            j = i + 1
            while j < len(ori_tags):
                if ori_tags[j].startswith('I') or not ori_tokens[j].strip():
                    value += ori_tokens[j]
                    i += 1
                    j += 1
                elif ori_tags[j] == 'X':
                    i += 1
                    j += 1
                else:
                    break
            values.append(value)
        i += 1
    return values


def generate_bio_tag(utterance, spans, tokenizer):
    """
    Generate BIO tags for a sentence with BERT tokenizer
    :param utterance: str,
    :param spans: list of (start, end) character-level span, utterance[start:end] gives the content
    :param tokenizer: a bert tokenizer by default
    :return:
    bert_tokens: tokenized utterance
    tags: bio tag list
    Xmask: continued_subword_mask
    ori_tokens: ori_tokens from _naive_tokenize
    bert_token2ori_token: index mapping from bert_token to ori_token
    """
    # char-span to word-span
    ori_tokens, bert_tokens, char2bert_token, bert_token2ori_token, continued_bert_token_idxs = _tokenize(
        utterance, tokenizer)
    tags = ['O'] * len(bert_tokens)
    for start, end in spans:
        assert start in char2bert_token and end - 1 in char2bert_token, print(utterance, bert_tokens, char2bert_token, start, end, spans)
        tok_start, tok_end = char2bert_token[start], char2bert_token[end - 1]
        # print(start, end)
        # print(char2bert_token)
        # print(tok_start, tok_end)
        tags[tok_start] = 'B'
        for tok_id in range(tok_start + 1, tok_end + 1):
            tags[tok_id] = 'I'
    Xmask = [1] * len(bert_tokens)
    # for idx in continued_bert_token_idxs:
    #     Xmask[idx] = 0
    #     tags[idx] = 'X'
    assert len(bert_tokens) == len(tags)
    return bert_tokens, tags, Xmask, ori_tokens, bert_token2ori_token, char2bert_token


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
                        assert char_in_token_idx < len(token), print(len(token), char_in_token_idx, utterance, [token], subwords, [c], [sw], i)
                        # assert (c == token[char_in_token_idx].lower() or subwords == ['[UNK]']), print(utterance, [token], subwords, c, token[char_in_token_idx], sw)
                        char2bert_token[char_index + char_in_token_idx] = len(bert_tokens) + i
                        char_in_token_idx += 1
            else:
                bert_token2ori_token[len(bert_tokens)] = j
                for char_in_token_idx in range(len(token)):
                    char2bert_token[char_index + char_in_token_idx] = len(bert_tokens)
                # print('[UNK]:', [token], utterance)

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
        keepers.append(c)                            # 否则，保存当前字符。
        # 如果不是组合字符，那就是新的基字符
        if not unicodedata.combining(c):             # 检测新的基字符，判断是不是拉丁字母。
            latin_base = c in string.ascii_letters  # ascii 是 拉丁字符罗？ 为false时，即为标记.
    shaved = ''.join(keepers)
    return unicodedata.normalize('NFC', shaved)      # 重组所有字符


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


def preprocess(dataset_names):
    """
    preprocess datasets for all task
    :param dataset_names: List, elements of dataset_names should match names of dataset directories in $data_dir
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    for _ds in dataset_names:
        logging.info('preprocessing {} dataset'.format(_ds))
        _dataset_dir = os.path.join(data_dir, _ds)
        _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
        if not os.path.exists(_data_zipfile):
            raise FileNotFoundError('Data Not Found!', _data_zipfile)
        else:
            _dialogs = read_zipped_json(_data_zipfile, 'data.json')
            _ontology = json.load(open(os.path.join(_dataset_dir, 'ontology.json')))

        if _ds in ['multiwoz21', 'multiwoz25', 'schema', 'oos', 'hwu', 'clinc', 'banking', 'restaurant8k', 'top', 'dstc2', 'm2m']:
            # only training set
            _dialogs = [d for d in _dialogs if d['data_split'] == 'train']

        for d in _ontology['domains']:
            for s in _ontology['domains'][d]['slots']:
                _ontology['domains'][d]['slots'][s]['possible_values'] = {}

        train_data_json = []
        dev_data_json = []
        da2utt_dict = {"user": defaultdict(list), "system": defaultdict(list)}
        random.seed(42)
        num_train_samples, num_dev_samples = 0, 0

        print_example = True

        for dial_id, _d in enumerate(tqdm(_dialogs, desc='processing {}'.format(_ds))):
            history = []
            turns_intents = []
            spans = []
            da_list = []
            _turns = _d['turns']
            if len(_turns) > 60:
                # exceed max turn_id limit = (30)
                print('cut turn', len(_turns))
                _turns = _turns[:59]
            for _turn in _turns:
                _utterance = _turn['utterance']
                history.append(_utterance)
            #     intents = []
            #     da_seqs = [_turn["speaker"]]
            #     noncat_cnt = defaultdict(list)
            #     # dialogue act
            #     idx_in_span_mask = [0] * len(_utterance)
            #     for typ, values in _turn['dialogue_act'].items():
            #         if typ == 'non-categorical':
            #             for value in values:
            #                 if value['value'] == 'dontcare':
            #                     continue
            #                 if 'start' not in value:
            #                     # print('da no span value:', _utterance, value)
            #                     continue
            #                 value_span_strip(value)
            #                 value['utt_idx'] = _turn['utt_idx']
            #
            #                 # assert all([idx_in_span_mask[i] == 0 for i in range(value['start'], value['end'])]), print(_utterance, value, spans)
            #                 if all([idx_in_span_mask[i] == 0 for i in range(value['start'], value['end'])]):
            #                     spans.append(value)
            #                     idx_in_span_mask[value['start']:value['end']] = [1] * (value['end'] - value['start'])
            #
            #                     da_seq = str((value["intent"], value["domain"], value["slot"]))
            #                     # remove redundant da, add cnt to distinct different values for the same slot.
            #                     if value["value"] not in noncat_cnt[da_seq]:
            #                         noncat_cnt[da_seq].append(value["value"])
            #                         da_seq = str((value["intent"], value["domain"], value["slot"], len(noncat_cnt[da_seq])))
            #                         da_seqs.append(da_seq)
            #                 # else:
            #                 #     # ignore overlap span
            #                 #     print(_utterance, value)  # only three cases have overlap spans for schema dataset
            #         else:
            #             for value in values:
            #                 da_seq = str((value["intent"], value["domain"], value["slot"], value["value"]))
            #                 if da_seq not in da_seqs:
            #                     da_seqs.append(da_seq)
            #         for value in values:
            #             _intent = value['intent']
            #             _domain = value['domain']
            #             _slot = value['slot']
            #             _value = value['value']
            #             if [_intent, _domain, _slot] not in intents:
            #                 intents.append([_intent, _domain, _slot])
            #             if typ != 'binary':
            #                 # add to ontology
            #                 _ontology['domains'][_domain]['slots'][_slot]['possible_values'][_value] = \
            #                 _ontology['domains'][_domain]['slots'][_slot]['possible_values'].get(_value, 0) + 1
            #     turns_intents.append(intents)
            #     da_list.append("-".join(sorted(da_seqs)))
            #     # # state update, not use for schema although some slot will refer to the same value in different turns.
            #     # if _turn['speaker'] == 'system':
            #     #     continue
            #     # for typ, values in _turn['state_update'].items():
            #     #     if typ == 'non-categorical':
            #     #         for value in values:
            #     #             if value['value'] == 'dontcare':
            #     #                 continue
            #     #             if 'start' not in value:
            #     #                 # print('state update no span value:', _utterance, value)
            #     #                 continue
            #     #             if 'fixed_value' in value:
            #     #                 value['value'] = value['fixed_value']
            #     #             value_span_strip(value)
            #     #             assert (value["utt_idx"], value["start"], value["end"]) in [(x["utt_idx"], x["start"], x["end"]) for x in spans], print(dial_id, spans, value)
            #     #             spans.append(value)
            #
            # # DONE: tokenize utterance, update history and spans
            # spans = sorted(spans, key=lambda x:x['utt_idx'])
            # turns_spans = []
            turns_tokens = []
            # turns_bio_tags = []
            for utt_idx, utt in enumerate(history):
                cur_spans = []
            #     for i, span in enumerate(spans):
            #         if span['utt_idx'] == utt_idx:
            #             cur_spans.append(span)
            #         if span['utt_idx'] > utt_idx:
            #             break
            #     cur_spans = sorted(cur_spans, key=lambda x:x["start"])  # sort by start loc
                bert_tokens, tags, Xmask, ori_tokens, bert_token2ori_token, char2bert_token = generate_bio_tag(utt, [[x['start'], x['end']] for x in cur_spans], tokenizer)
                turns_tokens.append(bert_tokens)
            #     turns_bio_tags.append(tags)
            #     turns_spans.append([])
            #     for span in cur_spans:
            #         start = span['start']
            #         end = span['end']
            #         tok_start, tok_end = char2bert_token[start], char2bert_token[end - 1] + 1
            #         assert utt_idx == span['utt_idx']
            #         ele = {
            #             "intent": span["intent"],
            #             "domain": span["domain"],
            #             "slot": span["slot"],
            #             "value": span["value"],
            #             "utt_idx": utt_idx,
            #             "start": tok_start,
            #             "end": tok_end
            #         }
            #         if ele not in turns_spans[-1]:
            #             turns_spans[-1].append(ele)
            #     # turns_spans[-1] = sorted(turns_spans[-1], key=lambda x: x['end']-x['start'])
            #     da2utt_dict[da_list[utt_idx].split('-')[-1]][da_list[utt_idx]].append({
            #         "utterance": bert_tokens,
            #         "da_spans": turns_spans[-1],
            #         "intent": turns_intents[utt_idx]
            #     })

            if not print_example:
                pprint(turns_tokens)
                # pprint(turns_spans)
                # print(turns_bio_tags)
                print(turns_intents)
                print()
                print_example = True

            T = len(history)
            if random.random() < dev_ratio:
                new_data = dev_data_json
                num_dev_samples += 1
            else:
                new_data = train_data_json
                num_train_samples += 1

            new_data.append({
                'num_utt': T,
                'dialogue': turns_tokens,
                # 'da_list': da_list,
                # 'spans': turns_spans,
                'dataset': _ds,
                # 'bio_tag': turns_bio_tags,
                # 'intent': turns_intents
            })

        json.dump(train_data_json, open(os.path.join(processed_data_dir, _ds + '_data_train.json'), 'w'), indent=2)
        json.dump(dev_data_json, open(os.path.join(processed_data_dir, _ds + '_data_dev.json'), 'w'), indent=2)
        json.dump(_ontology, open(os.path.join(processed_data_dir, _ds + '_ontology.json'), 'w'), indent=2)
        json.dump(da2utt_dict, open(os.path.join(processed_data_dir, _ds + '_utt_pool.json'), 'w'), indent=2)
        map_len = {x: len(y) for x, y in da2utt_dict.items()}
        json.dump(map_len, open(os.path.join(processed_data_dir, _ds + '_map_len.json'), 'w'), indent=2)
        logging.info(
            '{} dataset preprocessed, train: {} dialogues {} samples, dev: {} dialogues {} samples'.format(
                _ds, len(train_data_json), num_train_samples, len(dev_data_json), num_dev_samples))


def preprocess4rsa_test(dataset_names):
    """
    preprocess datasets for RSA
    :param dataset_names: List, elements of dataset_names should match names of dataset directories in $data_dir
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    for _ds in dataset_names:
        logging.info('preprocessing {} dataset'.format(_ds))
        _dataset_dir = os.path.join(data_dir, _ds)
        _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
        if not os.path.exists(_data_zipfile):
            raise FileNotFoundError('Data Not Found!', _data_zipfile)
        else:
            _dialogs = read_zipped_json(_data_zipfile, 'data.json')

        if _ds in ['multiwoz21', 'multiwoz25', 'schema', 'oos', 'hwu', 'clinc', 'banking', 'restaurant8k', 'top', 'dstc2', 'm2m']:
            # only dev set
            _dialogs = [d for d in _dialogs if d['data_split'] == 'test']
        else:
            continue
        new_data = []
        for dial_id, _d in enumerate(tqdm(_dialogs, desc='processing {}'.format(_ds))):
            history = []
            _turns = _d['turns']
            if len(_turns) > 60:
                # exceed max turn_id limit = (30)
                print('cut turn', len(_turns))
                _turns = _turns[:59]
            for _turn in _turns:
                _utterance = _turn['utterance']
                history.append(_utterance)
            turns_tokens = []
            for utt_idx, utt in enumerate(history):
                cur_spans = []
                bert_tokens, tags, Xmask, ori_tokens, bert_token2ori_token, char2bert_token = generate_bio_tag(utt, [[x['start'], x['end']] for x in cur_spans], tokenizer)
                turns_tokens.append(bert_tokens)

            T = len(history)

            new_data.append({
                'num_utt': T,
                'dialogue': turns_tokens,
                'dataset': _ds,
            })

        json.dump(new_data, open(os.path.join(processed_data_dir, 'rsa_' + _ds + '_test_data.json'), 'w'), indent=2)


if __name__ == '__main__':
    # full dialogue, cut to <= 59 utts (max turn id is 30)
    # write 117867 dials, 1529295 utts
    # without multiwoz: 109433 dials, 1424229 utts
    preprocess(dataset_names)
    preprocess4rsa_test(dataset_names)
