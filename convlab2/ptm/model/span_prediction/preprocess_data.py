import os
import logging
logging.basicConfig(level=logging.INFO)
import sys
import random
import json
from tqdm import tqdm
import six
import re

self_dir = os.path.dirname(os.path.abspath(__file__))


def parent_dir(path, time=1):
    for _ in range(time):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(self_dir, 4))
print(sys.path[-1])

from convlab2.util.file_util import read_zipped_json, write_zipped_json
from convlab2.ptm.model.transformers.tokenization_bert import BertTokenizer

# import data_ptm.camrest.preprocess as camrest_preprocess
# import data_ptm.multiwoz.preprocess as multiwoz_preprocess
# import data_ptm.frames.preprocess as frames_preprocess
# import data_ptm.mdc.preprocess as mdc_preprocess
# import data_ptm.mit_movie.preprocess as mit_movie_preprocess
import data_ptm.schema.preprocess as schema_preprocess
# import data_ptm.woz.preprocess as woz_preprocess


random.seed(1234)
dev_ratio = 0.01  # ratio of data used to evaluate mlm

# datasets used in pretraining
# dev/test set of multiwoz and schema are excluded automatically
dataset_names = [
    # 'camrest',
    # 'multiwoz',
    # 'woz',
    # 'mit_movie',
    # 'frames',
    'schema'
]

data_dir = os.path.join(parent_dir(os.path.abspath(__file__), 5), 'data')
data_file_paths = []


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
            # Store the alignment for the index of character to corresponding
            # token
            token_idx = 0
            for i, sw in enumerate(subwords):
                if sw.startswith('##'):
                    sw = sw[2:]
                    continued_bert_token_idxs.append(len(bert_tokens) + i)
                bert_token2ori_token[len(bert_tokens) + i] = j
                for c in sw:
                    assert c == token[token_idx].lower(), print(
                        token, subwords)
                    char2bert_token[
                        char_index + token_idx] = len(bert_tokens) + i
                    token_idx += 1
            bert_tokens.extend(subwords)
        char_index += len(token)
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


def preprocess(
        dataset_names,
        out_train_filename='_span_prediction_data_train.json',
        out_dev_filename='_span_prediction_data_dev.json'):
    """
    preprocess datasets for span prediction, containing
    :param dataset_names: List, elements of dataset_names should match names of dataset directories in data/
    :param history_length: number of history turns. 0 means only this turn
    :return:
    todo: returns could start with sys
    output form: [usr_t  sys_t  usr_t-1  sys_t-1 ... usr_t-history_length, sys_t-history_length], padded with ' '
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('/home/libing/pretrained_bert/bert-base-uncased')

    for _ds in dataset_names:
        logging.info('preprocessing {} dataset'.format(_ds))
        _dataset_dir = os.path.join(data_dir, _ds + '_ptm') if os.path.isdir(
            os.path.join(data_dir, _ds + '_ptm')) else os.path.join(data_dir, _ds)
        _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
        if not os.path.exists(_data_zipfile):
            _dialogs, _ontology = eval(_ds + '_preprocess').preprocess()
        else:
            _dialogs = read_zipped_json(_data_zipfile, 'data.json')
            _ontology = json.load(open(os.path.join(_dataset_dir, 'ontology.json')))

        # if _ds in ['multiwoz', 'schema']:
        #     _dialogs = [d for d in _dialogs if d['data_split'] == 'train']
        json.dump(_ontology, open(os.path.join(self_dir, _ds + '_ontology.json'), 'w'), indent=2)

        train_data_json = []
        dev_data_json = []
        random.seed(42)

        for _d in tqdm(_dialogs, desc='processing {}'.format(_ds)):
            history = []

            for _turn in _d['turns']:
                _utterance = _turn['utterance']
                _da = _turn['dialogue_act']

                for typ, values in _da.items():
                    if typ != 'non-categorical':
                        continue

                    for value in values:
                        _origin_domain = value['domain']
                        _intent = value['intent']
                        _origin_slot = value['slot']
                        _processed_utt, _tokenized_span, Xmask, ori_tokens, bert2oritoken = generate_span_data(_utterance, [value['start'], value['end']], tokenizer)
                        history.insert(0, _processed_utt)

                        if _ds in ['multiwoz', 'schema'] :
                            if  _d['data_split'] in ['train']:
                                train_data_json.append({
                                    "dialogue": history,
                                    "span": _tokenized_span,
                                    'continue_word_mask': Xmask,
                                    'dataset': _ds,
                                    'domain': _origin_domain,
                                    'slot': _origin_slot,
                                    'intent': _intent
                                })
                            elif _d['data_split'] in ['val', 'test']:
                                dev_data_json.append({
                                    "dialogue": history,
                                    "span": _tokenized_span,
                                    'continue_word_mask': Xmask,
                                    'dataset': _ds,
                                    'domain': _origin_domain,
                                    'slot': _origin_slot,
                                    'intent': _intent
                                })
                            else:
                                raise ValueError('data split {} not available'.format(_d['data_split']))

                        else:
                            if random.random() < dev_ratio:
                                dev_data_json.append({
                                    "dialogue": history,
                                    "span": _tokenized_span,
                                    'continue_word_mask': Xmask,
                                    'dataset': _ds,
                                    'domain': _origin_domain,
                                    'slot': _origin_slot,
                                    'intent': _intent
                                })
                            else:
                                train_data_json.append({
                                    "dialogue": history,
                                    "span": _tokenized_span,
                                    'continue_word_mask': Xmask,
                                    'dataset': _ds,
                                    'domain': _origin_domain,
                                    'slot': _origin_slot,
                                    'intent': _intent
                                })

        json.dump(train_data_json, open(os.path.join(self_dir, _ds + out_train_filename), 'w'), indent=2)
        json.dump(dev_data_json, open(os.path.join(self_dir, _ds + out_dev_filename), 'w'), indent=2)
        logging.info(
            '{} dataset preprocessed, written {} train examples, {} dev examples'.format(
                _ds, len(train_data_json), len(dev_data_json)))


if __name__ == '__main__':
    preprocess(dataset_names)
