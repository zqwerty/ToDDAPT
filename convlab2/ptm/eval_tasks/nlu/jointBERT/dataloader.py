import numpy as np
import random
import math
import re
from copy import deepcopy


class Dataloader:
    def __init__(self, intent_vocab, tag_vocab, processor):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param prepare_input_fn: prepare input before training, including append context
        :param batch_fn: function that prepare batch input when training
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        self.tag2id['X'] = -1
        self.id2tag[-1] = 'X'
        self.processor = processor
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)

    def load_data(self, data, data_key, cut_sen_len, context_size):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data: preprocessed data
        :param cut_sen_len: maximum length of bert input
        :param context_size: context window size
        :return:
        """
        if data_key == 'train':
            self.intent_weight = [1] * len(self.intent2id)
        self.data[data_key] = deepcopy(data)
        max_sen_len = 0
        sen_len = []
        for d in self.data[data_key]:
            role, tokens, tags, intents, da, context, ori_tokens, bert_token2ori_token = d
            tokenized_tokens, tokenized_tags = self.processor.prepare_input_fn(role, tokens, tags, context, context_size, cut_sen_len)
            actual_sen_len = len(tokenized_tokens)
            max_sen_len = max(actual_sen_len, max_sen_len)
            sen_len.append(actual_sen_len)
            assert len(tokenized_tokens) == len(tokenized_tags) and len(tokenized_tags) <= cut_sen_len

            # d = ('user'/'sys', tokens, tags, intents, dialog_act, context, ori_tokens, bert_token2ori_token
            # tokenized_tokens, tokenized_tags, tags_id, intents_id)
            d.append(tokenized_tokens)
            d.append(tokenized_tags)
            d.append(self.seq_tag2id(tokenized_tags))
            d.append(self.seq_intent2id(intents))
            if data_key=='train':
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == 'train':
            train_size = len(self.data['train'])
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
        print('max sen bert len', max_sen_len)
        # print('intent weight', self.intent_weight)
        # print(sorted(Counter(sen_len).items()))

    def seq_tag2id(self, tags):
        return [self.tag2id[x] for x in tags]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    def pad_batch(self, batch_data):
        return self.processor.batch_fn(batch_data)

    def get_train_batch(self, batch_size):
        # d = ('user'/'sys', tokens, tags, intents, dialog_act, context,
        # tokenized_tokens, tokenized_tags, tags_id, intents_id)
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)


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
                    assert token_idx<len(token) and (c == token[token_idx].lower() or subwords == ['[UNK]']), print(utterance, token, subwords)
                    char2bert_token[char_index + token_idx] = len(bert_tokens) + i
                    token_idx += 1
            bert_tokens.extend(subwords)
        char_index += len(token)
    return ori_tokens, bert_tokens, char2bert_token, bert_token2ori_token, continued_bert_token_idxs


def generate_bio_tag(utterance, spans, tokenizer):
    """
    Generate BIO tags for a sentence with BERT tokenizer
    :param utterance: str,
    :param spans: list of (start, end, name) character-level span, utterance[start:end] gives the content
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
    for start, end, name in spans:
        assert start in char2bert_token and (end - 1) in char2bert_token, print(utterance, bert_tokens, char2bert_token, start, end)
        tok_start, tok_end = char2bert_token[start], char2bert_token[end - 1]
        # print(start, end)
        # print(char2bert_token)
        # print(tok_start, tok_end)
        tags[tok_start] = 'B' + '-' + name
        for tok_id in range(tok_start + 1, tok_end + 1):
            tags[tok_id] = 'I' + '-' + name
    Xmask = [1] * len(bert_tokens)
    for idx in continued_bert_token_idxs:
        Xmask[idx] = 0
        tags[idx] = 'X'
    assert len(bert_tokens) == len(tags)
    return bert_tokens, tags, Xmask, ori_tokens, bert_token2ori_token


def tag2triples(word_seq, tag_seq):
    assert len(word_seq) == len(tag_seq)
    triples = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            intent, slot = tag[2:].split('+')
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    value += ' ' + word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            triples.append([intent, slot, value])
        i += 1
    return triples
