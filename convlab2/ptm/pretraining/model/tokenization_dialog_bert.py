# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""


import collections
import logging
import os
import unicodedata
from typing import List, Optional
from collections import OrderedDict
import torch
from tokenizers import BertWordPieceTokenizer

from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
        "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
        "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
        "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/vocab.txt",
        "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/vocab.txt",
        "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "bert-base-finnish-cased-v1": 512,
    "bert-base-finnish-uncased-v1": 512,
    "bert-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "bert-base-finnish-cased-v1": {"do_lower_case": False},
    "bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "bert-base-dutch-cased": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class DialogBertTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to do basic tokenization before WordPiece.
        never_split (:obj:`bool`, `optional`, defaults to :obj:`True`):
            List of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese:
            see: https://github.com/huggingface/transformers/issues/328
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def get_num_tokens_to_remove(self, sequence, total_len, max_length):
        # the sequence always has utterance suffix, so remove from backward
        # ... [sys] utt [sep] [usr] utt [sep]

        # truncate # tokens for  ... [sys] utt [usr] utt [sep] tokens
        # if total_len <= max_length:
        #     return 0
        # else:
        #     role_tokens = [self.added_tokens_encoder['[USR]'], self.added_tokens_encoder['[SYS]']]
        #     for i in range(total_len-2, -1, -1):  # ignore [SEP] at the end
        #         if sequence[i] in role_tokens:
        #             if i + 1 <= max_length:
        #                 return total_len - i - 2
        #     raise ValueError

        # truncate # tokens for  ... [sys] utt [sep] [usr] utt [sep]
        if total_len <= max_length:
            return 0
        else:
            role_tokens = [self.added_tokens_encoder['[USR]'], self.added_tokens_encoder['[SYS]']]
            for i in range(total_len-1, -1, -1):  # ignore [SEP] at the end
                if sequence[i] in role_tokens:
                    if i <= max_length:
                        return total_len - i
            raise ValueError

    def convert_utt_ids_to_input(self, utt_ids: List[List[int]]):
        input_ids = [self.cls_token_id]
        for ids in utt_ids:
            input_ids.extend(ids)

        turn_ids, role_ids = self.create_token_type_ids_from_sequences(input_ids)
        position_ids = self.create_positional_ids_from_sequences(input_ids)
        attn_mask = [1 for _ in input_ids]
        
        encoded_inputs = {}
        encoded_inputs["input_ids"] = input_ids
        encoded_inputs['attention_mask'] = attn_mask
        encoded_inputs["turn_ids"] = turn_ids
        encoded_inputs['role_ids'] = role_ids
        encoded_inputs['position_ids'] = position_ids
        encoded_inputs['length'] = len(input_ids)
        
        return encoded_inputs

    def build_inputs_with_special_tokens(
        self, sentences: List[List[int]], utt_role='user'
    ) -> List[int]:
        """
        Build model inputs from a list of utterances with added special tokens
        and optionally prefixes

        :param
        sentences: list of list[int], representing utterances of one dialog. [u_t, s_t-1, u_t-1, ...]
        prefix_dict: a mapping of str -> str, description_special_token -> description text.
                    e.g. {'[DOMAIN]': domain_description, '[SLOT]': slot_description}
        utt_role: in ['user', 'system'], current utterance role

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        ret = [self.cls_token_id]
        sep = [self.sep_token_id]

        if utt_role == 'user':
            for i, ids in enumerate(sentences):
                if i % 2 == 0:
                    ret += [self.added_tokens_encoder['[USR]']] + ids + sep
                else:
                    ret += [self.added_tokens_encoder['[SYS]']] + ids + sep
        elif utt_role == 'system':
            for i, ids in enumerate(sentences):
                if i % 2 == 0:
                    ret += [self.added_tokens_encoder['[SYS]']] + ids + sep
                else:
                    ret += [self.added_tokens_encoder['[USR]']] + ids + sep
        else:
            raise ValueError('utt_role should be in [user, sys]')
        return ret

    def get_special_tokens_mask_bak(
        self, sequence, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        THIS FUNCTION SHOULD ONLY BE USED IN DIALOG MLM TASK !!!
        [cls] ... [usr] usr [sys] sys [usr] usr [sys]  sys [sep]
          1    1    1    0    1    0    1     1   1     1    1

        Args:
            sequence: List[int]

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        ret_mask = [1 for _ in sequence]
        special_tokens = list(self.unique_added_tokens_encoder)
        all_special_token_dict = {self.convert_tokens_to_ids(tok): tok for tok in special_tokens}
        is_first_turn = False

        for i, token_id in enumerate(sequence):
            if token_id in all_special_token_dict:
                if all_special_token_dict[token_id] == '[USR]' and is_first_turn is False:
                    is_first_turn = True
                elif all_special_token_dict[token_id] == '[USR]' and is_first_turn is True:
                    break
            if is_first_turn and token_id not in all_special_token_dict:
                ret_mask[i] = 0
        # display special token mask
        utt_print = ' '.join([tok for tok in self.convert_ids_to_tokens(sequence) if tok != '[PAD]'])
        logging.warning('utterance: {}'.format(utt_print))
        type_print = ' '.join([str(id) for id in ret_mask[:len(utt_print.split())]])
        logging.warning('special_token_mask_ids: {}'.format(type_print))
        return ret_mask


    def get_special_tokens_mask(
        self, sequence, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        THIS FUNCTION SHOULD ONLY BE USED IN DIALOG MLM TASK !!!
        [cls] ... [usr] usr [sys] sys [usr] usr [sys]  sys [sep]
          1    1    1    0    1    0    1     0   1     0    1

        Args:
            sequence: List[int]

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        ret_mask = [1 for _ in sequence]
        special_tokens = list(self.unique_added_tokens_encoder)
        all_special_token_dict = {self.convert_tokens_to_ids(tok): tok for tok in special_tokens}

        for i, token_id in enumerate(sequence):
            if token_id not in all_special_token_dict:
                ret_mask[i] = 0
        # display special token mask
        # utt_print = ' '.join([tok for tok in self.convert_ids_to_tokens(sequence) if tok != '[PAD]'])
        # logging.warning('utterance: {}'.format(utt_print))
        # type_print = ' '.join([str(id) for id in ret_mask[:len(utt_print.split())]])
        # logging.warning('special_token_mask_ids: {}'.format(type_print))
        return ret_mask

    def get_tokens_mask(
        self, token_ids, mask_tokens
    ) -> List[int]:
        """
        mask specific tokens in the input_ids

        Args:
            token_ids: List[int]

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for not masked tokens, 0 for masked tokens.
        """

        ret_mask = [1 for _ in token_ids]
        all_special_token_dict = {self.convert_tokens_to_ids(tok): tok for tok in mask_tokens}

        for i, token_id in enumerate(token_ids):
            if token_id in all_special_token_dict:
                ret_mask[i] = 0
        return ret_mask

    def get_tokens_not_mask(
        self, token_ids, not_mask_tokens
    ) -> List[int]:
        """
        mask specific tokens in the input_ids

        Args:
            token_ids: List[int]

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for not masked tokens, 0 for masked tokens.
        """

        ret_mask = [1 for _ in token_ids]
        all_special_token_dict = {self.convert_tokens_to_ids(tok): tok for tok in not_mask_tokens}

        for i, token_id in enumerate(token_ids):
            if token_id not in all_special_token_dict:
                ret_mask[i] = 0
        return ret_mask

    def create_positional_ids_from_sequences(self, sequence) -> List[int]:
        """
        create positional ids for bert inputs

        [cls] [desc] desc [usr] usr [sys] sys  [sep]
          0     1      2    0    1    0    1     2
        """

        positional_ids = []
        special_tokens = list(self.unique_added_tokens_encoder)
        all_special_token_dict = {self.convert_tokens_to_ids(tok): tok for tok in special_tokens}

        current_pos_id = 0
        for token_id in sequence:
            if token_id in all_special_token_dict:
                if all_special_token_dict[token_id] in ['[USR]', '[SYS]', '[DOMAIN]', '[SLOT]', '[VALUE]']:
                    current_pos_id = 0
                    positional_ids.append(current_pos_id)
                    current_pos_id += 1
                    continue
            positional_ids.append(current_pos_id)
            current_pos_id += 1

        # display positional ids
        # utt_print = ' '.join([tok for tok in self.convert_ids_to_tokens(sequence) if tok != '[PAD]'])
        # logging.warning('utterance: {}'.format(utt_print))
        # type_print = ' '.join([str(id) for id in positional_ids[:len(utt_print.split())]])
        # logging.warning('positional_ids: {}'.format(type_print))
        return positional_ids


    def create_token_type_ids_from_sequences(self, sequence) -> List[List[int]]:
        """
        :param sequence: List[int]

                [cls] [desc] desc [usr] usr [sys] sys  [usr]  usr  [sys] sys  [sep]
        turn_ids  0      0    0    1      1   1    1     2     2     2    2     2
        role_ids  0      0    0    1      1   2    2     1     1     2    2     2

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        turn_ids = []
        role_ids = []
        special_tokens = list(self.unique_added_tokens_encoder)
        all_special_token_dict = {self.convert_tokens_to_ids(tok): tok for tok in special_tokens}

        current_turn_id = 0
        current_role_id = 0
        for token_id in sequence:
            if token_id in all_special_token_dict:
                if all_special_token_dict[token_id] in ['[USR]', '[SYS]']:
                    if all_special_token_dict[token_id] == '[USR]':
                        current_role_id = 1
                        current_turn_id += 1

                    else:
                        current_role_id = 2
                    role_ids.append(current_role_id)
                    turn_ids.append(current_turn_id)
                    continue
            turn_ids.append(current_turn_id)
            role_ids.append(current_role_id)

        # for t in turn_ids:
        #     assert t < 11
        # for r in role_ids:
        #     assert r < 3
        # display type_ids
        # utt_print = ' '.join([tok for tok in self.convert_ids_to_tokens(sequence) if tok != '[PAD]'])
        # logging.warning('utterance: {}'.format(utt_print))
        # type_print = ' '.join([str(id) for id in turn_ids[:len(utt_print.split())]])
        # logging.warning('turn_ids: {}'.format(type_print))
        # role_print = ' '.join([str(id) for id in role_ids[:len(utt_print.split())]])
        # logging.warning('role_ids: {}'.format(role_print))
        return [turn_ids, role_ids]


    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


    def prepare_input_seq(self, sentences: List[List[str]], prefix_dict: OrderedDict = None, last_role='user',
                          max_length: Optional[int] = None,
                          truncation_strategy: str = "backwards",
                          return_tensors: Optional[str] = None,
                          return_overflowing_tokens: bool = False,
                          return_lengths: bool = False,
                          pad_to_max_len: bool = False):
        """
        Build model inputs from a list of utterances with added special tokens and optionally prefixes
        if `max_length` is not None, cut previous turns so the result dialog is shorter than `max_length` and always keep complete utterances

        :param
        sentences: list of list of tokens, representing utterances of one dialog. in sequential order [turn_t, turn_t+1, ...]
        prefix_dict: a mapping of str -> list of tokens, description_special_token -> description text.
                    e.g. {'[DOMAIN]': domain_description, '[SLOT]': slot_description, '[VALUE]': value_list}
        last_role: in ['user', 'system'], last utterance role

        Returns:
            A Dictionary of shape::

                {
                    input_ids: list[int], reverse order
                    turn_ids: list[int]
                    role_ids: list[int]
                    position_ids: list[int]
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    length: int if return_lengths is True
                }

            With the fields:
                - ``input_ids``: list of token ids to be fed to a model
                - ``turn_ids``: list of turn ids
                - ``role_ids``: list of role ids
                - ``position_ids``: list of pos ids
                - ``overflowing_tokens``: list of overflowing tokens if a max length is specified. start with sep
                - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified. truncate sentences
                - ``length``: this is the length of ``input_ids``
        """
        sequence = [self.cls_token_id]

        # add prefix input ids
        if prefix_dict:
            for key, value in prefix_dict.items():
                assert key in ['[INTENT]', '[DOMAIN]', '[SLOT]', '[VALUE]'] #TODO: replace with a function call for addtional tokens
                if key in ['[INTENT]', '[DOMAIN]', '[SLOT]']:
                    sequence += self.convert_tokens_to_ids([key] + self.tokenize(value)) + [self.sep_token_id]
                elif key in ['[VALUE]']:
                    for v in value:
                        sequence += self.convert_tokens_to_ids([key] + self.tokenize(v)) + [self.sep_token_id]

        # add dialogue input ids
        if last_role == 'user':
            odd_role_token_id = self.convert_tokens_to_ids('[USR]')
            even_role_token_id = self.convert_tokens_to_ids('[SYS]')
        elif last_role == 'system':
            odd_role_token_id = self.convert_tokens_to_ids('[SYS]')
            even_role_token_id = self.convert_tokens_to_ids('[USR]')
        else:
            raise ValueError('utt_role should be in [user, sys]')
        for i, sen in enumerate(sentences):
            if i % 2 == 0:
                role_token_id = odd_role_token_id
            else:
                role_token_id = even_role_token_id
            sequence += [role_token_id] + self.convert_tokens_to_ids(sen) + [self.sep_token_id]

        encoded_inputs = {}

        total_len = len(sequence)
        if max_length and total_len > max_length:
            sequence, overflowing_tokens = self.truncate_sequences(
                sequence,
                num_tokens_to_remove=self.get_num_tokens_to_remove(sequence, total_len, max_length),
                truncation_strategy=truncation_strategy,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = len(overflowing_tokens)

        # attn_mask = [1 for _ in sequence]
        #
        # if pad_to_max_len:
        #     attn_mask += [0] * (max_length - len(sequence))
        #     sequence += self.convert_tokens_to_ids([self.pad_token] * (max_length - len(sequence)))

        encoded_inputs["input_ids"] = sequence
        # encoded_inputs['attention_mask'] = attn_mask

        # add role, turn, positional ids
        # turn_ids, role_ids = self.create_token_type_ids_from_sequences(sequence)
        # position_ids = self.create_positional_ids_from_sequences(sequence)
        # encoded_inputs["turn_ids"] = turn_ids
        # encoded_inputs['role_ids'] = role_ids
        # encoded_inputs['position_ids'] = position_ids

        assert max_length is None or len(encoded_inputs["input_ids"]) <= max_length, print(max_length, len(
            encoded_inputs["input_ids"]))

        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(encoded_inputs["input_ids"]), self.model_max_length)
            )

        if return_lengths:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        if return_tensors == "pt":
            encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

            # if "token_type_ids" in encoded_inputs:
            #     encoded_inputs["turn_ids"] = torch.tensor([encoded_inputs["turn_ids"]])
            #     encoded_inputs['role_ids'] = torch.tensor([encoded_inputs["role_ids"]])
            #     encoded_inputs['position_ids'] = torch.tensor([encoded_inputs['position_ids']])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])
        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )

        return encoded_inputs



class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
            Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
