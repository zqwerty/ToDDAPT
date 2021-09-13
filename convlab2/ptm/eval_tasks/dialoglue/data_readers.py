import csv
import logging
import json
import numpy as np
import os
import pickle
import torch

from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict

from transformers import BertTokenizer

from constants import SPECIAL_TOKENS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class IntentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_intent_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            reader = csv.reader(open(data_path, encoding="utf-8"))
            next(reader, None)
            out = []
            for utt, intent in tqdm(reader):
                encoded = tokenizer.encode(utt)

                self.examples.append({
                    "input_ids": np.array(encoded.ids)[:max_seq_length],
                    "attention_mask": np.array(encoded.attention_mask)[:max_seq_length],
                    "token_type_ids": np.array(encoded.type_ids)[:max_seq_length],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TodBertIntentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_todbert_intent_cached".format(split))
        if not os.path.exists(cached_path):
            self.examples = []
            reader = csv.reader(open(data_path, encoding="utf-8"))
            next(reader, None)
            out = []
            for utt, intent in tqdm(reader):

                encoded = tokenizer.encode(utt)

                encoded_ids = encoded.ids[:1] + [tokenizer.token_to_id('[usr]')] + encoded.ids[1:]

                self.examples.append({
                    "input_ids": np.array(encoded_ids)[:max_seq_length],
                    "attention_mask": np.array(encoded.attention_mask[:1] + [1] + encoded.attention_mask[1:])[
                                      :max_seq_length],
                    "token_type_ids": np.array(encoded.type_ids[:1] + [0] + encoded.type_ids[1:])[:max_seq_length],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })

            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DialogBertIntentDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_length: int,
                 vocab_file_name: str,
                 ):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_dialogbert_intent_cached".format(split))
        if not os.path.exists(cached_path):
            self.examples = []
            if os.path.exists(data_path):
                reader = csv.reader(open(data_path, encoding="utf-8"))
                next(reader, None)
                out = []
                for utt, intent in tqdm(reader):
                    tokenized_utt = tokenizer.tokenize(utt)
                    encoded_dict = tokenizer.prepare_input_seq([tokenized_utt], max_length=max_seq_length, pad_to_max_len=True)

                    assert len(encoded_dict['input_ids']) == max_seq_length
                    self.examples.append({
                        "input_ids": np.array(encoded_dict['input_ids']),
                        'attention_mask': np.array(encoded_dict['attention_mask']),
                        'turn_ids': np.array(encoded_dict['turn_ids']),
                        'role_ids': np.array(encoded_dict['role_ids']),
                        'position_ids': np.array(encoded_dict['position_ids']),
                        "intent_label": self.intent_label_to_idx[intent],
                        "ind": len(self.examples),
                    })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k:torch.tensor(v) for k, v in self.examples[idx].items()}




class DialogBertSlotDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer,
                 max_seq_length: int,
                 vocab_file_name: str,
                 help_tokenizer):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.txt")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Process data
        self.dialogbert_tokenizer = tokenizer
        self.tokenizer = help_tokenizer
        cached_path = os.path.join(data_dirname, "{}_dialogbert_slots_cached".format(split))
        texts = []
        slotss = []
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path, encoding="utf-8"))
            for example in tqdm(data):
                text, slots = self.parse_example(example)
                texts.append(text)
                slotss.append(slots)
                encoded = self.tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               self.tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                # encoded and encoded_slot_labels are not padded
                encoded_slot_labels = encoded_slot_labels.tolist()
                encoded_slot_labels = encoded_slot_labels[:1] + encoded_slot_labels[:1] + encoded_slot_labels[1:]
                encoded_dict = self.dialogbert_tokenizer.prepare_input_seq([encoded.tokens[1:-1]], max_length=max_seq_length, pad_to_max_len=True)
                encoded_slot_labels = encoded_slot_labels + encoded_slot_labels[:1] * (len(encoded_dict['input_ids']) - len(encoded_slot_labels))

                self.examples.append({
                    # tensors in encoded_dict should be exactly 50, so I remove [:max_seq_length]'s to ensure
                    "input_ids": np.array(encoded_dict['input_ids']),
                    'attention_mask': np.array(encoded_dict['attention_mask']),
                    'turn_ids': np.array(encoded_dict['turn_ids']),
                    'role_ids': np.array(encoded_dict['role_ids']),
                    'position_ids': np.array(encoded_dict['position_ids']),
                    # previously follow dialoglue style, now change it
                    "slot_labels": np.array(encoded_slot_labels[:max_seq_length]),
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:

        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))

        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def parse_example(self, example):
        text = example['userInput']['text']

        # Create slots dictionary
        word_to_slot = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)

            for word in text[start:end].split():
                word_to_slot[word] = slot

        # Add context if it's there
        if 'context' in example:
            for req in example['context'].get('requestedSlots', []):
                text = req + " " + text

        # Create slots list
        slots = []
        cur = None
        for word in text.split():
            if word in word_to_slot:
                slot = word_to_slot[word]
                if cur is not None and slot == cur:
                    slots.append("I-" + slot)
                else:
                    slots.append("B-" + slot)
                    cur = slot
            else:
                slots.append("O")
                cur = None

        return text, " ".join(slots)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k:torch.tensor(v) for k, v in self.examples[idx].items()}


class SlotDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.txt")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_slots_cached".format(split, vocab_file_name))
        texts = []
        slotss = []
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path, encoding="utf-8"))
            for example in tqdm(data):
                text, slots = self.parse_example(example) 

                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[:max_seq_length],
                    "attention_mask": np.array(encoded.attention_mask)[:max_seq_length],
                    "token_type_ids": np.array(encoded.type_ids)[:max_seq_length],
                    "slot_labels": encoded_slot_labels[:max_seq_length],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def parse_example(self, example):
        text = example['userInput']['text']

        # Create slots dictionary
        word_to_slot = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)

            for word in text[start:end].split():
                word_to_slot[word] = slot
          
        # Add context if it's there
        if 'context' in example:
            for req in example['context'].get('requestedSlots', []):
                text = req + " " + text

        # Create slots list
        slots = []
        cur = None
        for word in text.split():
            if word in word_to_slot:
                slot = word_to_slot[word]
                if cur is not None and slot == cur:
                    slots.append("I-" + slot) 
                else:
                    slots.append("B-" + slot) 
                    cur = slot
            else:
                slots.append("O")
                cur = None

        return text, " ".join(slots) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TodBertSlotDataset(SlotDataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.txt")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_todbert_slots_cached".format(split, vocab_file_name))

        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path, encoding="utf-8"))
            for example in tqdm(data):
                text, slots = self.parse_example(example)

                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                encoded_ids = encoded.ids[:1] + [tokenizer.token_to_id('[usr]')] + encoded.ids[1:]
                # print(encoded_ids)
                self.examples.append({
                    "input_ids": np.array(encoded_ids)[:max_seq_length],
                    "attention_mask": np.array(encoded.attention_mask[:1] + [1] + encoded.attention_mask[1:])[:max_seq_length],
                    "token_type_ids": np.array(encoded.type_ids[:1] + [0] + encoded.type_ids[1:])[:max_seq_length],
                    "slot_labels": encoded_slot_labels[:max_seq_length],
                    "ind": len(self.examples),
                })

            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:

        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))

        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 2:len(encoded_labels) + 2] = encoded_labels
        return encoded.squeeze()



class TOPDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.slot")
        slot_names = [e.strip() for e in open(slot_vocab_path).readlines()]
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "vocab.intent")
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_top_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            data = [e.strip() for e in open(data_path, encoding="utf-8").readlines() ]
            for example in tqdm(data):
                example, intent = example.split(" <=> ")
                text = " ".join([e.split(":")[0] for e in example.split()])
                slots = " ".join([e.split(":")[1] for e in example.split()])
                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[:max_seq_length],
                    "attention_mask": np.array(encoded.attention_mask)[:max_seq_length],
                    "token_type_ids": np.array(encoded.type_ids)[:max_seq_length],
                    "slot_labels": encoded_slot_labels[:max_seq_length],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]




class TodBertTOPDataset(TOPDataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.slot")
        slot_names = [e.strip() for e in open(slot_vocab_path).readlines()]
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "vocab.intent")
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_todbert_top_cached".format(split))
        if not os.path.exists(cached_path):
            self.examples = []
            # debug
            data = [e.strip() for e in open(data_path, encoding="utf-8").readlines()]
            for example in tqdm(data):
                example, intent = example.split(" <=> ")
                text = " ".join([e.split(":")[0] for e in example.split()])
                slots = " ".join([e.split(":")[1] for e in example.split()])
                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                # print('max_seq_length', max_seq_length)
                encoded_ids = encoded.ids[:1] + [tokenizer.token_to_id('[usr]')] + encoded.ids[1:]
                self.examples.append({
                    "input_ids": np.array(encoded_ids)[:max_seq_length],
                    "attention_mask": np.array(encoded.attention_mask[:1] + [1] + encoded.attention_mask[1:])[:max_seq_length],
                    "token_type_ids": np.array(encoded.type_ids[:1] + [0] + encoded.type_ids[1:])[:max_seq_length],
                    "slot_labels": encoded_slot_labels[:max_seq_length],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
                # print('example input shape', self.examples[-1]['input_ids'].shape)

            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:

        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))

        encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 2:len(encoded_labels) + 2] = encoded_labels[:min(max_length-2, len(encoded_labels))]
        # print('encoded shape', encoded.shape)
        # encoded = encoded[:, :max_length]
        return encoded.squeeze()


class DialogBertTOPDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str,
                 help_tokenizer):
        '''

        Args:
            tokenizer: dialogbert tokenizer
            help_tokenizer: wordpiece tokenizer
        '''
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.slot")
        slot_names = [e.strip() for e in open(slot_vocab_path).readlines()]
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "vocab.intent")
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = help_tokenizer
        self.dialogbert_tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_top_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            data = [e.strip() for e in open(data_path, encoding='utf-8').readlines()]
            for example in tqdm(data):
                example, intent = example.split(" <=> ")
                text = " ".join([e.split(":")[0] for e in example.split()])
                slots = " ".join([e.split(":")[1] for e in example.split()])
                encoded = self.tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               self.tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)

                encoded_slot_labels = encoded_slot_labels.tolist()
                encoded_slot_labels = encoded_slot_labels[:1] + encoded_slot_labels[:1] + encoded_slot_labels[1:]
                encoded_dict = self.dialogbert_tokenizer.prepare_input_seq([encoded.tokens[1:-1]],
                                                                           max_length=max_seq_length,
                                                                           pad_to_max_len=True)
                encoded_slot_labels = encoded_slot_labels + encoded_slot_labels[:1] * (
                            len(encoded_dict['input_ids']) - len(encoded_slot_labels))

                self.examples.append({
                    "input_ids": np.array(encoded_dict['input_ids']),
                    'attention_mask': np.array(encoded_dict['attention_mask']),
                    'turn_ids': np.array(encoded_dict['turn_ids']),
                    'role_ids': np.array(encoded_dict['role_ids']),
                    'position_ids': np.array(encoded_dict['position_ids']),
                    "slot_labels": np.array(encoded_slot_labels[:max_seq_length]),
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })

            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:

        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))

        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
