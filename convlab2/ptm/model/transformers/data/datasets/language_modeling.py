import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer=None, file_path: str=None, block_size=None, display=False):
        # assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        self.examples = None

        if tokenizer and file_path:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,
                                                         )
            self.examples = batch_encoding["input_ids"]
            self.num_example_turns = batch_encoding['num_utts']

            if display:
                for i in range(3):
                    logging.info('display 3 examples')
                    logging.info('=====================input text==========================')
                    logging.info(tokenizer.convert_ids_to_tokens(batch_encoding['input_ids'][i]))
                    logging.info('=====================input ids==========================')
                    logging.info(batch_encoding['input_ids'][i])
                    logging.info('=====================turn ids==========================')
                    logging.info(batch_encoding['turn_ids'][i])
                    logging.info('=====================role ids==========================')
                    logging.info(batch_encoding['role_ids'][i])
                    logging.info('=====================position ids==========================')
                    logging.info(batch_encoding['position_ids'][i])


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
