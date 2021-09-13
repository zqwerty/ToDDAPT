import os
import logging
logging.basicConfig(level=logging.INFO)
import sys
import random
import re

self_dir = os.path.dirname(os.path.abspath(__file__))


def parent_dir(path, time=1):
    for _ in range(time):
        path = os.path.dirname(path)
    return path


sys.path.append(parent_dir(self_dir, 4))
print(sys.path[-1])

from data_ptm.camrest import preprocess as camrest_preprocess
from data_ptm.dstc2 import preprocess as dstc2_preprocess
from data_ptm.frames import preprocess as frames_preprocess
from data_ptm.kvret import preprocess as kvret_preprocess
from data_ptm.m2m import preprocess as m2m_preprocess
from data_ptm.mdc import preprocess as mdc_preprocess
from data_ptm.metalwoz import preprocess as metalwoz_preprocess
from data_ptm.multiwoz25 import preprocess as multiwoz25_preprocess
from data_ptm.schema import preprocess as schema_preprocess
from data_ptm.taskmaster import preprocess as taskmaster_preprocess
from data_ptm.woz import preprocess as woz_preprocess


random.seed(1234)
dev_ratio = 0.01  # ratio of data used to evaluate mlm

# datasets used in pretraining
# dev/test set of multiwoz25 and schema are excluded automatically

dataset_names = [
    'camrest',
    # 'multiwoz25',
    'woz',
    'schema',
    'metalwoz',  # todo
    'taskmaster',
    'dstc2',
    'frames',
    'kvret',  # todo
    'm2m',
    'mdc'  # todo
]

data_dir = os.path.join(parent_dir(os.path.abspath(__file__), 4), 'data_ptm')
data_file_paths = []


def preprocess(dataset_names, mode='one_turn', out_train_filename='mlm_data_train.txt', out_dev_filename='mlm_data_dev.txt', num_turns=1):
    '''
    preprocess datasets for mlm
    :param out_dev_filename:
    :param out_train_filename:
    :param dataset_names: List, elements of dataset_names should match names of dataset directories in data/
    :param mode: in ['one_turn', 'full_dialog', 'multi_turn']
    :param num_turns: if mode == 'multi_turn', this param must be specified
    :return:
    output form: [usr_t  sys_t  usr_t-1  sys_t-1 ... usr_t-history_length, sys_t-history_length], padded with ' '
                always end with user turn
    '''
    if os.path.exists(out_train_filename) or os.path.exists(out_dev_filename):
        logging.warning('output train/dev file exists!')
        raise ValueError

    for _ds in dataset_names:

        _dataset_dir = os.path.join(data_dir, _ds+'_ptm') if os.path.isdir(os.path.join(data_dir, _ds+'_ptm')) else os.path.join(data_dir, _ds)
        _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
        _dialogs, _ = eval(_ds + '_preprocess').preprocess()

        lines_written = 0

        train_writer = open(os.path.join(self_dir, out_train_filename), 'a+')
        dev_writer = open(os.path.join(self_dir, out_dev_filename), 'a+')

        logging.info('preprocessing {} dataset'.format(_ds))
        logging.info('{} dataset, {} dialogs'.format(_ds, len(_dialogs)))

        str_to_train = ''
        str_to_dev = ''

        for _d in _dialogs:

            _turns = _d['turns']
            assert _d['data_split'] in ['train', 'val', 'test', 'validate', 'all']

            if mode == 'one_turn':
                _history = ['']
                for _turn in (_turns):
                    _utt = _turn['utterance']
                    if _turn['speaker'] == 'user':
                        _history.insert(0, _utt)
                        if _ds in ['multiwoz25', 'schema'] and _d['data_split'] == 'train':
                            str_to_train += '\t'.join(_history[:2]) + '\n'
                        elif _ds in ['multiwoz25', 'schema'] and _d['data_split'] in ['val', 'test', 'validate']:
                            str_to_dev += '\t'.join(_history[:2]) + '\n'
                        elif random.random() < dev_ratio:
                            str_to_dev += '\t'.join(_history[:2]) + '\n'
                        else:
                            str_to_train += '\t'.join(_history[:2]) + '\n'
                        lines_written += 1
                    else:
                        _history.insert(0, _utt)

            elif mode == 'full_dialog':
                if len(_turns) > 59:
                    _turns = _turns[:59]
                if _turns[-1]['speaker'] == 'system':
                    _turns = _turns[:-1]
                if _turns[0]['speaker'] == 'system':
                    _turns = _turns[1:]

                split = ''
                if random.random() < dev_ratio:
                    split = 'dev'
                else:
                    split = 'train'

                _history = ['']
                for _turn in (_turns):
                    _utt = re.sub('\s+', ' ', _turn['utterance'])
                    if _turn['speaker'] == 'user':
                        _history.insert(0, _utt)
                        # assert len(_history) < 60
                        if _ds in ['multiwoz25', 'schema'] and _d['data_split'] == 'train':
                            str_to_train += '\t'.join(_history) + '\n'
                        elif _ds in ['multiwoz25', 'schema'] and _d['data_split'] in ['val', 'test', 'validate']:
                            str_to_dev += '\t'.join(_history) + '\n'
                        elif split == 'dev':
                            str_to_dev += '\t'.join(_history) + '\n'
                        else:
                            str_to_train += '\t'.join(_history) + '\n'
                        lines_written += 1
                    else:
                        _history.insert(0, _utt)
                assert len(_history) <= 60

            elif mode == 'multi_turn':
                # always end with user utterance
                _history = ['']  # system is '' at first turn
                assert num_turns > 0
                if _turns[-1]['speaker'] == 'system':  # make sure user speak last
                    _turns = _turns[:-1]
                if _turns[0]['speaker'] == 'system':
                    _turns = _turns[1:]

                split = ''
                if random.random() < dev_ratio:
                    split = 'dev'
                else:
                    split = 'train'

                for _turn in (_turns):
                    _utt = _turn['utterance']

                    if _turn['speaker'] == 'user':
                        _history.insert(0, _utt)

                        if len(_history) == num_turns:
                            if _ds in ['multiwoz25', 'schema'] and _d['data_split'] == 'train':
                                str_to_train += '\t'.join(_history) + '\n'
                            elif _ds in ['multiwoz25', 'schema'] and _d['data_split'] in ['val', 'test', 'validate']:
                                str_to_dev += '\t'.join(_history) + '\n'
                            elif split == 'dev':
                                str_to_dev += '\t'.join(_history) + '\n'
                            else:
                                str_to_train += '\t'.join(_history) + '\n'
                            lines_written += 1
                            _history = []
                    else:
                        assert len(_history) < num_turns
                        _history.insert(0, _utt)

                if len(_history) > 0:
                    if _ds in ['multiwoz25', 'schema'] and _d['data_split'] == 'train':
                        str_to_train += '\t'.join(_history) + '\n'
                    elif _ds in ['multiwoz25', 'schema'] and _d['data_split'] in ['val', 'test', 'validate']:
                        str_to_dev += '\t'.join(_history) + '\n'
                    elif random.random() < dev_ratio:
                        str_to_dev += '\t'.join(_history) + '\n'
                    else:
                        str_to_train += '\t'.join(_history) + '\n'
                    lines_written += 1
                    _history = []

            else:
                raise NotImplementedError

        train_writer.write(str_to_train)
        dev_writer.write(str_to_dev)
        train_writer.close()
        dev_writer.close()
        logging.info('{} dataset preprocessed, written {} examples'.format(_ds, lines_written))


def count_turns(dataset_names):
    import json

    if not os.path.isdir('dataset_count'):
        os.mkdir('./dataset_count')

    all_turn_count = [0 for _ in range(60)]
    all_token_count = [0 for _ in range(50)]

    for _ds in dataset_names:
        turn_count = [0 for _ in range(60)]
        token_count = [0 for _ in range(50)]

        _dataset_dir = os.path.join(data_dir, _ds+'_ptm') if os.path.isdir(os.path.join(data_dir, _ds+'_ptm')) else os.path.join(data_dir, _ds)
        _data_zipfile = os.path.join(_dataset_dir, 'data.zip')
        _dialogs, _ = eval(_ds + '_preprocess').preprocess()

        for _d in _dialogs:
            _turns = _d['turns']
            if _turns[-1]['speaker'] == 'system':  # make sure user speak last
                _turns = _turns[:-1]
            if _turns[0]['speaker'] == 'system':
                _turns = _turns[1:]
            num_turn = len(_turns) + 1
            if num_turn >= len(turn_count):
                num_turn = len(turn_count) - 1
            turn_count[num_turn] += 1
            all_turn_count[num_turn] += 1

            _num_token = 0
            for _t in _turns:
                _num_token += len(_t['utterance'])
            _token_idx = _num_token // 50
            _token_idx = 49 if _token_idx > 49 else _token_idx
            token_count[_token_idx] += 1
            all_token_count[_token_idx] += 1

        json.dump(turn_count, open('./dataset_count/turns_count_{}.json'.format(_ds), 'w'), indent=4)
        json.dump(token_count, open('./dataset_count/tokens_count_{}.json'.format(_ds), 'w'), indent=4)

    json.dump(all_turn_count, open('./dataset_count/turns_count_all.json', 'w'), indent=4)
    json.dump(all_token_count, open('./dataset_count/tokens_count_all.json', 'w'), indent=4)


if __name__== '__main__':
    # for statistics
    preprocess(dataset_names, mode='full_dialog', out_dev_filename='all_but_mwoz_fulldialog_dev_8.18.txt', out_train_filename='all_but_mwoz_fulldialog_train_8.18.txt')
    # count_turns(dataset_names)