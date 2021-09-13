import json
import os
import shutil
import zipfile
import csv
from tarfile import TarFile
from tempfile import NamedTemporaryFile

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(sys.path[-1])
from convlab2.util.file_util import read_zipped_json, write_zipped_json

self_dir = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(self_dir)), 'data')
origin_data_dir = self_dir
dataset = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
print('preprocessing', dataset)


def convert_format(idx, utt, dataset, split):
    ptm_dial = {
        'dataset': dataset,
        'data_split': split,
        'dialogue_id': idx,
        'domains': [],
        'turns': [
            {
                'speaker': 'user',
                'utterance': utt,
                'utt_idx': 0,
                'dialogue_act': {
                    'categorical': [],
                    'non-categorical': [],
                    'binary': [],
                }
            }
        ]
    }
    return ptm_dial


def preprocess():
    original_zipped_path = os.path.join(self_dir, 'original_data.zip')
    if not os.path.exists(original_zipped_path):
        raise FileNotFoundError(original_zipped_path)

    if not os.path.exists(os.path.join(self_dir, 'data.zip')) or not os.path.exists(
            os.path.join(self_dir, 'ontology.json')):

        archive = zipfile.ZipFile(original_zipped_path, 'r')
        archive.extractall(os.path.join(self_dir, 'original_data'))

        ptm_data = []
        if dataset in ['clinc']:
            for split in ['train', 'val', 'test']:
                data = []
                with open(os.path.join(self_dir, 'original_data', '{}.csv'.format(split)), encoding='utf-8') as f:
                    for row in f:
                        data.append(row.split('","')[0])
                data = data[1:]
                data = [d.replace('"', '') for d in data]
                # print(dataset, data[:10])
                ptm_data.extend(
                    [convert_format('{}_{}'.format(dataset, idx), utt, dataset, split) for idx, utt in enumerate(data)])

        if dataset in ['banking', 'hwu']:
            for split in ['train', 'val', 'test']:
                data = []
                with open(os.path.join(self_dir, 'original_data', '{}.csv'.format(split)), encoding='utf-8') as f:
                    for row in f:
                        data.append(' '.join(row.split(',')[:-1]))
                data = data[1:]
                data = [d.replace('"', '') for d in data]
                ptm_data.extend(
                    [convert_format('{}_{}'.format(dataset, idx), utt, dataset, split) for idx, utt in enumerate(data)])

        if dataset in ['restaurant8k']:
            for split in ['train', 'val', 'test']:
                with open(os.path.join(self_dir, 'original_data', '{}.json'.format(split))) as f:
                    data = json.load(f)
                sentences = [d['userInput']['text'] for d in data]
                ptm_data.extend([convert_format('{}_{}'.format(dataset, idx), utt, dataset, split) for idx, utt in
                                 enumerate(sentences)])

        if dataset in ['top']:
            for split in ['train', 'eval', 'test']:
                sentences = []
                with open(os.path.join(self_dir, 'original_data', '{}.txt'.format(split))) as f:
                    for line in f:
                        sent = line.split('<=>')[0]
                        sent_list = sent.split()
                        sent_list = [t.split(':')[0] for t in sent_list]
                        sent = ' '.join(sent_list)
                        sentences.append(sent)
                if split == 'eval':
                    split = 'val'
                ptm_data.extend([convert_format('{}_{}'.format(dataset, idx), utt, dataset, split) for idx, utt in
                                 enumerate(sentences)])

        ont = {
            'domains': {},
            'intents': {},
            'binary_dialogue_act': [],
            'state': {},
        }

        # debug
        # print(json.dumps(ptm_data[:10], indent=4))

        json.dump(ptm_data, open('./data.json', 'w'), indent=4)
        json.dump(ont, open('./ontology.json', 'w'), indent=4)
        write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        os.remove('data.json')

    else:
        ptm_data = read_zipped_json(os.path.join(self_dir, './data.zip'), 'data.json')
        ont = json.load(open(os.path.join(self_dir, './ontology.json'), 'r'))

    return ptm_data, ont


if __name__ == '__main__':
    preprocess()
