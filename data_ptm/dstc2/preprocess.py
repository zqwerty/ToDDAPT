import json
import os
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from tarfile import TarFile
from tempfile import NamedTemporaryFile


dataset = 'dstc2'
self_dir = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(self_dir)), 'data')
origin_data_dir = self_dir


def preprocess():
    for filename in ['dstc2_traindev.tar.gz', 'dstc2_test.tar.gz']:
        TarFile.open(os.path.join(origin_data_dir, filename)).extractall()
    ori_ontology = json.load(open('scripts/config/ontology_dstc2.json'))
    ontology = {
        'domains': {},
        'intents': {},
        'binary_dialogue_act': [],
        'state': {}
    }
    os.rename('scripts/config/dstc2_dev.flist', 'scripts/config/dstc2_val.flist')
    dialog_id = 0
    data = []
    for split in ['train', 'val', 'test']:
        for line in open(f'scripts/config/dstc2_{split}.flist').readlines():
            path = line.strip()
            log = json.load(open(os.path.join('data', path, 'log.json')))
            label = json.load(open(os.path.join('data', path, 'label.json')))
            dialog = {
                "dataset": dataset,
                "data_split": split,
                "dialogue_id": f'{dataset}_{dialog_id}',
                "original_id": log['session-id'],
                "domains": [],
            }
            dialog_id += 1
            turns = []
            # ref: http://camdial.org/~mh521/dstc/downloads/handbook.pdf
            for turn_id, (log_turn, label_turn) in enumerate(zip(log['turns'], label['turns'])):
                # discard first system turn
                if turn_id:
                    turns.append({
                        'speaker': 'system',
                        'utt_idx': turn_id * 2 - 1,
                        'utterance': log_turn['output']['transcript'],
                        'dialogue_act': {
                            'categorical': [],
                            'non-categorical': [],
                            'binary': [],
                        },
                        'binary': [],
                    })
                turns.append({
                    'speaker': 'user',
                    'utt_idx': turn_id * 2,
                    'utterance': label_turn['transcription'],
                    'dialogue_act': {
                        'categorical': [],
                        'non-categorical': [],
                        'binary': [],
                    },
                    'state': {},
                    'state_update': {
                        'categorical': [],
                        'non-categorical': []
                    }
                })
            dialog['turns'] = turns
            data.append(dialog)

    shutil.rmtree('data')
    shutil.rmtree('scripts')
    json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w'), indent=4)
    json.dump(data, open('data.json', 'w'), indent=4)
    ZipFile(os.path.join(self_dir, 'data.zip'), 'w', ZIP_DEFLATED).write('data.json')
    os.remove('data.json')


if __name__ == '__main__':
    preprocess()
