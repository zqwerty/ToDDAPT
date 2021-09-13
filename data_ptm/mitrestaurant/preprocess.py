import json
import os
import copy
import zipfile

from convlab2.util.file_util import read_zipped_json, write_zipped_json

self_dir = os.path.dirname(os.path.abspath(__file__))

init_sess = {
    "dataset": "mitrestaurant",
    "data_split": "all",
    "dialogue_id": "",
    # "original_id": "",
    "domains": [
        "restaurant"
    ],
    "turns": []
}

init_turn = {
    "speaker": "",  # user or system
    "utterance": "",
    "utt_idx": 0,
    "dialogue_act": {
        "binary": [
        ],
        "categorical": [
        ],
        "non-categorical": [
        ]
    },
    "state": {},
    "state_update": {
        "categorical": [],
        "non-categorical": []
    }
}

ontology = {
    'domains': {},
    'intents': {},
    'binary_dialogue_act': [],
    "state": {}
}

def log_ontology(acts):
    global ontology
    for item in acts:
        intent, domain, slot, value = item['intent'], item['domain'], item['slot'], item['value']
        if domain not in ontology['domains']:
            ontology['domains'][domain] = {'description': "", 'slots': {}}
        if slot not in ontology['domains'][domain]['slots']:
            ontology['domains'][domain]['slots'][slot] = {
                'description': '',
                'is_categorical': False,
                'possible_values': []
            }
        if intent not in ontology['intents']:
            ontology['intents'][intent] = {
                "description": ''
            }
        pairs4 = {
            "intent": intent,
            "domain": domain,
            "slot": slot,
            "value": "" if value not in ['yes', 'no', 'true', 'false', 'dontcare', 'none', 'dont_care'] else value
        }
        def is_in(a, b):
            get = False
            for item in b:
                all_same = True
                for k in a.keys():
                    if a[k] != item[k]:
                        all_same = False
                        break
                if all_same:
                    get = True
                    break
            return get
        if not is_in(pairs4, ontology['binary_dialogue_act']):
            ontology['binary_dialogue_act'].append(pairs4)


def reformat_data(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'original_data.zip')):
    global ontology
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        if not os.path.exists('./original_data'):
            os.makedirs('./original_data')
        zip_ref.extractall('./original_data')

    files = [
        'original_data/restauranttest.bio.txt',
        'original_data/restauranttrain.bio.txt',
    ]
    # print(files)
    final_data = []
    id_count = 1
    ontology['domains']['restaurant'] = {'description': None, 'slots': {}}
    for file in files:
        f = open(file)
        content = f.read()
        content = content.strip()
        sesses = content.split('\n\n')
        # print(sesses[4])
        for sess in sesses:
            lines = [item.strip() for item in sess.strip().split('\n')]
            sess = copy.deepcopy(init_sess)
            sess['data_split'] = 'train' if 'train' in file else 'test'
            sess['dialogue_id'] = 'mitrestaurant_' + str(id_count)
            id_count += 1
            # sess['original_id'] = None
            # sess['domains'] = []
            curr_turn = copy.deepcopy(init_turn)
            curr_turn['speaker'] = 'user'

            word_seq, tag_seq = [], []
            for tag_word in lines:
                tag, word = tag_word.split('\t')
                word_seq.append(word)
                tag_seq.append(tag)
            da = []
            i = 0
            utt = ''
            while i < len(tag_seq):
                tag = tag_seq[i]
                if tag.startswith('B'):
                    slot = tag[2:]
                    value = word_seq[i]
                    start = len(utt)
                    end = start + len(value)
                    utt += value + ' '
                    j = i + 1
                    while j < len(tag_seq):
                        if tag_seq[j].startswith('I') and tag_seq[j][2:] == slot:
                            value += ' ' + word_seq[j]
                            end += len(word_seq[j]) + 1
                            utt += word_seq[j] + ' '
                            i += 1
                            j += 1
                        else:
                            break
                    da.append({
                        "intent": "inform",
                        "domain": "restaurant",
                        "slot": slot.lower(),
                        "value": value,
                        "start": start,
                        "end": end
                    })
                else:
                    assert tag == 'O'
                    utt += word_seq[i] + ' '
                i += 1
            utt = utt.strip()
            curr_turn['dialogue_act']['non-categorical'] = da
            curr_turn['utterance'] = utt

            log_ontology(curr_turn['dialogue_act']['non-categorical'])
            sess['turns'] = [curr_turn]
            final_data.append(sess)

    json.dump(final_data, open('data.json', 'w+'), indent=2)
    write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
    os.remove('data.json')
    # save ontology json
    json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w+'), indent=2)

def preprocess():
    # if True:
    if not os.path.exists(os.path.join(self_dir, './data.zip')):
        reformat_data()
    processed_dialogue = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
    return processed_dialogue, None


if __name__ == '__main__':
    preprocess()
