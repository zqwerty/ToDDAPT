import json
import os
import copy
import zipfile

from convlab2.util.file_util import read_zipped_json, write_zipped_json

self_dir = os.path.dirname(os.path.abspath(__file__))

init_sess = {
    "dataset": "facebook",
    "data_split": "all",
    "dialogue_id": "",
    # "original_id": "",
    "domains": [
    ],
    "turns": []
}

init_turn = {
    "speaker": "user",  # user or system
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
    'state': {}
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
        # if not is_in(pairs4, ontology['binary_dialogue_act']):
        #     ontology['binary_dialogue_act'].append(pairs4)

def reformat_data(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'original_data.zip')):
    global ontology
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        if not os.path.exists('./original_data'):
            os.makedirs('./original_data')
        zip_ref.extractall('./original_data')

    files = [
        'original_data/en/test-en.conllu',
        'original_data/en/eval-en.conllu',
        'original_data/en/train-en.conllu'
    ]
    # print(files)
    final_data = []
    id_count = 1
    for file in files:
        f = open(file)
        content = f.read()
        content = content.strip()
        sesses = content.split('\n\n')
        # print(sesses[4])
        for sess in sesses:
            lines = [item.strip() for item in sess.strip().split('\n')]
            sess = copy.deepcopy(init_sess)
            sess['data_split'] = file.split('-')[0].split('/')[-1]
            if sess['data_split'] == 'eval':
                sess['data_split'] = 'val'
            sess['dialogue_id'] = 'facebook_' + str(id_count)
            id_count += 1
            # sess['original_id'] = None
            curr_turn = copy.deepcopy(init_turn)

            turn_intent = None
            turn_domain = None
            for line in lines:
                if not line.startswith('#'): continue
                line = line[2:]
                if line.startswith('text'):
                    uttr = line[6:]
                    # curr_turn['speaker'] = None
                    curr_turn['utterance'] = uttr
                elif line.startswith('intent'):
                    line = line[8:].strip()
                    assert(',') not in line
                    domain, intent = line.split('/')
                    sess['domains'] = [domain]
                    turn_domain = domain
                    turn_intent = intent
                    curr_turn['dialogue_act']['binary'].append({
                        'intent': turn_intent,
                        'domain': turn_domain,
                        'slot': '',
                        'value': '',
                    })
                    if curr_turn['dialogue_act']['binary'][-1] not in ontology['binary_dialogue_act']:
                        ontology['binary_dialogue_act'].append(curr_turn['dialogue_act']['binary'][-1])
                elif line.startswith('slots'):
                    line = line[7:].strip()
                    if line.strip() == '':
                        continue
                    items = line.split(',') if ',' in line else [line]
                    values = []
                    for sv in items:
                        try:
                            start, end, slot = sv.strip().split(':')
                        except:
                            print(line, items)
                        start, end = int(start), int(end)
                        value = curr_turn['utterance'][start:end]
                        values.append({
                            'intent': turn_intent,
                            'domain': turn_domain,
                            'slot': slot,
                            'value': value,
                            'start': start,
                            'end': end
                        })
                    curr_turn['dialogue_act']['non-categorical'] = values
                    log_ontology(values)
            sess['turns'] = [curr_turn]
            final_data.append(sess)

    json.dump(final_data, open('data.json', 'w+'), indent=2)
    write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
    os.remove('data.json')
    # save ontology json
    json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w+'), indent=2)

def preprocess():
    if True:
    # if not os.path.exists(os.path.join(self_dir, './data.zip')):
        reformat_data()
    processed_dialogue = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
    return processed_dialogue, None


if __name__ == '__main__':
    preprocess()
