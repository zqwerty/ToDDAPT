import json
from zipfile import ZipFile, ZIP_DEFLATED

dataset = 'oos'


def preprocess():
    intents = {}
    ontology = {
        'domains': {},
        'intents': intents,
        'binary_dialogue_act': [],
        'state': {},
    }
    ori_data = json.load(ZipFile('original_data.zip').open(f'data_full.json'))
    data = []

    for split in ['train', 'val', 'test']:
        for utt, intent in ori_data[f'{split}'] + ori_data[f'oos_{split}']:
            binary = {
                'intent': intent,
                'domain': '',
                'slot': '',
                'value': '',
            }
            if intent not in intents:
                ontology['intents'][intent] = {'description': ''}
                ontology['binary_dialogue_act'].append(binary)
            data.append({
                'dataset': dataset,
                'data_split': split,
                'dialogue_id': f'{dataset}_{len(data)}',
                'original_id': "",
                'domains': [],
                'turns': [{
                    'speaker': 'user',
                    'utterance': utt,
                    'utt_idx': 0,
                    'dialogue_act': {
                        'binary': [binary],
                        'categorical': [],
                        'non-categorical': [],
                    },
                    'state': {},
                    'state_update': {},
                }]
            })

    json.dump(ontology, open('ontology.json', 'w'), indent=4, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zipf:
        zipf.open('data.json', 'w').write(json.dumps(data, indent=4, ensure_ascii=False).encode())


if __name__ == '__main__':
    preprocess()
