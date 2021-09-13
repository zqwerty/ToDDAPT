import json
from zipfile import ZipFile, ZIP_DEFLATED

dataset = 'smd'
domains = ['navigate', 'schedule', 'weather']


def preprocess():
    ontology = {
        'domains': {
            domain: {
                'description': "",
                'slots': {}
            }
            for domain in domains
        },
        'intents': {},
        'binary_dialogue_act': {},
        'state': {},
    }
    data = []
    for split in ['train', 'dev', 'test']:
        raw_data = json.load(ZipFile('original_data.zip').open(f'kvret_{split}_public.json'))
        if split == 'dev':
            split = 'val'
        for ori_dialog in raw_data:
            turns = []
            domain = ori_dialog['scenario']['task']['intent']
            dialog = {
                'dataset': dataset,
                'data_split': split,
                'dialogue_id': f'{dataset}_{len(data)}',
                'original_id': "",
                'domains': [domain],
                'turns': turns
            }
            for ori_turn in ori_dialog['dialogue']:
                speaker = {
                    'driver': 'user',
                    'assistant': 'system',
                }[ori_turn['turn']]
                utt = ori_turn['data']['utterance']
                if turns and speaker == turn['speaker']:
                    if utt != turn['utterance']:
                        turn['utterance'] += f' {utt}'
                    continue
                else:
                    turn = {
                        'speaker': speaker,
                        'utterance': utt,
                        'utt_idx': len(turns),
                        'dialogue_act': {
                            'binary': [],
                            'categorical': [],
                            'non-categorical': [],
                        }
                    }
                if speaker == 'user':
                    turn['state'] = {}
                    turn['state_update'] = {}
                turns.append(turn)

            if turns:
                data.append(dialog)

    for dialog in data:
        turns = dialog['turns']
        if turns[-1]['speaker'] == 'system':
            turns.pop()

    json.dump(ontology, open('ontology.json', 'w'), indent=4, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zipf:
        zipf.open('data.json', 'w').write(json.dumps(data, indent=4, ensure_ascii=False).encode())


if __name__ == '__main__':
    preprocess()
