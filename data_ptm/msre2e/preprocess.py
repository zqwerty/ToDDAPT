import json
from zipfile import ZipFile, ZIP_DEFLATED

domains = ['movie', 'restaurant', 'taxi']
dataset = 'msre2e'


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
    with ZipFile('original_data.zip') as zipf:
        for domain in domains:
            lines = zipf.open(f'{domain}_all.tsv').readlines()
            for line in lines[1:]:
                line = line.decode().split('\t')
                (session_id, msg_id, _, msg_from, msg), dialog_acts = line[:5], line[5:]
                session_id, msg_id = map(int, [session_id, msg_id])
                speaker = 'system' if msg_from == 'agent' else msg_from
                if msg_id == 1:
                    turns = []
                    data.append({
                        'dataset': dataset,
                        'data_split': 'train',
                        'dialogue_id': f'{dataset}_{len(data)}',
                        'original_id': session_id,
                        'domains': [domain],
                        'turns': turns
                    })
                elif speaker == turn['speaker']:
                    turn['utterance'] += f' {msg}'
                    continue
                turn = {
                    'speaker': speaker,
                    'utterance': msg,
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
                if len(turns) > 0 or speaker == 'user':
                    turns.append(turn)

    for dialog in data:
        turns = dialog['turns']
        if turns[-1]['speaker'] == 'system':
            turns.pop()

    json.dump(ontology, open('ontology.json', 'w'), indent=4, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zipf:
        zipf.open('data.json', 'w').write(json.dumps(data, indent=4, ensure_ascii=False).encode())


if __name__ == '__main__':
    preprocess()
