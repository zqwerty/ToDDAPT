import json
import os
from tqdm import tqdm
import copy
import zipfile

from convlab2.util.file_util import read_zipped_json, write_zipped_json

self_dir = os.path.dirname(os.path.abspath(__file__))

ontology = {
    'domains': {},
    'intents': {},
    'binary_dialogue_act': [],
    'state': {}
}

init_sess = {
    "dataset": "rnnlg",
    "data_split": "train",
    "dialogue_id": "",
    # "original_id": "",
    "domains": [
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

id_count = 0

def parse_dialogact(act_str, domain, uttr):
    act_str = act_str.replace('?', '')
    if '|' in act_str:
        acts = act_str.strip().split('|')
    else:
        acts = [act_str]
    binary_dic = []
    nc_dic = []
    for act in acts:
        if act is None or act.strip() == '' or act == 'Unrecognizable': continue
        try:
            assert '(' in act and ')' in act
        except:
            print('Invalid act: ['+act+']')
            exit()
        intent, a = act.split('(', 1)
        svs = a.split(')', 1)[0]
        if ';' in svs:
            svs = svs.split(';')
        else:
            svs = [svs]
        for sv in svs:
            if len(sv) == 0:
                binary_dic.append({
                    "intent": intent,
                    "domain": domain,
                    "slot": "",
                    "value": ""
                })
                continue
            if '=' in sv:
                s, v = sv.split('=')
                if v.startswith("'") and v.endswith("'"):
                    v = v[1:-1]
                if v in ['yes', 'no', 'true', 'false', 'dontcare', 'none', 'dont_care']:
                    start, end = 0, 0
                elif v in uttr:
                    start = uttr.find(v)
                    end = start + len(v)
                else:
                    print(f'Not found [{v}]', uttr, act)
                    start, end = 0, 0
                if v in ['yes', 'no', 'true', 'false']:
                    binary_dic.append(
                        {
                            "intent": intent,
                            "domain": domain,
                            "slot": s,
                            "value": v
                        }
                    )
                else:
                    if end == 0:
                        nc_dic.append(
                            {
                                "intent": intent,
                                "domain": domain,
                                "slot": s,
                                "value": v
                            }
                        )
                    else:
                        nc_dic.append(
                            {
                                "intent": intent,
                                "domain": domain,
                                "slot": s,
                                "value": v,
                                "start": start,
                                "end": end
                            }
                        )
            else:
                s, v = sv, ""
                binary_dic.append(
                    {
                        "intent": intent,
                        "domain": domain,
                        "slot": s,
                        "value": "",
                    }
                )
    log_ontology(binary_dic)
    log_ontology(nc_dic)
    return binary_dic, nc_dic


def read_sclstm_data():
    files = ['original_data/sfxhotel/train+valid+test.json', 'original_data/sfxrestaurant/train+valid+test.json']
    new_data = []
    global id_count
    for path in files:
        if path == files[0]:
            domain = 'hotel'
        else:
            domain = 'restaurant'
        content_str = ''
        with open(path) as f:
            for line in f.readlines():
                line = line.strip()
                if line[0] == '#': continue
                content_str += line
        data = json.loads(content_str)
        for sess in data:
            new_sess = copy.deepcopy(init_sess)
            new_sess['domains'] = [domain]
            new_sess['original_id'] = sess['id']
            new_sess['dialogue_id'] = 'rnnlg_' + str(id_count)
            id_count += 1
            turns = sess['dial']
            utt_idx = 0
            for i, turn in enumerate(turns):
                new_turn1 = copy.deepcopy(init_turn)
                new_turn2 = copy.deepcopy(init_turn)
                sys_turn = turn['S']
                user_turn = turn['U']
                new_turn1['speaker'] = 'system'
                new_turn2['speaker'] = 'user'
                new_turn1['utterance'] = sys_turn['base']
                new_turn2['utterance'] = user_turn['hyp']
                binary, nc = parse_dialogact(sys_turn['dact'], domain, new_turn1['utterance'])
                new_turn1['dialogue_act']['binary'] = binary
                new_turn1['dialogue_act']['non-categorical'] = nc

                binary, nc = parse_dialogact(user_turn['dact'], domain, new_turn2['utterance'])
                new_turn2['dialogue_act']['binary'] = binary
                new_turn2['dialogue_act']['non-categorical'] = nc

                if i == 0:
                    new_turn2['utt_idx'] = utt_idx
                    utt_idx += 1
                    new_sess['turns'] += [new_turn2]
                else:
                    new_turn1['utt_idx'] = utt_idx
                    utt_idx += 1
                    new_turn2['utt_idx'] = utt_idx
                    utt_idx += 1
                    new_sess['turns'] += [new_turn1, new_turn2]
            new_data.append(new_sess)

    return new_data


def read_laptop_tv_data():
    domains = ['laptop', 'tv']
    splits = ['train', 'valid', 'test']

    all_data = []
    global id_count
    for domain in domains:
        for split in splits:
            path = os.path.join('original_data', domain, split+'.json')
            data = json.load(open(path))
            for da, a, b in data:
                new_sess = copy.deepcopy(init_sess)
                new_sess['domains'] = [domain]
                new_sess['dialogue_id'] = 'rnnlg_' + str(id_count)
                id_count += 1
                new_turn = copy.deepcopy(init_turn)
                new_turn['utterance'] = a
                binary, nc = parse_dialogact(da, domain ,a)
                new_turn['dialogue_act']['binary'] = binary
                new_turn['dialogue_act']['non-categorical'] = nc
                new_sess['turns'] = [new_turn]
                all_data.append(new_sess)
    return all_data

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


def reformat_data():
    global ontology
    data = read_sclstm_data()# + read_laptop_tv_data()
    json.dump(data, open('data.json', 'w+'), indent=2)
    write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
    os.remove('data.json')
    # save ontology json
    json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w+'), indent=2)


def preprocess():
    if True:
    # if not os.path.exists(os.path.join(self_dir, './data.zip')):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'original_data.zip')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            if not os.path.exists('./original_data'):
                os.makedirs('./original_data')
            zip_ref.extractall('./original_data')
        reformat_data()
    processed_dialogue = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
    return processed_dialogue, None



if __name__ == '__main__':
    preprocess()
    # stat(get_data())  # 0.9525666323630767, 0.9905733559834641(dontcare)
    # reformated_data = reformat_rnnlg(get_data())
    # json.dump(reformated_data, open('all_data.json', 'w+'), indent=4)
