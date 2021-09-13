import json
import os
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy

special_values = ['dontcare']
dataset = 'm2m'
self_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(self_dir)), 'data')
# origin_data_dir = os.path.join(DATA_PATH, dataset)
origin_data_dir = self_dir

descriptions = {
    "domains": {
        "movie": {
            "description": "query movie information and book movie tickets",
            "slots": {
                "time": "performance time of the movie",
                "num_tickets": "the number of movie tickets",
                "movie": "name of the movie",
                "date": "performance date of the movie",
                "theatre_name": "name of the theatre where the movie is performed"
            }
        },
        "restaurant": {
            "description": "restaurant search and reservation service",
            "slots": {
                "num_people": "the number of reservation",
                "restaurant_name": "name of the restaurant",
                "date": "date of reservation",
                "time": "time of reservation",
                "meal": "meal of reservation",
                "location": "location of the restaurant",
                "price_range": "price range of the restaurant",
                "category": "category of the restaurant",
                "rating": "rating of the restaurant"
            }
        }
    },
    "intents": {
        "INFORM": "inform the value for a slot",
        "REQUEST": "request the value of a slot.",
        "CONFIRM": "confirm the value of a slot before making a transactional service call",
        "AFFIRM": "agree to the system's proposition",
        "NOTIFY_SUCCESS": "inform the user that their request was successful",
        "THANK_YOU": "thank the system",
        "GOOD_BYE": "end the dialogue",
        "CANT_UNDERSTAND": "express not understanding what the system says",
        "GREETING": "express greeting",
        "NOTIFY_FAILURE": "inform the user that their request was unsuccessful",
        "NEGATE": "deny the system's proposal",
        "SELECT": "one option of the system's proposal",
        "OFFER": "offer a certain value for a slot to the user",
        "OTHER": "something not relevant to the current dialogue",
        "REQUEST_ALTS": "ask for other results instead of the ones offered by the system"
    }
}


def find_slot(values, slot_name):
    ret = []
    for value in values:
        if value['slot'] == slot_name:
            ret.append(value)
    return ret


# return span of value
def find_value(tokens, start_pos, values, value):
    for item in values:
        start = item['start']
        end = item['exclusive_end']
        if ' '.join(tokens[start:end]) == value:
            return start_pos[start], start_pos[end] - 1
    return None


def make_da(intent, domain='', slot='', value='', start=None, end=None):
    da = {
        'intent': intent,
        'domain': domain,
        'slot': slot,
        'value': value,
    }
    if start is not None:
        assert end is not None
        da['start'] = start
        da['end'] = end
    return da


def preprocess():
    domains = {}
    intents = {}
    empty_state = {}
    # bdas = set()
    data = []

    def add_intent(intent_name):
        if intent_name not in intents:
            intents[intent_name] = {
                'description': descriptions['intents'][intent_name]
            }
    # TODO: description of each domain, slot, intent
    with ZipFile(os.path.join(origin_data_dir, 'm2m.zip')) as zipfile:
        dialog_id = 0
        for suffix, domain_name in [('M', 'movie'), ('R', 'restaurant')]:
            empty_state[domain_name] = {}
            domain = domains[domain_name] = {
                'slots': {},
                'description': descriptions['domains'][domain_name]['description']
            }

            def add_slot(slot_name):
                if slot_name not in domain['slots']:
                    domain['slots'][slot_name] = {
                        'description': descriptions['domains'][domain_name]['slots'][slot_name],
                        'is_categorical': False,
                        'possible_values': [],
                    }

            for split in ['train', 'val', 'test']:
                for ori_dialog in json.load(zipfile.open(os.path.join(f'sim-{suffix}', f'{"dev" if split == "val" else split}.json'))):
                    turns = []
                    for turn_id, ori_turn in enumerate(ori_dialog['turns']):
                        # sigle domain state
                        state = {}
                        for item in ori_turn['dialogue_state']:
                            slot_name = item['slot']
                            add_slot(slot_name)
                            empty_state[domain_name][item['slot']] = ""
                            state[slot_name] = item['value']
                        state = {
                            domain_name: state
                        }

                        if 'system_acts' in ori_turn:
                            bdas = []
                            das = []
                            tokens = ori_turn['system_utterance']['tokens']
                            values = ori_turn['system_utterance']['slots']
                            start_pos = [0]
                            for token in tokens:
                                start_pos.append(start_pos[-1] + len(token) + 1)

                            for system_act in ori_turn['system_acts']:
                                intent_name = system_act['type']
                                add_intent(intent_name)
                                if 'slot' in system_act:
                                    slot_name = system_act['slot']
                                    if 'value' in system_act:
                                        value = system_act['value']
                                        add_slot(slot_name)
                                        if value in special_values:
                                            das.append(make_da(intent_name, domain_name, slot_name, value))
                                        else:
                                            start, end = find_value(ori_turn['system_utterance']['tokens'], start_pos, values, value)
                                            das.append(make_da(intent_name, domain_name, slot_name, value, start, end))
                                    else:
                                        bdas.append(make_da(intent_name, domain_name, slot_name, ''))
                                else:
                                    bdas.append(make_da(intent_name, domain_name, '', ''))

                                # for item in ori_turn['system_utterance']['slots']:
                                #     add_slot(item['slot'])
                            turns.append({
                                'speaker': 'system',
                                'utterance': ori_turn['system_utterance']['text'],
                                'utt_idx': turn_id * 2 - 1,
                                'dialogue_act': {
                                    'binary': bdas,
                                    'categorical': [],
                                    'non-categorical': das,
                                }
                            })

                        # process user-side dialog act
                        # this work is hard
                        das = []
                        bdas = []
                        values = ori_turn['user_utterance']['slots']
                        tokens = ori_turn['user_utterance']['tokens']
                        start_pos = [0]
                        for token in tokens:
                            start_pos.append(start_pos[-1] + len(token) + 1)
                        for user_act in ori_turn['user_acts']:
                            intent_name = user_act['type']
                            add_intent(intent_name)
                            if intent_name in ['GREETING', 'THANK_YOU', 'GOOD_BYE', 'CANT_UNDERSTAND', 'OTHER']:
                                bdas.append(make_da(intent_name))
                                assert 'slot' not in user_act
                                continue
                            if 'slot' in user_act:
                                slot_name = user_act['slot']
                                if values:
                                    value = find_slot(values, slot_name)
                                    if value:
                                        assert len(value) == 1
                                        value = value[0]
                                        start = value['start']
                                        end = value['exclusive_end']
                                        add_slot(slot_name)
                                        das.append(make_da(intent_name, domain_name, slot_name, ' '.join(tokens[start:end]), start_pos[start], start_pos[end] - 1))
                                    else:
                                        bdas.append(make_da(intent_name, domain_name, slot_name))
                                else:
                                    bdas.append(make_da(intent_name, domain_name, slot_name))
                            else:
                                if intent_name in ['NEGATE', 'AFFIRM', 'REQUEST_ALTS']:
                                    bdas.append(make_da(intent_name, domain_name))
                                else:
                                    assert intent_name == 'INFORM'
                                    if values:
                                        for value in values:
                                            slot_name = value['slot']
                                            start = value['start']
                                            end = value['exclusive_end']
                                            add_slot(slot_name)
                                            das.append(make_da(intent_name, domain_name, slot_name, ' '.join(tokens[start:end]), start_pos[start], start_pos[end] - 1))
                                    else:
                                        last_turn = turns[-1]
                                        last_all_da = last_turn['dialogue_act']
                                        last_bdas = last_all_da['binary']
                                        if last_bdas:
                                            for bda in last_bdas:
                                                assert bda['intent'] == 'REQUEST'
                                                das.append(make_da(intent_name, domain_name, bda['slot'], 'dontcare'))
                                        else:
                                            last_das = last_all_da['non-categorical']
                                            assert last_das
                                            for da in last_das:
                                                last_intent = da['intent']
                                                if last_intent == 'CONFIRM':
                                                    das.append(make_da(intent_name, domain_name, bda['slot'], 'dontcare'))
                                                else:
                                                    assert last_intent in ['OFFER', 'NOTIFY_SUCCESS']
                                                    bdas.append(make_da(intent_name, domain_name, bda['slot']))


                        turns.append({
                            'speaker': 'user',
                            'utterance': ori_turn['user_utterance']['text'],
                            'utt_idx': turn_id * 2,
                            'dialogue_act': {
                                'binary': bdas,
                                'categorical': [],
                                'non-categorical': das,
                            },
                            'state': state
                        })

                    data.append({
                        "dataset": dataset,
                        "data_split": split,
                        "dialogue_id": f'{dataset}_{dialog_id}',
                        "original_id": ori_dialog['dialogue_id'],
                        "domains": [domain_name],
                        'turns': turns
                    })
                    dialog_id += 1

    bdas = set()
    for dialog in data:
        turns = dialog['turns']
        for turn_id, turn in enumerate(turns):
            for bda in turn['dialogue_act']['binary']:
                bdas.add(tuple(bda.values()))
            if turn_id % 2 == 0:
                # single domain
                domain_name = dialog['domains'][0]

                state = turn['state']
                for slot_name in empty_state[domain_name]:
                    if slot_name not in state[domain_name]:
                        state[domain_name][slot_name] = ""

                state = state[domain_name]
                last_state = turns[turn_id - 2]['state'][domain_name] if turn_id else deepcopy(empty_state[domain_name])
                state_update = []
                for slot_name in empty_state[domain_name]:
                    if last_state[slot_name] != state[slot_name]:
                        value = state[slot_name]
                        item = {
                            'domain': domain_name,
                            'slot': slot_name,
                            'value': value
                        }
                        if value not in special_values:
                            item['utt_idx'] = item['start'] = item['end'] = 0
                            for utt_idx in reversed(range(turn_id + 1)):
                                pos = turns[utt_idx]['utterance'].lower().find(value.lower())
                                if pos != -1:
                                    item['utt_idx'] = utt_idx
                                    item['start'] = pos
                                    item['end'] = pos + len(value)
                                    break

                        state_update.append(item)
                turn['state_update'] = {
                    'categorical': [],
                    'non-categorical': state_update
                }

    ontology =  {
        'domains': domains,
        'intents': intents,
        'binary_dialogue_act': [
            {
                'intent': intent,
                'domain': domain,
                'slot': slot,
                'value': value,
            } for intent, domain, slot, value in bdas
        ],
        'state': empty_state
    }

    json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w'), indent=4)
    json.dump(data, open('data.json', 'w'), indent=4)
    ZipFile(os.path.join(self_dir, 'data.zip'), 'w', ZIP_DEFLATED).write('data.json')
    os.remove('data.json')


if __name__ == '__main__':
    preprocess()
