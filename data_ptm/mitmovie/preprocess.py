import zipfile
import json
import os
from pprint import pprint
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
from convlab2.util.file_util import read_zipped_json, write_zipped_json
import logging
logging.basicConfig(level=logging.INFO)
self_dir = os.path.dirname(os.path.abspath(__file__))


def get_des(name):
    return {
        'actor': 'Name of an actor starring in the movie',
        'year': 'Year in which the movie was released',
        'title': 'Name of the movie',
        'genre': 'Type of the movie',
        'director': 'Director of the movie',
        'song': 'Song used in the movie',
        'plot': 'Plot in the movie',
        'review': 'What do people think about the movie',
        'character': 'Character in the movie',
        'rating': "Rate a film's suitability for certain audiences, based on its content",
        'ratings_average': 'Average user rating of the movie',
        'trailer': 'Trailer of the movie',
        'opinion': "User's opinion about the movie",
        'award': 'Award of the movie',
        'origin': 'The origin of the movie',
        'soundtrack': 'Soundtrack of the movie',
        'relationship': 'Relationship between two movie',
        'character_name': 'Character in the movie',
        'quote': 'Quote in the movie',
        'inform': 'Inform the value for a slot',
        'movie': 'Provide information about the movie that requested by the user'
    }[name]


def preprocess():
    processed_dialogue = []
    slots = Counter()
    original_zipped_path = os.path.join(self_dir, 'original_data.zip')
    new_dir = os.path.join(self_dir, 'original_data')
    if not os.path.exists(original_zipped_path):
        raise FileNotFoundError(original_zipped_path)
    # if True:
    if not os.path.exists(os.path.join(self_dir, 'data.zip')) or not os.path.exists(os.path.join(self_dir, 'ontology.json')):
        print('unzip to', new_dir)
        print('This may take several minutes')
        archive = zipfile.ZipFile(original_zipped_path, 'r')
        archive.extractall(new_dir)
        cnt = 1
        for dataset in ['eng', 'trivia10k13']:
            for data_split in ['train', 'test']:
                dataset_name = 'mitmovie'
                filepath = os.path.join(new_dir, dataset+data_split+'.bio')
                f = open(filepath)
                sentences = f.read().strip().split('\n\n')
                for sen in tqdm(sentences, desc='processing mit-movie-{}-{}'.format(dataset, data_split)):
                    dialogue = {
                        "dataset": dataset_name,
                        "data_split": data_split,
                        "dialogue_id": dataset_name+'_'+str(cnt),
                        "domains": ['movie']
                    }
                    cnt += 1
                    word_seq, tag_seq = [], []
                    for tag_word in sen.split('\n'):
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
                            end = start+len(value)
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
                                "domain": "movie",
                                "slot": slot.lower(),
                                "value": value,
                                "start": start,
                                "end": end
                            })
                            slots.update([slot.lower()])
                        else:
                            assert tag == 'O'
                            utt += word_seq[i] + ' '
                        i += 1
                    utt = utt.strip()
                    turn = {
                        'speaker': 'user',
                        'utterance': utt,
                        'utt_idx': 0,
                        'dialogue_act': {
                            'binary': [],
                            'categorical': [],
                            'non-categorical': da,
                        },
                        "state": {},
                        "state_update": {
                            "categorical": [],
                            "non-categorical": []
                        }
                    }
                    dialogue['turns'] = [turn]
                    processed_dialogue.append(deepcopy(dialogue))

        slots = {s: {'description': get_des(s), "is_categorical": False, "possible_values": []} for s in slots}
        ontology = {
            'domains': {'movie':{'description': get_des('movie'), 'slots': slots}},
            'intents': {'inform':{'description': get_des('inform')}},
            'binary_dialogue_act': [],
            'state': {}
        }
        json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w'), indent=2)
        json.dump(processed_dialogue, open('data.json', 'w'), indent=2)
        write_zipped_json(os.path.join(self_dir,'data.zip'), 'data.json')
        os.remove('data.json')
    else:
        # read from file
        processed_dialogue = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        ontology = json.load(open(os.path.join(self_dir, 'ontology.json')))
    return processed_dialogue, ontology


if __name__ == '__main__':
    preprocess()
