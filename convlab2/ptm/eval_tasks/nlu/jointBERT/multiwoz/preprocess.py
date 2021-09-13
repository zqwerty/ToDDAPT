import json
import os
import zipfile
import sys
from collections import Counter
from transformers import BertTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.getcwd()))))))))
print(sys.path)
from convlab2.util.file_util import read_zipped_json
from convlab2.ptm.eval_tasks.nlu.jointBERT.dataloader import generate_bio_tag


def da2triples(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        for slot, value in svs:
            triples.append([intent, slot, value])
    return triples


def preprocess(name, mode, context_size, tokenizer):
    assert mode == 'all' or mode == 'usr' or mode == 'sys'
    assert name in ['multiwoz', 'multiwoz23', 'multiwoz25']
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../../../data/'+name)
    processed_data_dir = os.path.join(cur_dir, name+'/{}_data'.format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_da = []
    all_intent = []
    all_tag = []
    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            context = []
            for is_sys, turn in enumerate(sess['log']):
                if mode == 'usr' and is_sys % 2 == 1:
                    context.append(turn['text'])
                    continue
                elif mode == 'sys' and is_sys % 2 == 0:
                    context.append(turn['text'])
                    continue

                role = 'sys' if is_sys % 2 == 1 else 'user'

                split_tokens = turn["text"].split()
                utterance = ' '.join(split_tokens)
                char_spans = []
                dialog_act = {}
                for span in turn["span_info"]:
                    name = span[0] + "+" + span[1]
                    tok_start, tok_end = span[3], span[4]
                    char_start = len(' '.join(split_tokens[:tok_start]))
                    if char_start > 0:
                        char_start += 1
                    char_end = len(' '.join(split_tokens[:tok_end+1]))
                    assert utterance[char_start:char_end] == ' '.join(split_tokens[tok_start:tok_end+1])
                    char_spans.append((char_start, char_end, name))
                    if span[0] not in dialog_act:
                        dialog_act[span[0]] = []
                    dialog_act[span[0]].append([span[1], " ".join(split_tokens[span[3]: span[4] + 1])])

                bert_tokens, tags, _, ori_tokens, bert_token2ori_token = generate_bio_tag(utterance, char_spans, tokenizer)

                intents = []
                for dacts in turn["dialog_act"]:
                    for dact in turn["dialog_act"][dacts]:
                        if dacts not in dialog_act or dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                            if dact[1] in ["none", "?", "yes", "no", "do nt care", "do n't care", "dontcare"]:
                                intents.append(dacts + "+" + dact[0] + "*" + dact[1])

                processed_data[key].append([role, bert_tokens, tags, intents, da2triples(turn["dialog_act"]), context[-context_size:], ori_tokens, bert_token2ori_token])
                all_da += [da for da in turn['dialog_act']]
                all_intent += intents
                all_tag += tags

                context.append(turn['text'])

        all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
        all_intent = [x[0] for x in dict(Counter(all_intent)).items() if x[1]]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]
        all_tag.remove('X')

        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key], open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w'), indent=2)

    print('dialog act num:', len(all_da))
    print('sentence label num:', len(all_intent))
    print('tag num:', len(all_tag))
    json.dump(all_da, open(os.path.join(processed_data_dir, 'all_act.json'), 'w'), indent=2)
    json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w'), indent=2)
    json.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.json'), 'w'), indent=2)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for name in ['multiwoz23']:
        for mode in ['all']:
            preprocess(name, mode, context_size=3, tokenizer=tokenizer)
