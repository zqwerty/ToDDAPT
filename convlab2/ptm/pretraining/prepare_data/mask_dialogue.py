import json
import os
import copy
from tqdm import tqdm


def extract_spans_from_bio_tags(tags):
    """
    Extract spans indicated by BIO tags
    :param tags: list of bio tags
    :return: list of extracted spans [(start_idx, exclusive end_idx),...]
    """
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith('B'):
            start_idx = i
            while i + 1 < len(tags):
                if tags[i + 1].startswith('I') or tags[i + 1] == 'X':
                    i += 1
                else:
                    break
            end_idx = i + 1
            spans.append((start_idx, end_idx))
        i += 1
    return spans


if __name__ == '__main__':
    dataset_names = [
        'mitmovie',
        'mitrestaurant',
        # 'rnnlg',
        'facebook',
        'camrest',
        # 'woz',
        # 'kvret',
        # 'dstc2',
        'frames',
        'm2m',
        # 'mdc',
        'multiwoz21',
        'multiwoz25',
        'schema',
        # 'metalwoz',
        'taskmaster',
    ]
    data_dir = 'full_dialog'
    for dataset in dataset_names:
        print('load {} dataset:'.format(dataset))
        for phase in ['dev', 'train']:
            data = json.load(open(os.path.join(data_dir, dataset + '_data_{}.json'.format(phase))))
            print_example = True
            for dial in tqdm(data):
                new_dial = []
                for utt, utt_bio in zip(dial['dialogue'], dial['bio_tag']):
                    new_utt = copy.deepcopy(utt)
                    spans = extract_spans_from_bio_tags(utt_bio)
                    for i, (start, end) in enumerate(spans):
                        new_utt[start] = '[VALUE{}]'.format(i)
                        new_utt[start+1:end] = ['[DEL]'] * (end - start - 1)
                    new_dial.append(' '.join([w for w in new_utt if w != '[DEL]']))
                    if print_example:
                        print(utt)
                        print(new_dial[-1])
                        print()
                print_example = False
                dial['dialogue'] = [' '.join(x) for x in dial['dialogue']]
                dial['masked_dialogue'] = new_dial
                del dial['spans']
                del dial['bio_tag']
                del dial['intent']
            json.dump(data, open(os.path.join(data_dir, dataset + '_data_{}_biomasked.json'.format(phase)), 'w'),
                      indent=2)
