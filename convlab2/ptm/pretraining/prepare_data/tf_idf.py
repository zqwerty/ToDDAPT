import json
import os
import numpy as np
from collections import OrderedDict, Counter


def rescale_tf_idf(tf_idf, max_range=10):
    flat_tf_idf = [x for utt in tf_idf for x in utt]
    _min, _max = min(flat_tf_idf), max(flat_tf_idf)
    scaling = lambda x: ((max_range) * (x - _min) / (_max - _min + 1e-8))
    scaled_tf_idf = []
    for utt in tf_idf:
        scaled_tf_idf.append([])
        for x in utt:
            scaled_tf_idf[-1].append(scaling(x))
    return scaled_tf_idf


if __name__ == '__main__':
    dataset_names = [
        # 'camrest',
        # 'multiwoz25',
        # 'woz',
        'schema',
        # 'metalwoz',
        # 'taskmaster',
        # 'dstc2',
        # 'frames',
        # 'kvret',
        # 'm2m',
        # 'mdc'
    ]
    data_dir = 'full_dialog'
    # all_idfs = {}
    # all_dial = 0
    for dataset in dataset_names:
        print('load {} dataset:'.format(dataset))
        if not os.path.exists(os.path.join(data_dir, dataset + '_idf.json')):
            print('calculate idf, dump to {}_idf.json'.format(dataset))
            idfs = {}
            total_dial = 0
            for phase in ['dev', 'train']:
                data = json.load(open(os.path.join(data_dir, dataset + '_data_{}.json'.format(phase))))
                total_dial += len(data)
                # if dataset != 'multiwoz25':
                #     all_dial += len(data)
                for dial in data:
                    all_token_set = set([word for utt in dial['dialogue'] for word in utt])
                    for word in all_token_set:
                        idfs[word] = idfs.get(word, 0) + 1
                        # if dataset != 'multiwoz25':
                        #     all_idfs[word] = all_idfs.get(word, 0) + 1
            idfs = OrderedDict(sorted({k: np.log(float(total_dial) / float(1 + idfs[k])) for k in idfs.keys()}.items(),
                                      key=lambda x: x[1]))
            json.dump(idfs, open(os.path.join(data_dir, dataset + '_idf.json'), 'w'), indent=2)
        else:
            print('load idf from {}_idf.json'.format(dataset))
            idfs = json.load(open(os.path.join(data_dir, dataset + '_idf.json')))

        print('calculate tf-idf, dump to {}_tf_idf.json')
        for phase in ['dev', 'train']:
            data = json.load(open(os.path.join(data_dir, dataset + '_data_{}.json'.format(phase))))
            for dial in data:
                tf = Counter([word for utt in dial['dialogue'] for word in utt])
                tf_idf = []
                for utt in dial['dialogue']:
                    tf_idf.append([])
                    for word in utt:
                        tf_idf[-1].append(tf[word]*idfs[word])
                    # print(list(zip(utt, tf_idf[-1])))
                dial['tf_idf'] = rescale_tf_idf(tf_idf)
            json.dump(data, open(os.path.join(data_dir, dataset + '_data_{}_tf_idf.json'.format(phase)), 'w'),
                      indent=2)
    # all_idfs = OrderedDict(
    #     sorted({k: np.log(float(all_dial) / float(1 + all_idfs[k])) for k in all_idfs.keys()}.items(),
    #            key=lambda x: x[1]))
    # json.dump(all_idfs, open(os.path.join(data_dir, 'all_idf.json'), 'w'), indent=2)
    # all_idfs = json.load(open(os.path.join(data_dir, 'all_idf.json')))
    # for dataset in dataset_names:
    #     print('load {} dataset:'.format(dataset))
    #     print('calculate tf-idf, dump to {}_tf_allidf.json')
    #     for phase in ['dev', 'train']:
    #         data = json.load(open(os.path.join(data_dir, dataset + '_data_{}.json'.format(phase))))
    #         for dial in data:
    #             tf = Counter([word for utt in dial['dialogue'] for word in utt])
    #             tf_idf = []
    #             for utt in dial['dialogue']:
    #                 tf_idf.append([])
    #                 for word in utt:
    #                     tf_idf[-1].append(tf[word] * all_idfs[word])
    #                 # print(list(zip(utt, tf_idf[-1])))
    #             dial['tf_idf'] = rescale_tf_idf(tf_idf)
    #         json.dump(data, open(os.path.join(data_dir, dataset + '_data_{}_tf_allidf.json'.format(phase)), 'w'),
    #                   indent=2)




