import numpy as np
import json
import os
from collections import Counter
import matplotlib.pyplot as plt


def cal_statistics(datasets):
    turn2tokens = {}
    turn2cnt = []
    context_lens = []
    token_range2turns = {">128": [], "<=128": [], ">256": [], "<=256": [], ">512": [], "<=512": []}
    token_range2len = {">128": [], "<=128": [], ">256": [], "<=256": [], ">512": [], "<=512": []}
    samples = 0
    for dataset in datasets:
        data = json.load(open(os.path.join('processed_data', dataset + '_data_dev.json')))
        for sess in data:
            num_utt = sess['num_utt']
            dial = sess['dialogue']
            turn_len = [len(t)+2 for t in dial]  # [USR]/[SYS] + [SEP]
            assert num_utt % 2 == 1
            # turn2cnt[num_utt] += 1
            for i in range(1, num_utt+1, 2):

                # turn2tokens.setdefault(i, [])
                # turn2tokens[i].append(sum(turn_len[:i]))
                context_len = sum(turn_len[:i])
                context_lens.append(context_len)
                turn2cnt.append(i)
                # if context_len >= 128:
                #     token_range2turns[">128"].append(i)
                #     token_range2len[">128"].append(context_len)
                #     for j in range(0,i,2):
                #         c_len = sum(turn_len[j:i])
                #         if c_len < 128:
                #             token_range2len["<=128"].append(c_len)
                #             token_range2turns["<=128"].append(i - j)
                #             break
                #     else:
                #         assert 0
                # else:
                #     token_range2turns["<=128"].append(i)
                #     token_range2len["<=128"].append(context_len)
                if context_len >= 256:
                    samples += 1
                    token_range2turns[">256"].append(i)
                    token_range2len[">256"].append(context_len)
                    for j in range(0,i,2):
                        c_len = sum(turn_len[j:i])
                        if c_len < 256:
                            token_range2len["<=256"].append(c_len)
                            token_range2turns["<=256"].append(i-j)
                            break
                    else:
                        assert 0, print(dataset,dial[:i])
                else:
                    token_range2turns["<=256"].append(i)
                    token_range2len["<=256"].append(context_len)
                # if context_len >= 512:
                #     token_range2turns[">512"].append(i)
                #     token_range2len[">512"].append(context_len)
                #     for j in range(0,i,2):
                #         c_len = sum(turn_len[j:i])
                #         if c_len < 512:
                #             token_range2len["<=512"].append(c_len)
                #             token_range2turns["<=512"].append(i - j)
                #             break
                #     else:
                #         assert 0
                # else:
                #     token_range2turns["<=512"].append(i)
                #     token_range2len["<=512"].append(context_len)
                # for j in range(0, i, 2):
                #     turn2tokens.setdefault(i-j, [])
                #     turn2tokens[i-j].append(sum(turn_len[j:i]))
                #     break
            # break
    # plt.hist(context_lens, bins=list(range(0, 612, 50)), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.hist(token_range2len["<=512"], bins=list(range(0, 600, 50)), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.hist(token_range2turns["<=512"], bins=list(range(31)), facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(token_range2len["<=256"], bins=list(range(0, 276, 25)), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.hist(token_range2turns["<=256"], bins=list(range(31)), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.hist(token_range2len["<=128"], bins=list(range(0, 150, 10)), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.hist(token_range2turns["<=128"], bins=list(range(31)), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.hist(turn2cnt, bins=list(range(0, 20, 1)), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.plot()
    plt.show()
    print(samples)


def basic_stat(datasets, data_dir):
    total_dial = 0
    total_utt = 0
    for dataset in datasets:
        num_dial = 0
        num_utt = 0
        avg_tokens = []
        avg_utt = []
        train_data = json.load(open(os.path.join(data_dir, dataset + '_data_train.json')))
        dev_data = json.load(open(os.path.join(data_dir, dataset + '_data_dev.json')))
        for data in [train_data, dev_data]:
            for d in data:
                num_dial += 1
                num_utt += d['num_utt']
                avg_utt.append(d['num_utt'])
                for t in d['dialogue']:
                    avg_tokens.append(len(t))
        total_dial += num_dial
        total_utt += num_utt
        print('dataset:',dataset)
        print('\t num_dial', num_dial)
        print('\t num_utt', num_utt)
        print('\t avg_tokens %.2f(%.2f)' % (np.mean(avg_tokens),np.std(avg_tokens)))
        print('\t avg_utt %.2f(%.2f)' % (np.mean(avg_utt),np.std(avg_utt)))
        print()
    print('total_dial', total_dial)
    print('total_utt', total_utt)


if __name__ == '__main__':
    dataset_names = [
        'camrest',
        'multiwoz25',
        'woz',
        'schema',
        'metalwoz',
        'taskmaster',
        'dstc2',
        'frames',
        'kvret',
        'm2m',
        'mdc'
    ]
    data_dir = 'full_dialog'
    # data_dir = 'prefix_dialog'
    basic_stat(dataset_names, data_dir)
