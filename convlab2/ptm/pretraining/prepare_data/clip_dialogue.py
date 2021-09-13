import json
import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    # prefix dialogue, cut to <= 59 utts (max turn id is 30)
    # write 823581 dials, 7174037 utts
    # without multiwoz: 766831 dials, 6731659 utts
    dataset_names = [
        # 'mitmovie',
        # 'mitrestaurant',
        # 'facebook',
        # 'camrest',
        # 'woz',
        # 'kvret',
        # 'dstc2',
        # 'frames',
        # 'm2m',
        # 'mdc',
        # 'multiwoz21',
        # 'multiwoz25',
        'schema',
        # 'metalwoz',
        # 'taskmaster',
    ]
    from_data_dir = 'full_dialog'
    cut_T = 59
    to_data_dir = 'prefix_dialog_cut{}'.format(cut_T)
    suffix = '_data_{}.json'
    if not os.path.exists(to_data_dir):
        os.mkdir(to_data_dir)
    for dataset in dataset_names:
        shutil.copy2(os.path.join(from_data_dir, dataset+'_ontology.json'), to_data_dir)
        print('load {} dataset:'.format(dataset))
        for phase in ['dev', 'train']:
            data = json.load(open(os.path.join(from_data_dir, dataset + suffix.format(phase))))
            new_data = []
            for dial in tqdm(data):
                T = len(dial['dialogue'])

                for t in range(1, T + 1, 2):
                    # cut to <= 30 turns
                    if t > cut_T:
                        t0 = t - cut_T
                        # print('exceed max turn', t, 'cut to [{}, {}]'.format(t0, t))
                    else:
                        t0 = 0
                    new_data.append({
                        'num_utt': t - t0,
                        'dialogue': dial['dialogue'][t0:t],
                        'da_list': dial['da_list'][t0:t],
                        'spans': dial['spans'][t0:t],
                        'dataset': dataset,
                        'bio_tag': dial['bio_tag'][t0:t],
                        'intent': dial['intent'][t0:t],
                    })

                # for t in range(2, T + 1, 2):
                #     # cut to <= 30 turns
                #     if T - t > cut_T:
                #         t0 = t + cut_T
                #         print('exceed max turn', t, 'cut to [{}, {}]'.format(t, t0))
                #     else:
                #         t0 = T
                #     new_data.append({
                #         'num_utt': t0 - t,
                #         'dialogue': dial['dialogue'][t:t0],
                #         'spans': dial['spans'][t:t0],
                #         'dataset': dataset,
                #         'bio_tag': dial['bio_tag'][t:t0],
                #         'intent': dial['intent'][t:t0]
                #     })
            json.dump(new_data, open(os.path.join(to_data_dir, dataset + suffix.format(phase)), 'w'),
                      indent=2)
