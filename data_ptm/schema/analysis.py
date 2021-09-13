import json
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from functools import reduce
import random
from convlab2.util.file_util import read_zipped_json
from copy import deepcopy

da_utt = defaultdict(list)
da2delexutt = defaultdict(list)

random.seed(981217)

# jobj = read_zipped_json('data.zip', 'data.json')
with open("data.json", "r") as f:
    jobj = json.load(f)

num_da = 0
num_utt = 0

# all_utt = []

placeholder_char = '√'

for dial in tqdm(jobj):
    turns = dial["turns"]
    for turn in turns:
        utt_da = turn["dialogue_act"]
        utt = turn["utterance"]
        # all_utt.append(utt["utterance"])
        da = [turn["speaker"]]
        for x in chain(utt_da["binary"], utt_da["categorical"]):
            tmp_da = str((x["intent"], x["domain"], x["slot"], x["value"]))
            da.append(tmp_da)
            num_da += 1
        span_cnt = defaultdict(int)
        value_cnt = defaultdict(int)
        for x in utt_da["non-categorical"]:
            tmp_da = str((x["intent"], x["domain"], x["slot"], x["value"]))
            span_cnt[tmp_da] += 1
            da.append(tmp_da)
            num_da += 1

        da = "-".join(sorted(da))
        da_utt[da].append(utt)

        delex_utt = utt
        spans = sorted(utt_da["non-categorical"], key=lambda x: x['start'])
        assert placeholder_char not in delex_utt, print(delex_utt)
        for x in spans:
            start, end = x['start'], x['end']
            delex_utt = delex_utt[:start] + placeholder_char * (end - start) + delex_utt[end:]
        for x in spans:
            start, end = x['start'], x['end']
            k = '[{}]'.format('-'.join([x["intent"], x["domain"], x["slot"]]))
            delex_utt = delex_utt.replace(placeholder_char * (end - start), k, 1)
        da2delexutt[da].append(delex_utt)
        # print(da, copy_utt)

        num_utt += 1
# single utterance
print("mean_da_num", num_da / num_utt)

with open("map.json", "w") as f:
    json.dump(dict(da_utt), f, indent=4)
with open("delex_map.json", "w") as f:
    json.dump(dict(da2delexutt), f, indent=4)
# exit(1)
da_utt = dict(da_utt)

map_len = {x: len(y) for x, y in da_utt.items()}

print(len(map_len))
print(max(list(map_len.values())))
print(min(list(map_len.values())))
print(sum([x for x in map_len.values() if x == 1]))

with open("map_len.json", "w") as f:
    json.dump(map_len, f, indent=4)

# with open("all_utt.json", "w") as f:
#     json.dump(all_utt, f, indent=4)

# multiple utterance
use_delex_utt = True
output_file_name = "aug_{}.json" if not use_delex_utt else "delex_aug_{}.json"
N = 2
aug = [[] for _ in range(N)]
for i in range(N):
    for dial in tqdm(jobj):
        turns = dial["turns"]
        aug[i].append([])
        for turn in turns:
            utt_da = turn["dialogue_act"]
            da = [turn["speaker"]]
            for x in chain(utt_da["binary"], utt_da["categorical"], utt_da["non-categorical"]):
                tmp_da = str((x["intent"], x["domain"], x["slot"]))
                da.append(tmp_da)
            da = "-".join(sorted(da))
            new_utt = turn["utterance"]
            if len(da_utt[da]) > 1:
                if not use_delex_utt:
                    idx = da_utt[da].index(turn["utterance"])
                    assert idx >= 0
                    da_utt[da].pop(idx)
                    new_utt = random.choice(da_utt[da])
                    da_utt[da].append(turn["utterance"])
                else:
                    delex_utt = new_utt
                    spans = sorted(utt_da["non-categorical"], key=lambda x: x['start'])
                    assert placeholder_char not in delex_utt, print(delex_utt)
                    for x in spans:
                        start, end = x['start'], x['end']
                        delex_utt = delex_utt[:start] + placeholder_char * (end - start) + delex_utt[end:]
                    for x in spans:
                        start, end = x['start'], x['end']
                        k = '[{}]'.format('-'.join([x["intent"], x["domain"], x["slot"]]))
                        delex_utt = delex_utt.replace(placeholder_char * (end - start), k, 1)
                    idx = da2delexutt[da].index(delex_utt)
                    assert idx >= 0
                    da2delexutt[da].pop(idx)
                    new_utt = random.choice(da2delexutt[da])
                    da2delexutt[da].append(delex_utt)

                    # assert placeholder_char not in delex_utt, print(delex_utt)
                    for x in spans:
                        k = '[{}]'.format('-'.join([x["intent"], x["domain"], x["slot"]]))
                        v = x['value']
                        new_utt = new_utt.replace(k, v, 1)

            aug[i][-1].append(new_utt)

    with open(output_file_name.format(i), "w") as f:
        json.dump(aug[i], f, indent=4)
