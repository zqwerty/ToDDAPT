import json
import random
import os
from zipfile import ZipFile

def preprocess():
    dataset = "kvret"
    self_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(self_dir)), 'data')
    # path = os.path.join(DATA_PATH, dataset)
    path = self_dir

    format_all_data = []
    did = 0
    # path = "../data/kvret"
    with ZipFile(os.path.join(path, "original_data.zip")) as z:
        z.extractall(path)
    file_names = ["kvret_train_public.json", "kvret_dev_public.json", "kvret_test_public.json"]
    splits = ["train", "val", "test"]

    dn_dial = {"schedule": 0, "navigate": 0, "weather": 0}
    dn_utt = {"schedule": 0, "navigate": 0, "weather": 0}
    dn_avg_turn = {"schedule": 0, "navigate": 0, "weather": 0}
    dn_avg_tokens = {"schedule": 0, "navigate": 0, "weather": 0}

    for file_name, split in zip(file_names, splits):
        with open(os.path.join(path, file_name), "r") as f:
            format_data = []
            jobj = json.load(f)

            n_utt = 0
            avg_turns = 0
            avg_tokens = 0

            for dial in jobj:
                domain = dial["scenario"]["task"]["intent"]
                dial = dial["dialogue"]

                if len(dial) == 0:
                    continue

                dn_dial[domain] += 1
                data = {
                    "dataset": "kvret",
                    "data_split": split,
                    "dialogue_id": "kvret_" + str(did),
                    "domains": [],
                    "turns": []
                }
                n_tokens = 0
                uid = 0
                for i, utt in enumerate(dial):
                    if len(data["turns"]) > 0 and (utt["turn"], data["turns"][-1]["speaker"]) in [("driver", "user"), ("assistant", "system")]:
                        data["turns"][-1]["utterance"] += " " + utt["data"]["utterance"]
                        continue

                    data["turns"].append({
                        "speaker": "user" if utt["turn"] == "driver" else "system",
                        "utterance": utt["data"]["utterance"],
                        "utt_idx": uid,
                        "dialogue_act": {
                            "categorical": [],
                            "non-categorical": [],
                            "binary": []
                        },
                        "state": {},
                        "state_update": {
                            "categorical": [],
                            "non-categorical": [],
                        }
                    })

                    uid += 1
                    n_utt += 1
                    n_tokens += len(utt["data"]["utterance"].strip().split(" "))

                    dn_utt[domain] += 1
                    dn_avg_tokens[domain] += len(utt["data"]["utterance"].strip().split(" "))
                
                sp = data["turns"][-1]["speaker"]
                while sp == "system":
                    uid -= 1
                    n_utt -= 1
                    n_tokens -= len(data["turns"][-1]["utterance"].strip().split(" "))

                    dn_utt[domain] -= 1
                    dn_avg_tokens[domain] -= len(data["turns"][-1]["utterance"].strip().split(" "))

                    data["turns"].pop()
                    sp = data["turns"][-1]["speaker"]

                dn_avg_turn[domain] += uid // 2

                avg_turns += uid // 2
                avg_tokens += n_tokens               

                did += 1

                format_data.append(data)
            
            random.shuffle(format_data)
            format_all_data.extend(format_data)

    for key in dn_avg_tokens.keys():
        dn_avg_tokens[key] /= dn_dial[key]
        dn_avg_turn[key] /= dn_dial[key]

    with open("data.json", "w") as f:
        json.dump(format_all_data, f, indent=4)

    with ZipFile("data.zip", "w") as z:
        z.write("data.json")

    onto = {
        "domains": {},
        "intents": {},
        "binary_dialogue_act": [],
        "state": {}
    }
    with open("ontology.json", "w") as f:
        json.dump(onto, f, indent=4)

    return format_all_data, onto


if __name__ == "__main__":
    preprocess()
