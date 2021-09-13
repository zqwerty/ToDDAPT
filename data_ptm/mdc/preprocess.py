import json
import random
import os
import zipfile
from zipfile import ZipFile

def preprocess():
    dataset = "mdc"
    self_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(self_dir)), 'data')
    # path = os.path.join(DATA_PATH, dataset)
    path = self_dir
    
    # path = "../data/mdc"
    
    with ZipFile(os.path.join(path, "original_data.zip"), "r", zipfile.ZIP_DEFLATED) as z:
        z.extractall(path)

    file_names = ["movie_all.tsv", "restaurant_all.tsv", "taxi_all.tsv"]
    domain_names = ["movie", "restaurant", "taxi"]
    format_all_data = []
    did = 0
    for file_name, domain_name in zip(file_names, domain_names):
        all_data = []
        with open(os.path.join(path, file_name), "r", encoding='utf-8') as f:
            lines = f.readlines()[1:]
        lines = [s.strip().split("\t") for s in lines]
        i = 1
        all_data.append([])
        for line in lines:
            if int(line[0]) == i:
                all_data[-1].append(line[3:5])
            else:
                i = int(line[0])
                all_data.append([line[3:5]])
        
        format_data = []

        n_utt = 0
        avg_turns = 0
        avg_tokens = 0
        for dial in all_data:
            data = {
                "dataset": "mdc",
                "data_split": "train",
                "dialogue_id": "mdc_" + str(did),
                # "domains": [domain_name],
                "domains": [],
                "turns": []
            }
            n_tokens = 0
            uid = 0

            if did == 7473:
                dial = dial[1:]

            if did == 2400:
                dial[0][0] = 'user'
                dial[1][0] = 'agent'

            if did == 2386:
                # correct the error
                dial[0][0] = 'user'
            
            for i, utt in enumerate(dial):
                if len(data["turns"]) > 0 and (utt[0], data["turns"][-1]["speaker"]) in [("user", "user"), ("agent", "system")]:
                    data["turns"][-1]["utterance"] += " " + utt[1]
                    continue

                data["turns"].append({
                    "speaker": "user" if utt[0] == "user" else "system",
                    "utterance": utt[1].strip("\" "),
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
                n_tokens += len(utt[1].strip().split(" "))

            sp = data["turns"][-1]["speaker"]
            while sp == "system":
                uid -= 1
                n_utt -= 1
                n_tokens -= len(data["turns"][-1]["utterance"].strip().split(" "))

                data["turns"].pop()
                sp = data["turns"][-1]["speaker"]
            
            avg_turns += uid // 2
            avg_tokens += n_tokens
            format_data.append(data)
            did += 1

        format_all_data.extend(format_data)

    random.shuffle(format_all_data)

    with open("data.json", "w") as f:
        json.dump(format_all_data, f, indent=4)
    
    with ZipFile("data.zip", "w") as z:
        z.write("data.json")

    onto = {
        "domains": {
            # "movie": {
            #     "description": "",
            #     "slots": {}
            # },
            # "restaurant": {
            #     "description": "",
            #     "slots": {}
            # },
            # "taxi": {
            #     "description": "",
            #     "slots": {}
            # }
        },
        "intents": {},
        "binary_dialogue_act": [],
        "state": {
            # "movie": {},
            # "restaurant": {},
            # "taxi": {}
        }
    } 
    with open("ontology.json", "w") as f:
        json.dump(onto, f, indent=4)

    return format_all_data, onto

if __name__ == "__main__":
    preprocess()
