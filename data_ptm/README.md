# Pretraining Model for Task-oriented Dialog

## Unified data format with example

Under `data_ptm` directory.

single turn->dialogue with one turn

Each dataset have at least 4 files:

- `README.md`: dataset description and the main changes from original data to processed data.
- `preprocess.py`: python script that preprocess the data. By running `python preprocess.py` we can get the following two files. The structure `preprocess.py`  should be:

```python
def preprocess():
    pass
if __name__ == '__main__':
    preprocess()
```



- `ontology.json`: dataset ontology, contains descriptions, state definition, etc.
- `data.json.zip`: contains `data.json`.

### README

- Data source: publication, original data download link, etc.
- Data description:
  - Annotations: whether have dialogue act, belief state annotation.
  - Statistics: \# domains, # dialogues, \# utterances, Avg. turns, Avg. tokens (split by space), etc.
- Main changes from original data to processed data.

### Ontology

`ontology.json`: a *dict* containing:

- `domains`: (*dict*) descriptions for domains, slots. Must contains all slots in the state and non-binary dialogue acts.
  - `$domain_name`: (*dict*)
    - `description`: (*str*) description for this domain.
    - `slots`: (*dict*)
      - `$slot_name`: (*dict*)
        - `description`: (*str*) description for this slot.
        - `is_categorical`: (*bool*) categorical slot or not.
        - `possible_values`: (*list*) List of possible values the slot can take. If the slot is a categorical slot, it is a complete list of all the possible values. If the slot is a non categorical slot, it is either an empty list or a small sample of all the values taken by the slot.

- `intents`: (*dict*) descriptions for intents.
  - `$intent_name`: (*dict*)
    - `description`: (*str*) description for this intent.
- `binary_dialogue_act`: (*list* of *dict*) special dialogue acts that the value may not present in the utterance, e.g. request the address of a hotel.
  - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. domain, slot, value may be empty.
- `state`: (*dict*) belief state of all domains.
  - `$domain_name`: (*dict*)
    - `$slot_name: ""`: slot with empty value. Note that the slot set are the subset of the slot set in Part 1 definition.

### Dialogues

`data.json`: a *list* of dialogues containing:

- `dataset`: (*str*) dataset name, must be one of  ['schema', 'multiwoz', 'camrest', 'woz', ...], and be the same as the current dataset.
- `data_split`: (*str*) in [train, val, test].
- `dialogue_id`: (*str*) use dataset name as prefix, add count.
- `domains`: (*list*) domains in this dialogue.
- `turns`: (*list* of *dict*)
  - `speaker`: (*str*) "user" or "system". **User side first, user side final**, "user" and "system" appear alternately?
  - `utterance`: (*str*) sentence.
  - `utt_idx`: (*int*) `turns['utt_idx']` gives current turn.
  - `dialogue_act`: (*dict*)
    - `categorical`: (*list* of *dict*) for categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. Value sets are defined in the ontology.
    - `non-categorical` (*list* of *dict*) for non-categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str), "start": (int), "end": (int)}`. `start` and `end` are character indexes for the value span.
    - `binary` (*list* of *dict*) for binary dialogue acts in ontology.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. Possible dialogue acts are listed in the `ontology['binary_dialogue_act']`.
  - `state`: (*dict*, optional, user side) full state are shown in `ontology['state']`.
    - `$domain_name`: (*dict*) contains all slots in this domain.
      - `$slot_name`: (*str*) value for this slot.
  - `state_update`: (*dict*, optional, user side) records the difference of states between the current turn and the last turn.
    - `categorical`: (*list* of *dict*) for categorical slots.
      - `{"domain": (str), "slot": (str), "value": (str)}`. Value sets are defined in the ontology (**dontcare** may not be included).
    - `non-categorical` (*list* of *dict*) for non-categorical slots.
      - `{"domain": (str), "slot": (str), "value": (str), "utt_idx": (int), "start": (int), "end": (int)}`. `utt_idx` is the utterance index of the value. `start` and `end` are character indexes for the value span in the current turn. `turn[utt_idx]['utterance'][start:end]` gives the value.

Other attributes are optional.

## Issues

- span contains trailing spaces

- schema

  - `dontcare` not present in possible values

- multiwoz:

  - redundant slot description in state
  - dialogue act type (categorical or non-categorical) mismatch
  - dialogue id format

- woz:

  - dialog act is empty in system role
  - `dontcare` value with span (0, 0), e.g.: 

  ```json
  {
      "intent": "inform", 
      "domain": "restaurant", 
      "slot": "food", 
      "value": "dontcare", 
      "start": 0, 
      "end": 0
  }
  ```

  - data split `validate`
  - dialogue id format
  - `domains` is *str* type, expected *list*

- camrest:

  - 3 values are not found in original utterances due to typos. 
  - `dontcare` value in non-categorical slots are calculated as missing during evaluation.
  
- taskmaster

  - no `state` in ontology
  - empty intent set, null intent everywhere
  - Continuous speech:![](https://i.imgur.com/7v1KwoV.jpg)
  - dialogue with zero turn 
  - id format: get `Taskmaster` while `taskmaster` expected
  - split: `all`

- metalwoz

  - ontology: no slots nor dialogue acts, can only be used as corpus for MLM

### Fixed

- `dialogue` and `dialog`

## Example of Schema

```json
{
  "dataset": "schema",
  "data_split": "val",
  "dialogue_id": "schema_17555",
  "original_id": "5_00022",
  "domains": [
    "travel_1"
  ],
  "turns": [
    {
      "speaker": "user",
      "utterance": "Can you tell me about some good attractions that are near me?",
      "utt_idx": 0,
      "dialogue_act": {
        "binary": [
          {
            "intent": "inform_intent",
            "domain": "travel_1",
            "slot": "intent",
            "value": "FindAttractions"
          }
        ],
        "categorical": [],
        "non-categorical": []
      },
      "state": {
        "travel_1": {
          "location": "",
          "attraction_name": "",
          "category": "",
          "phone_number": "",
          "free_entry": "",
          "good_for_kids": ""
        }
      },
      "state_update": {
        "categorical": [],
        "non-categorical": []
      }
    },
    {
      "speaker": "system",
      "utterance": "Where are you right now?",
      "utt_idx": 1,
      "dialogue_act": {
        "binary": [
          {
            "intent": "request",
            "domain": "travel_1",
            "slot": "location",
            "value": ""
          }
        ],
        "categorical": [],
        "non-categorical": []
      }
    },
    {
      "speaker": "user",
      "utterance": "I'm in London and I want to see something big.",
      "utt_idx": 2,
      "dialogue_act": {
        "binary": [],
        "categorical": [],
        "non-categorical": [
          {
            "intent": "inform",
            "domain": "travel_1",
            "slot": "location",
            "value": "London",
            "start": 7,
            "end": 13
          }
        ]
      },
      "state": {
        "travel_1": {
          "location": "london",
          "attraction_name": "",
          "category": "",
          "phone_number": "",
          "free_entry": "",
          "good_for_kids": ""
        }
      },
      "state_update": {
        "categorical": [],
        "non-categorical": [
          {
            "domain": "travel_1",
            "slot": "location",
            "value": "london",
            "utt_idx": 2,
            "start": 7,
            "end": 13
          }
        ]
      }
    },
    {
      "speaker": "system",
      "utterance": "There are 10 near you. But, why don't you look at 30 st Mary Axe (The Gherkin). It is really big and a historical landmark.",
      "utt_idx": 3,
      "dialogue_act": {
        "binary": [],
        "categorical": [
          {
            "intent": "offer",
            "domain": "travel_1",
            "slot": "category",
            "value": "Historical Landmark"
          }
        ],
        "non-categorical": [
          {
            "intent": "offer",
            "domain": "travel_1",
            "slot": "attraction_name",
            "value": "30 st Mary Axe (The Gherkin)",
            "start": 50,
            "end": 78
          },
          {
            "intent": "inform_count",
            "domain": "travel_1",
            "slot": "count",
            "value": "10",
            "start": 10,
            "end": 12
          }
        ]
      }
    },
    {
      "speaker": "user",
      "utterance": "Cool, is there anything else without an entry fee? Maybe something wet and slippery.",
      "utt_idx": 4,
      "dialogue_act": {
        "binary": [
          {
            "intent": "request_alts",
            "domain": "",
            "slot": "",
            "value": ""
          }
        ],
        "categorical": [
          {
            "intent": "inform",
            "domain": "travel_1",
            "slot": "free_entry",
            "value": "True"
          }
        ],
        "non-categorical": []
      },
      "state": {
        "travel_1": {
          "location": "london",
          "attraction_name": "",
          "category": "",
          "phone_number": "",
          "free_entry": "true",
          "good_for_kids": ""
        }
      },
      "state_update": {
        "categorical": [
          {
            "domain": "travel_1",
            "slot": "free_entry",
            "value": "true"
          }
        ],
        "non-categorical": []
      }
    },
    {
      "speaker": "system",
      "utterance": "There are 9 others, but maybe you should check out the sports venue Alexandra Palace Ice Rink. It's really wet and slippery.",
      "utt_idx": 5,
      "dialogue_act": {
        "binary": [],
        "categorical": [
          {
            "intent": "offer",
            "domain": "travel_1",
            "slot": "category",
            "value": "Sports Venue"
          }
        ],
        "non-categorical": [
          {
            "intent": "offer",
            "domain": "travel_1",
            "slot": "attraction_name",
            "value": "Alexandra Palace Ice Rink",
            "start": 68,
            "end": 93
          },
          {
            "intent": "inform_count",
            "domain": "travel_1",
            "slot": "count",
            "value": "9",
            "start": 10,
            "end": 11
          }
        ]
      }
    },
    {
      "speaker": "user",
      "utterance": "Okay that's all I wanted.",
      "utt_idx": 6,
      "dialogue_act": {
        "binary": [
          {
            "intent": "select",
            "domain": "travel_1",
            "slot": "",
            "value": ""
          },
          {
            "intent": "goodbye",
            "domain": "",
            "slot": "",
            "value": ""
          }
        ],
        "categorical": [],
        "non-categorical": []
      },
      "state": {
        "travel_1": {
          "location": "london",
          "attraction_name": "",
          "category": "",
          "phone_number": "",
          "free_entry": "true",
          "good_for_kids": ""
        }
      },
      "state_update": {
        "categorical": [],
        "non-categorical": []
      }
    }
  ]
}
```

