# README

## Features

- two domains, restaurant and movie
- value annotation for every utterance
- dialog act & state update value match rate 100%

- Annotations: 
    - dialog act, but user dialog act seems incomplete
    - character-level span for non-categorical slots, but user side seems 

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | 
| ----- | ------------ | ------------- | ---------- | ----------- | 
| train | 1500         | 14796         | 9.86       | 8.25       |
| val   | 469          | 3763          | 8.02       | 8.32       |
| test  | 1039         | 8561          | 8.24       | 8.34       |

## Main changes

- manually add descriptions in ontology (some are copy from other datasets)
- infer incomplete user-side dialog act
- search dialog state's values' span
- calculate dialog act update

## Original data

- https://github.com/google-research-datasets/simulated-dialogue
- [folder](../../data/m2m) in repository
