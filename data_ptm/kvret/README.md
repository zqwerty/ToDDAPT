# SMD(KVRET)

## Data source

+ publication: 

  Key-Value Retrieval Networks for Task-Oriented Dialogue. 2018. In *Proceedings of SIGDIAL*.

  https://arxiv.org/abs/1705.05414

+ Download:

  https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/



## Data Description

+ Annotations: 

  :x: Dialog Act

  :x: Belief State

+ Statistics:

  |            | Dialogues | Utterances | Avg. turns | Avg. tokens |
  | ---------- | --------- | ---------- | ---------- | ----------- |
  | schedule   | 1034      | 3096       | 2.99       | 11.5        |
  | weather    | 996       | 4394       | 4.41       | 10.9        |
  | navigation | 1000      | 5554       | 5.55       | 12.4        |
  | all        | 3030      | 13044      | 4.30       | 11.4        |

  |       | Dialogues | Utterances | Avg. turns | Avg. tokens |
  | ----- | --------- | ---------- | ---------- | ----------- |
  | train | 2424      | 10434      | 4.30       | 8.90        |
  | valid | 302       | 1276       | 4.23       | 8.85        |
  | test  | 304       | 1334       | 4.39       | 8.53        |



## Main Changes From Original Data

+ Delete the annotation of domain, slot, value, requested and knowledge base (kb).  
+ If the end speaker is the assistant in the origin dialogue, this utterance will be cut off.
+ If there exist more than one utterances in a row, the utterances are concatenated.  

