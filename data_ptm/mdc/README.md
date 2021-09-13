# MSR-E2E (MDC)

## Data source

+ publication: 

  Microsoft Dialogue Challenge: Building End-to-end Task-completion Dialogue Systems. 2018.

  https://arxiv.org/pdf/1807.11125.pdf

+ Download:

  https://github.com/xiul-msr/e2e_dialog_challenge



## Data Description

+ Annotations: 

  :x: Dialog Act

  :x: Belief State

+ Statistics:

  |            | dialogues | utterances | Avg. turns | Avg. tokens |
  | ---------- | --------- | ---------- | ---------- | ----------- |
  | movie      | 2890      | 19164      | 6.42       | 12.0        |
  | restaurant | 4103      | 25673      | 6.43       | 11.5        |
  | taxi       | 3094      | 20614      | 6.53       | 10.2        |
  | all        | 10087     | 65451      | 6.49       | 11.4        |



## Main Changes From Original Data

+ Delete the annotation of domain, time, Dialog-Acts. 
+ Correct some error of “user” and “agent” annotation.
+ If the end speaker is the assistant in the origin dialogue, this utterance will be cut off.
+ If there exist more than one utterances in a row, the utterances are concatenate.

