# README

## Features

- Annotations: dialogue act, character-level span for non-categorical slots. state and state updates.   

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | \# domains |
| ----- | ------------ | ------------- | ---------- | ----------- | ---------- |
| train | 8434         | 105066         | 12.46     | 15.75      | 7          |
| dev | 999         | 13731         | 13.74      | 16.1       | 7          |
| train | 1000         | 13744         | 13.74       | 16.08       | 7          |

## updated 7.30
   
- remove `empty value` from possible value lists
- state updates consider `empty value`, updates from some value to `empty` are set to value `''`


## Main changes

- only hotel-parking and hotel-internet are considered categorical slots.
- `dontcare` has no span annotations.
- only keep 5 domains in state annotations and dialog acts. 
- `booking` domain appears in dialog acts but not in states.
- some dialogs contain categorical slot change from `value` to empty value, and some of them are correct while some wrong. 



## annotation check


### dialogue act annotation errors

- some typical wrong annotations are listed below, among which typo errors are most significant.

| error type  | original dialog id | turn id |  wrong annotation (slot & value) | original utterance |
| ------------| -------------------| --------|  -----------------| -------------------|
| wrong anno  |  `MUL2168.json`   |    1     |  `Choice 000`     | `There are over 1,000 trains like that .   Where will you be departing from ?`|
|             |  `PMUL1454.json`  |    2     |  `Type churchills college` | `How about a college to visit instead ?`|
| typo        |  `MUL1790.json`   |    0     |  `Dest peterborough` | `I need to go from Cambridge to Peterbourgh Saturday .`|
|             |  `PMUL2203.json`  |     8    |  `Type night club` | `I 'd also like a nightclub to go to in the same area as the restaurant .` |
| expression  |  `PMUL2049.json`  |     2    |  `Type swimming pool` | `Parks for kids or a water slide`|
|             |  `MUL1490.json`   |     0    |  `Price moderate`  |  `I am looking for a restaurant that serves Modern European food at a reasonable price .`|
| missing     |  `PMUL1419.json`  |     7    |  `Depart Bishops Stortford` |  `Train TR0798 leaves Bishops Stortford at 9:29 and arrives in Cambridge at 10:07 . Would you like me to make a reservation ?`|
| time slot   |  `PMUL3486.json`  |     4    |  `Leave 17:30`     | `I would like to leave after 17;15 on sunday .`|

- some `span_info` annotations can't match `dialog_act` annotations. totally 112 such errors.

    - `MUL1618.json` turn 0
    ``` {
        "text": "Need a train leaving monday to bishops stortford .",
        "metadata": {},
        "dialog_act": {
          "Train-Inform": [
            [
              "Day",
              "monday"
            ],
            [
              "Depart",
              "bishops stortford"
            ]
          ]
        },
        "span_info": [
          [
            "Train-Inform",
            "Dest",
            "bishops stortford",
            6,
            7
          ],
          [
            "Train-Inform",
            "Day",
            "monday",
            4,
            4
          ],
          [
            "Train-Inform",
            "Depart",
            "bishops stortford",
            6,
            7
          ]
        ]
      }
    ```
  
  - wrong span annotation: `SNG0285.json` turn 7:
  ```json
    {
        "text": "No , I 'm sorry , it does n't appear that those trains run on Sundays .   Is there a different day you could travel ?   Also , will you leave from Broxbourne or Cambridge ?",
        "dialog_act": {
          "Train-NoOffer": [
            [
              "Day",
              "Sundays"
            ]
          ],
          "Train-Select": [
            [
              "Depart",
              "Broxbourne"
            ],
            [
              "Depart",
              "Cambridge"
            ]
          ],
          "Train-Inform": [
            [
              "Depart",
              "cambridge"
            ]
          ]
        },
        "span_info": [
          [
            "Train-Select",
            "Depart",
            "Cambridge",
            34,
            34
          ],
          [
            "Train-Inform",
            "Depart",
            "cambridge",
            35,
            35
          ]
        ]
      }    



### state annotation errors

- typos e.g `museum` and `museums`

### original data

- multiwoz2.5 by Huawei
- multiwoz original train/val/test split
- slot description by multiwoz2.2
- some hand-written descriptions. 


